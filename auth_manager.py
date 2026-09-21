from __future__ import annotations

import hashlib
import logging
import os
import re
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, Optional, Tuple

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError
from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.collection import Collection
from pymongo.errors import DuplicateKeyError, PyMongoError


logger = logging.getLogger(__name__)

ROLES = (
    "viewer",
    "analyst",
    "metadata_editor",
    "metadata_publisher",
    "admin",
)

ROLE_PRIORITY = ("admin", "metadata_publisher", "metadata_editor", "analyst", "viewer")

ROLE_CAPABILITIES = {
    "viewer": {"search"},
    "analyst": {"search", "interactions", "evaluation"},
    "metadata_editor": {"search", "metadata_read", "metadata_draft", "metadata_edit"},
    "metadata_publisher": {
        "search", "metadata_read", "metadata_draft", "metadata_edit", "metadata_publish",
    },
    "admin": {
        "search", "interactions", "evaluation", "metadata_read", "metadata_draft",
        "metadata_edit", "metadata_publish", "user_management",
    },
}

USERNAME_RE = re.compile(r"^[A-Za-z0-9._-]{3,64}$")


class AuthenticationError(ValueError):
    """Raised for an invalid login or invalid password change request."""


class AuthenticationUnavailableError(RuntimeError):
    """Raised when MongoDB cannot safely service an authentication request."""


class AuthorizationError(PermissionError):
    """Raised when a requested account operation is not allowed."""


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp for MongoDB records."""
    return datetime.now(timezone.utc)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class AuthManager:
    """Own MongoDB users, sessions, password hashes, and auth audit events."""

    def __init__(
        self,
        mongo_uri: Optional[str] = None,
        database_name: Optional[str] = None,
        session_days: Optional[int] = None,
    ) -> None:
        self.mongo_uri = mongo_uri or os.environ.get("MONGO_URI")
        self.database_name = database_name or os.environ.get("MONGO_DB_NAME", "semantic_search")
        self.users_collection_name = os.environ.get("MONGO_AUTH_USERS_COLLECTION", "users")
        self.sessions_collection_name = os.environ.get("MONGO_AUTH_SESSIONS_COLLECTION", "auth_sessions")
        self.audit_collection_name = os.environ.get("MONGO_AUDIT_COLLECTION", "audit_logs")
        self.session_days = session_days or self._positive_int_env("AUTH_SESSION_DAYS", 8, maximum=31)

        self.password_hasher = PasswordHasher(time_cost=3, memory_cost=65536, parallelism=2)
        self._client: Optional[MongoClient] = None
        self._users: Optional[Collection] = None
        self._sessions: Optional[Collection] = None
        self._audit: Optional[Collection] = None

    @staticmethod
    def _positive_int_env(name: str, default: int, maximum: int) -> int:
        try:
            value = int(os.environ.get(name, str(default)))
            return min(max(value, 1), maximum)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def auth_enabled() -> bool:
        """Whether route decorators must enforce browser authentication."""
        return os.environ.get("AUTH_ENABLED", "true").strip().lower() == "true"

    def _collections(self) -> Tuple[Collection, Collection, Collection]:
        """Connect lazily and create only the indexes required for auth."""
        if self._users is not None and self._sessions is not None and self._audit is not None:
            return self._users, self._sessions, self._audit
        if not self.mongo_uri:
            raise AuthenticationUnavailableError("MONGO_URI is required for authentication")

        try:
            self._client = MongoClient(
                self.mongo_uri,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
                tz_aware=True,
            )
            self._client.admin.command("ping")
            database = self._client[self.database_name]
            users = database[self.users_collection_name]
            sessions = database[self.sessions_collection_name]
            audit = database[self.audit_collection_name]

            # Case-insensitive normalized keys avoid duplicate users such as
            # "Admin" and "admin".  TTL removes expired sessions automatically.
            users.create_index([("username_normalized", ASCENDING)], unique=True)
            users.create_index([("email_normalized", ASCENDING)], unique=True, sparse=True)
            users.create_index([("role", ASCENDING), ("is_active", ASCENDING)])
            users.create_index([("roles", ASCENDING), ("is_active", ASCENDING)])
            sessions.create_index([("session_hash", ASCENDING)], unique=True)
            sessions.create_index("expires_at", expireAfterSeconds=0)
            sessions.create_index([("user_id", ASCENDING), ("revoked_at", ASCENDING)])
            audit.create_index([("timestamp", DESCENDING)])
            audit.create_index([("actor_id", ASCENDING), ("timestamp", DESCENDING)])
            audit.create_index([("action", ASCENDING), ("timestamp", DESCENDING)])

            self._users, self._sessions, self._audit = users, sessions, audit
            logger.info(
                "MongoDB authentication connected: database=%s, users=%s, sessions=%s",
                self.database_name,
                self.users_collection_name,
                self.sessions_collection_name,
            )
            return users, sessions, audit
        except PyMongoError as exc:
            logger.error("MongoDB authentication connection failed: %s", exc)
            raise AuthenticationUnavailableError("Authentication service is unavailable") from exc

    @staticmethod
    def _normalise_username(username: Any) -> Tuple[str, str]:
        display = str(username or "").strip()
        if not USERNAME_RE.fullmatch(display):
            raise AuthenticationError(
                "Username must contain 3-64 letters, numbers, dots, hyphens, or underscores"
            )
        return display, display.casefold()

    @staticmethod
    def _normalise_email(email: Any) -> Tuple[Optional[str], Optional[str]]:
        if email is None or not str(email).strip():
            return None, None
        display = str(email).strip()
        if len(display) > 254 or "@" not in display or display.startswith("@"):
            raise AuthenticationError("Enter a valid email address")
        return display, display.casefold()

    @staticmethod
    def _validate_roles(roles: Any) -> tuple[str, ...]:
        """Validate one legacy role or a list of selected access roles."""
        if isinstance(roles, str):
            supplied = [roles]
        elif isinstance(roles, (list, tuple, set)):
            supplied = list(roles)
        else:
            raise AuthenticationError("Choose at least one valid access role")

        selected = {str(role or "").strip() for role in supplied}
        selected.discard("")
        if not selected or not selected.issubset(set(ROLES)):
            raise AuthenticationError("Choose at least one valid access role")

        if "admin" in selected:
            return ("admin",)
        return tuple(role for role in ROLES if role in selected)

    @classmethod
    def _document_roles(cls, document: Dict[str, Any]) -> tuple[str, ...]:
        """Read current multi-role documents and old single-role documents."""
        stored_roles = document.get("roles")
        candidate = stored_roles if stored_roles is not None else document.get("role", "viewer")
        try:
            return cls._validate_roles(candidate)
        except AuthenticationError:
            logger.warning("User %s has invalid stored roles", document.get("_id"))
            return ("viewer",)

    @staticmethod
    def _primary_role(roles: Iterable[str]) -> str:
        selected = set(roles)
        return next((role for role in ROLE_PRIORITY if role in selected), "viewer")

    @staticmethod
    def _validate_password(password: Any) -> str:
        value = str(password or "")
        if len(value) < 8:
            raise AuthenticationError("Password must contain at least 8 characters")
        if len(value) > 256:
            raise AuthenticationError("Password is too long")
        return value

    @classmethod
    def public_user(cls, document: Dict[str, Any]) -> Dict[str, Any]:
        """Return only safe account data; never expose a password hash."""
        roles = cls._document_roles(document)
        return {
            "id": str(document.get("_id")),
            "username": document.get("username"),
            "email": document.get("email"),
            "role": cls._primary_role(roles),
            "roles": list(roles),
            "is_active": bool(document.get("is_active", False)),
            "must_change_password": bool(document.get("must_change_password", False)),
            "created_at": document.get("created_at"),
            "updated_at": document.get("updated_at"),
            "last_login_at": document.get("last_login_at"),
        }

    def create_user(
        self,
        username: Any,
        password: Any,
        roles: Any,
        email: Any = None,
        *,
        actor: Optional[Dict[str, Any]] = None,
        must_change_password: bool = True,
    ) -> Dict[str, Any]:
        """Create an account.  This is called only by the CLI or an admin API."""
        display_username, normalized_username = self._normalise_username(username)
        display_email, normalized_email = self._normalise_email(email)
        password_value = self._validate_password(password)
        role_values = self._validate_roles(roles)
        primary_role = self._primary_role(role_values)
        users, _, _ = self._collections()
        now = utc_now()
        document: Dict[str, Any] = {
            "_id": str(uuid.uuid4()),
            "username": display_username,
            "username_normalized": normalized_username,
            "role": primary_role,
            "roles": list(role_values),
            "password_hash": self.password_hasher.hash(password_value),
            "is_active": True,
            "must_change_password": bool(must_change_password),
            "created_at": now,
            "updated_at": now,
            "last_login_at": None,
            "created_by": actor.get("id") if actor else "bootstrap_cli",
        }
        if display_email:
            document["email"] = display_email
            document["email_normalized"] = normalized_email
        try:
            users.insert_one(document)
        except DuplicateKeyError as exc:
            raise AuthenticationError("A user with that username or email already exists") from exc
        except PyMongoError as exc:
            raise AuthenticationUnavailableError("Could not create user") from exc

        self.audit_event(
            actor=actor,
            action="auth.user_created",
            resource_type="user",
            resource_id=document["_id"],
            success=True,
            metadata={"username": display_username, "roles": list(role_values)},
        )
        return self.public_user(document)

    def create_initial_admin(self, username: Any, password: Any, email: Any = None) -> Dict[str, Any]:
        """Create the one bootstrap admin, refusing accidental duplicates."""
        users, _, _ = self._collections()
        if users.count_documents({"role": "admin", "is_active": True}, limit=1):
            raise AuthorizationError("An active admin already exists; use create-user or reset-password instead")
        return self.create_user(
            username,
            password,
            "admin",
            email,
            actor=None,
            must_change_password=False,
        )

    def authenticate(self, username: Any, password: Any) -> Tuple[Dict[str, Any], str, str, datetime]:
        try:
            _, normalized_username = self._normalise_username(username)
        except AuthenticationError:
            raise AuthenticationError("Invalid username or password") from None
        password_value = str(password or "")
        users, sessions, _ = self._collections()
        try:
            user = users.find_one({"username_normalized": normalized_username})
        except PyMongoError as exc:
            raise AuthenticationUnavailableError("Authentication service is unavailable") from exc

        if not user or not user.get("is_active"):
            # Use Argon2 work even for unknown users to reduce timing clues.
            self.password_hasher.hash("invalid-login-placeholder")
            raise AuthenticationError("Invalid username or password")
        try:
            valid = self.password_hasher.verify(user.get("password_hash", ""), password_value)
        except (VerifyMismatchError, VerificationError, InvalidHashError):
            valid = False
        if not valid:
            raise AuthenticationError("Invalid username or password")

        # Transparently upgrade a stored hash if Argon2 parameters are raised.
        if self.password_hasher.check_needs_rehash(user["password_hash"]):
            users.update_one({"_id": user["_id"]}, {"$set": {"password_hash": self.password_hasher.hash(password_value)}})

        now = utc_now()
        expires_at = now + timedelta(days=self.session_days)
        session_token = secrets.token_urlsafe(48)
        csrf_token = secrets.token_urlsafe(32)
        session_doc = {
            "_id": str(uuid.uuid4()),
            "session_hash": _sha256(session_token),
            "csrf_hash": _sha256(csrf_token),
            "user_id": user["_id"],
            "created_at": now,
            "last_seen_at": now,
            "expires_at": expires_at,
            "revoked_at": None,
        }
        try:
            sessions.insert_one(session_doc)
            users.update_one({"_id": user["_id"]}, {"$set": {"last_login_at": now, "updated_at": now}})
            user["last_login_at"] = now
            user["updated_at"] = now
        except PyMongoError as exc:
            raise AuthenticationUnavailableError("Could not create login session") from exc
        return self.public_user(user), session_token, csrf_token, expires_at

    def get_session_user(self, session_token: Optional[str]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Resolve a raw HttpOnly cookie to a safe user and session document."""
        if not session_token:
            return None, None
        users, sessions, _ = self._collections()
        now = utc_now()
        try:
            session = sessions.find_one({
                "session_hash": _sha256(session_token),
                "revoked_at": None,
                "expires_at": {"$gt": now},
            })
            if not session:
                return None, None
            user = users.find_one({"_id": session["user_id"], "is_active": True})
            if not user:
                return None, None
            sessions.update_one({"_id": session["_id"]}, {"$set": {"last_seen_at": now}})
            return self.public_user(user), session
        except PyMongoError as exc:
            raise AuthenticationUnavailableError("Authentication service is unavailable") from exc

    @staticmethod
    def validate_csrf(session: Dict[str, Any], cookie_token: Optional[str], header_token: Optional[str]) -> bool:
        """Require the public CSRF cookie and header to match the session hash."""
        if not cookie_token or not header_token:
            return False
        return secrets.compare_digest(cookie_token, header_token) and secrets.compare_digest(
            _sha256(header_token), str(session.get("csrf_hash", ""))
        )

    def revoke_session(self, session_token: Optional[str]) -> None:
        if not session_token:
            return
        _, sessions, _ = self._collections()
        try:
            sessions.update_one(
                {"session_hash": _sha256(session_token), "revoked_at": None},
                {"$set": {"revoked_at": utc_now()}},
            )
        except PyMongoError as exc:
            raise AuthenticationUnavailableError("Could not log out") from exc

    def revoke_all_user_sessions(self, user_id: str) -> None:
        _, sessions, _ = self._collections()
        try:
            sessions.update_many(
                {"user_id": user_id, "revoked_at": None},
                {"$set": {"revoked_at": utc_now()}},
            )
        except PyMongoError as exc:
            raise AuthenticationUnavailableError("Could not revoke sessions") from exc

    def change_password(self, user_id: str, current_password: Any, new_password: Any) -> None:
        """Change a user's own password and invalidate all old sessions."""
        new_value = self._validate_password(new_password)
        users, _, _ = self._collections()
        user = users.find_one({"_id": user_id, "is_active": True})
        if not user:
            raise AuthenticationError("User account is unavailable")
        try:
            valid = self.password_hasher.verify(user.get("password_hash", ""), str(current_password or ""))
        except (VerifyMismatchError, VerificationError, InvalidHashError):
            valid = False
        if not valid:
            raise AuthenticationError("Current password is incorrect")
        try:
            users.update_one(
                {"_id": user_id},
                {"$set": {
                    "password_hash": self.password_hasher.hash(new_value),
                    "must_change_password": False,
                    "updated_at": utc_now(),
                }},
            )
        except PyMongoError as exc:
            raise AuthenticationUnavailableError("Could not change password") from exc
        self.revoke_all_user_sessions(user_id)

    def list_users(self) -> list[Dict[str, Any]]:
        users, _, _ = self._collections()
        try:
            return [self.public_user(user) for user in users.find().sort("username_normalized", ASCENDING)]
        except PyMongoError as exc:
            raise AuthenticationUnavailableError("Could not list users") from exc

    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        users, _, _ = self._collections()
        user = users.find_one({"_id": user_id})
        return self.public_user(user) if user else None

    def update_user(
        self,
        user_id: str,
        *,
        roles: Any = None,
        role: Any = None,
        email: Any = None,
        is_active: Any = None,
        actor: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Update admin-controlled account fields and revoke on disable."""
        users, _, _ = self._collections()
        target = users.find_one({"_id": user_id})
        if not target:
            raise KeyError("User not found")
        changes: Dict[str, Any] = {"updated_at": utc_now()}
        metadata: Dict[str, Any] = {}
        if roles is not None and role is not None:
            raise AuthenticationError("Send roles, not both role and roles")
        selected_roles = roles if roles is not None else role
        if selected_roles is not None:
            values = self._validate_roles(selected_roles)
            changes["roles"] = list(values)
            # Keep the old field in sync for existing reports and indexes.
            changes["role"] = self._primary_role(values)
            metadata["roles"] = list(values)
        if email is not None:
            display_email, normalized_email = self._normalise_email(email)
            if display_email:
                changes["email"] = display_email
                changes["email_normalized"] = normalized_email
            else:
                changes["email"] = None
                changes["email_normalized"] = None
            metadata["email_changed"] = True
        if is_active is not None:
            if not isinstance(is_active, bool):
                raise AuthenticationError("is_active must be true or false")
            changes["is_active"] = is_active
            metadata["is_active"] = is_active

        if len(changes) == 1:
            return self.public_user(target)
        try:
            users.update_one({"_id": user_id}, {"$set": changes})
            updated = users.find_one({"_id": user_id})
        except DuplicateKeyError as exc:
            raise AuthenticationError("A user with that email already exists") from exc
        except PyMongoError as exc:
            raise AuthenticationUnavailableError("Could not update user") from exc
        if is_active is False:
            self.revoke_all_user_sessions(user_id)
        self.audit_event(
            actor=actor,
            action="auth.user_updated",
            resource_type="user",
            resource_id=user_id,
            success=True,
            metadata=metadata,
        )
        return self.public_user(updated)

    def delete_user(self, user_id: str, *, actor: Optional[Dict[str, Any]] = None) -> None:
        """Permanently remove an account after revoking every active session."""
        users, _, _ = self._collections()
        target = users.find_one({"_id": user_id})
        if not target:
            raise KeyError("User not found")
        if actor and actor.get("id") == user_id:
            raise AuthorizationError("You cannot delete your own account")
        if target.get("is_active") and "admin" in self._document_roles(target):
            active_admins = users.count_documents({
                "is_active": True,
                "$or": [{"roles": "admin"}, {"role": "admin"}],
            })
            if active_admins <= 1:
                raise AuthorizationError("You cannot delete the last active admin account")
        try:
            # Revoke first so no existing session can continue if the account
            # record is removed successfully.
            self.revoke_all_user_sessions(user_id)
            deleted = users.delete_one({"_id": user_id})
        except PyMongoError as exc:
            raise AuthenticationUnavailableError("Could not delete user") from exc
        if not deleted.deleted_count:
            raise KeyError("User not found")
        self.audit_event(
            actor=actor,
            action="auth.user_deleted",
            resource_type="user",
            resource_id=user_id,
            success=True,
            metadata={"username": target.get("username"), "roles": list(self._document_roles(target))},
        )

    def rename_username(
        self,
        current_username: Any,
        new_username: Any,
        *,
        actor: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Rename one account through a server-owner workflow and revoke sessions.

        This is deliberately separate from browser user management: changing a
        sign-in identifier is a sensitive operation and must not leave old
        sessions active after the account name changes.
        """
        _, current_normalized = self._normalise_username(current_username)
        display_new, normalized_new = self._normalise_username(new_username)
        users, _, _ = self._collections()
        target = users.find_one({"username_normalized": current_normalized})
        if not target:
            raise KeyError("User not found")
        try:
            users.update_one(
                {"_id": target["_id"]},
                {"$set": {
                    "username": display_new,
                    "username_normalized": normalized_new,
                    "updated_at": utc_now(),
                }},
            )
            updated = users.find_one({"_id": target["_id"]})
        except DuplicateKeyError as exc:
            raise AuthenticationError("A user with that username already exists") from exc
        except PyMongoError as exc:
            raise AuthenticationUnavailableError("Could not rename user") from exc
        self.revoke_all_user_sessions(target["_id"])
        self.audit_event(
            actor=actor,
            action="auth.username_changed",
            resource_type="user",
            resource_id=target["_id"],
            success=True,
            metadata={"previous_username": target.get("username"), "username": display_new},
        )
        return self.public_user(updated)

    def reset_password(self, user_id: str, new_password: Any, *, actor: Optional[Dict[str, Any]] = None) -> None:
        value = self._validate_password(new_password)
        users, _, _ = self._collections()
        try:
            result = users.update_one(
                {"_id": user_id},
                {"$set": {
                    "password_hash": self.password_hasher.hash(value),
                    "must_change_password": True,
                    "updated_at": utc_now(),
                }},
            )
        except PyMongoError as exc:
            raise AuthenticationUnavailableError("Could not reset password") from exc
        if result.matched_count != 1:
            raise KeyError("User not found")
        self.revoke_all_user_sessions(user_id)
        self.audit_event(
            actor=actor,
            action="auth.password_reset",
            resource_type="user",
            resource_id=user_id,
            success=True,
        )

    def audit_event(
        self,
        *,
        actor: Optional[Dict[str, Any]],
        action: str,
        resource_type: str,
        resource_id: Optional[str] = None,
        success: bool,
        method: Optional[str] = None,
        path: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """Best-effort immutable security event logging; never expose secrets."""
        try:
            _, _, audit = self._collections()
            document = {
                "_id": str(uuid.uuid4()),
                "actor_type": "user" if actor else "anonymous",
                "actor_id": actor.get("id") if actor else None,
                "actor_username": actor.get("username") if actor else None,
                "actor_email": actor.get("email") if actor else None,
                "actor_role": actor.get("role") if actor else None,
                "actor_roles": actor.get("roles") if actor else None,
                "action": action,
                "resource_type": resource_type,
                "resource_id": resource_id,
                "method": method,
                "path": path,
                "status_code": 200 if success else 401,
                "success": bool(success),
                "ip_address": ip_address,
                "user_agent": (user_agent or "")[:500] or None,
                "metadata": metadata or {},
                "error_message": error_message,
                "timestamp": utc_now(),
            }
            audit.insert_one(document)
        except Exception as exc:  # Audit failure must never block authentication.
            logger.warning("Could not write authentication audit event: %s", exc)

    @staticmethod
    def role_has_any(roles: Any, allowed_roles: Iterable[str]) -> bool:
        """Return whether an account has at least one required access role."""
        try:
            selected_roles = AuthManager._validate_roles(roles)
        except AuthenticationError:
            return False
        return bool(set(selected_roles).intersection(set(allowed_roles)))
