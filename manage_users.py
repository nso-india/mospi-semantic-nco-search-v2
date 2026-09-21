from __future__ import annotations

import argparse
import getpass
import sys

from dotenv import load_dotenv

from auth_manager import (
    AuthenticationError,
    AuthenticationUnavailableError,
    AuthManager,
    AuthorizationError,
    ROLES,
)


def _read_new_password() -> str:
    password = getpass.getpass("Password (at least 8 characters): ")
    confirmation = getpass.getpass("Confirm password: ")
    if password != confirmation:
        raise AuthenticationError("Passwords do not match")
    return password


def _print_user(user: dict) -> None:
    print(f"Created user: {user['username']} ({', '.join(user.get('roles') or [user['role']])})")
    if user.get("email"):
        print(f"Email: {user['email']}")


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Manage Semantic Search user accounts")
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_admin = subparsers.add_parser("create-admin", help="Create the first active admin")
    create_admin.add_argument("--username", help="Admin username; prompted when omitted")
    create_admin.add_argument("--email", help="Optional admin email")

    create_user = subparsers.add_parser("create-user", help="Create a user account")
    create_user.add_argument("--username", help="Username; prompted when omitted")
    create_user.add_argument("--email", help="Optional email")
    create_user.add_argument("--role", choices=ROLES, help="Legacy single access role")
    create_user.add_argument(
        "--roles", choices=ROLES, nargs="+",
        help="One or more access roles, for example: --roles analyst metadata_editor",
    )

    reset_password = subparsers.add_parser("reset-password", help="Reset a user's password")
    reset_password.add_argument("username")

    disable_user = subparsers.add_parser("disable-user", help="Disable a user and revoke sessions")
    disable_user.add_argument("username")

    rename_user = subparsers.add_parser(
        "rename-user",
        help="Rename an account and revoke its active sessions",
    )
    rename_user.add_argument("current_username")
    rename_user.add_argument("new_username")

    subparsers.add_parser("list-users", help="List users without password data")
    args = parser.parse_args()
    manager = AuthManager()

    try:
        if args.command == "create-admin":
            username = args.username or input("Admin username: ").strip()
            _print_user(manager.create_initial_admin(username, _read_new_password(), args.email))
            return 0

        if args.command == "create-user":
            username = args.username or input("Username: ").strip()
            selected_roles = args.roles or args.role or "viewer"
            _print_user(manager.create_user(username, _read_new_password(), selected_roles, args.email))
            return 0

        if args.command == "list-users":
            for user in manager.list_users():
                state = "active" if user["is_active"] else "disabled"
                access = ",".join(user.get("roles") or [user["role"]])
                print(f"{user['username']:24} {access:40} {state}")
            return 0

        if args.command == "rename-user":
            user = manager.rename_username(args.current_username, args.new_username)
            print(f"Renamed account to {user['username']}. Existing sessions were revoked.")
            return 0

        username = args.username
        normalized = username.casefold()
        target = next((user for user in manager.list_users() if user["username"].casefold() == normalized), None)
        if not target:
            raise AuthenticationError("User not found")
        if args.command == "reset-password":
            manager.reset_password(target["id"], _read_new_password())
            print(f"Password reset for {target['username']}. They must choose a new password after login.")
            return 0
        if args.command == "disable-user":
            manager.update_user(target["id"], is_active=False)
            print(f"Disabled {target['username']} and revoked their sessions.")
            return 0
    except (AuthenticationError, AuthorizationError, AuthenticationUnavailableError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
