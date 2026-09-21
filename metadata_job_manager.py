from __future__ import annotations

import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict

from pymongo import DESCENDING, MongoClient
from pymongo.errors import PyMongoError


logger = logging.getLogger(__name__)


class MetadataJobUnavailableError(RuntimeError):
    """Raised when the metadata-job MongoDB collection cannot be reached."""


class MetadataJobManager:
    """Store small job status documents and execute one task in a daemon thread."""

    def __init__(self) -> None:
        self.mongo_uri = os.environ.get("MONGO_URI")
        self.database_name = os.environ.get("MONGO_DB_NAME", "semantic_search")
        self.collection_name = os.environ.get("MONGO_METADATA_JOBS_COLLECTION", "metadata_jobs")
        self._client = None
        self._collection = None

    def _collection_handle(self):
        """Connect lazily and create indexes (which also creates the collection)."""
        if self._collection is not None:
            return self._collection
        if not self.mongo_uri:
            raise MetadataJobUnavailableError("MONGO_URI is required for metadata job progress")
        try:
            self._client = MongoClient(
                self.mongo_uri,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
                tz_aware=True,
            )
            self._client.admin.command("ping")
            collection = self._client[self.database_name][self.collection_name]
            collection.create_index([("owner.user_id", 1), ("updated_at", DESCENDING)])
            collection.create_index([("product_name", 1), ("updated_at", DESCENDING)])
            collection.create_index([("status", 1), ("updated_at", DESCENDING)])
            self._collection = collection
            logger.info(
                "MongoDB metadata jobs connected: database=%s, collection=%s",
                self.database_name,
                self.collection_name,
            )
            return collection
        except PyMongoError as exc:
            logger.error("MongoDB metadata jobs connection failed: %s", exc)
            raise MetadataJobUnavailableError("Metadata progress service is unavailable") from exc

    @staticmethod
    def _owner(actor: Any) -> Dict[str, str]:
        if not isinstance(actor, dict):
            return {"username": "system"}
        owner = {"username": str(actor.get("username") or "unknown").strip() or "unknown"}
        user_id = str(actor.get("id") or actor.get("_id") or "").strip()
        if user_id:
            owner["user_id"] = user_id
        return owner

    @staticmethod
    def _is_admin(actor: Any) -> bool:
        if not isinstance(actor, dict):
            return False
        roles = actor.get("roles", actor.get("role", []))
        if isinstance(roles, str):
            roles = [roles]
        return "admin" in roles if isinstance(roles, (list, tuple, set)) else False

    def _assert_access(self, job: Dict[str, Any], actor: Any) -> None:
        """Only the job owner or an administrator may inspect its status."""
        if self._is_admin(actor):
            return
        owner_id = str((job.get("owner") or {}).get("user_id") or "")
        actor_id = str((actor or {}).get("id") or (actor or {}).get("_id") or "")
        if not owner_id or owner_id != actor_id:
            raise PermissionError("You do not have access to this metadata job")

    @staticmethod
    def _public_job(job: Dict[str, Any]) -> Dict[str, Any]:
        """Return safe status fields only; never expose internal request data."""
        fields = (
            "job_id", "operation", "product_name", "status", "stage", "progress",
            "message", "created_at", "updated_at", "completed_at", "result", "error", "owner",
        )
        return {field: job.get(field) for field in fields if field in job}

    def create_job(self, operation: str, product_name: str, actor: Any) -> Dict[str, Any]:
        now = datetime.now(timezone.utc)
        job = {
            "_id": str(uuid.uuid4()),
            "job_id": str(uuid.uuid4()),
            "operation": operation,
            "product_name": product_name,
            "owner": self._owner(actor),
            "status": "queued",
            "stage": "queued",
            "progress": 0,
            "message": "Waiting to start",
            "created_at": now,
            "updated_at": now,
        }
        try:
            self._collection_handle().insert_one(job)
        except PyMongoError as exc:
            raise MetadataJobUnavailableError("Could not create metadata progress job") from exc
        return self._public_job(job)

    def update_job(
        self,
        job_id: str,
        *,
        status: str | None = None,
        stage: str | None = None,
        progress: int | None = None,
        message: str | None = None,
        result: Dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        changes: Dict[str, Any] = {"updated_at": datetime.now(timezone.utc)}
        if status is not None:
            changes["status"] = status
        if stage is not None:
            changes["stage"] = stage
        if progress is not None:
            changes["progress"] = max(0, min(int(progress), 100))
        if message is not None:
            changes["message"] = message
        if result is not None:
            changes["result"] = result
        if error is not None:
            changes["error"] = error
        if status in {"completed", "failed"}:
            changes["completed_at"] = datetime.now(timezone.utc)
        try:
            self._collection_handle().update_one({"job_id": job_id}, {"$set": changes})
        except PyMongoError as exc:
            # The actual metadata operation should not be rolled back just
            # because its cosmetic progress update could not be written.
            logger.error("Could not update metadata job %s: %s", job_id, exc)

    def start_job(
        self,
        operation: str,
        product_name: str,
        actor: Any,
        worker: Callable[[Callable[[str, int, str], None]], Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Create a job then execute ``worker`` in the background.

        ``worker`` receives a callback for reporting its deterministic stages:
        saving Mongo data, refreshing embeddings, and completion.
        """
        job = self.create_job(operation, product_name, actor)
        job_id = job["job_id"]

        def report(stage: str, progress: int, message: str) -> None:
            self.update_job(job_id, status="running", stage=stage, progress=progress, message=message)

        def run() -> None:
            try:
                report("starting", 5, "Preparing metadata update")
                result = worker(report)
                self.update_job(
                    job_id,
                    status="completed",
                    stage="completed",
                    progress=100,
                    message="Metadata update and search indexing completed",
                    result=result,
                )
                logger.info("Metadata job completed: id=%s operation=%s product=%s", job_id, operation, product_name)
            except Exception as exc:  # Worker errors are shown as safe job status.
                logger.error("Metadata job failed: id=%s operation=%s product=%s error=%s", job_id, operation, product_name, exc, exc_info=True)
                self.update_job(
                    job_id,
                    status="failed",
                    stage="failed",
                    progress=100,
                    message="Metadata was not fully indexed",
                    error=str(exc),
                )

        threading.Thread(target=run, name=f"metadata-{operation}-{product_name}", daemon=True).start()
        return job

    def get_job(self, job_id: str, actor: Any) -> Dict[str, Any]:
        try:
            job = self._collection_handle().find_one({"job_id": job_id})
        except PyMongoError as exc:
            raise MetadataJobUnavailableError("Could not load metadata progress") from exc
        if job is None:
            raise KeyError("Metadata job not found")
        self._assert_access(job, actor)
        return self._public_job(job)
