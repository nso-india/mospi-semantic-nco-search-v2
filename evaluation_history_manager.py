from __future__ import annotations

import csv
import io
import json
import logging
import os
import re
import threading
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from pymongo import ASCENDING, DESCENDING, MongoClient
    from pymongo.errors import PyMongoError
except ImportError:  # pragma: no cover - reported clearly at runtime
    ASCENDING = DESCENDING = None
    MongoClient = None
    PyMongoError = Exception


logger = logging.getLogger(__name__)


class EvaluationHistoryUnavailableError(RuntimeError):
    """Raised when MongoDB cannot save or retrieve evaluation history."""


class EvaluationHistoryManager:
    """Persist and retrieve the latest evaluation results for each product."""

    def __init__(
        self,
        mongo_uri: Optional[str] = None,
        database_name: Optional[str] = None,
        runs_collection: Optional[str] = None,
        rows_collection: Optional[str] = None,
    ) -> None:
        self.mongo_uri = mongo_uri or os.environ.get("MONGO_URI") or os.environ.get("MONGODB_URI")
        self.database_name = database_name or os.environ.get("MONGO_DB_NAME", "semantic_search")
        self.runs_collection_name = runs_collection or os.environ.get(
            "MONGO_EVALUATION_HISTORY_RUNS_COLLECTION", "evaluation_history_runs"
        )
        self.rows_collection_name = rows_collection or os.environ.get(
            "MONGO_EVALUATION_HISTORY_ROWS_COLLECTION", "evaluation_history_rows"
        )
        self._client = None
        self._runs = None
        self._rows = None
        self._lock = threading.RLock()

    @staticmethod
    def _text(value: Any) -> str:
        return "" if value is None else str(value).strip()

    @classmethod
    def _normalise_text(cls, value: Any) -> str:
        return " ".join(cls._text(value).casefold().split())

    @classmethod
    def _product_key(cls, product_name: Any) -> str:
        """Create a stable case-insensitive Mongo key from a product label."""
        normalised = cls._normalise_text(product_name)
        key = re.sub(r"[^a-z0-9]+", "_", normalised).strip("_")
        if not key:
            raise ValueError("Expected Dataset is required to save evaluation history")
        return key

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _iso(value: Any) -> Optional[str]:
        if isinstance(value, datetime):
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone.utc)
            return value.astimezone(timezone.utc).isoformat()
        return str(value) if value else None

    def ensure_collections(self) -> None:
        """Connect, then automatically create the two collections and indexes."""
        with self._lock:
            if self._runs is not None and self._rows is not None:
                return
            if not self.mongo_uri:
                raise EvaluationHistoryUnavailableError("MONGO_URI is not configured")
            if MongoClient is None:
                raise EvaluationHistoryUnavailableError("pymongo is not installed")
            try:
                safe_target = self.mongo_uri.rsplit("@", 1)[-1]
                logger.info("MongoDB evaluation history connecting to %s", safe_target)
                self._client = MongoClient(
                    self.mongo_uri,
                    serverSelectionTimeoutMS=3000,
                    connectTimeoutMS=3000,
                    tz_aware=True,
                )
                self._client.admin.command("ping")
                database = self._client[self.database_name]
                self._runs = database[self.runs_collection_name]
                self._rows = database[self.rows_collection_name]

                # ``create_index`` creates a collection automatically when it
                # does not exist yet.  No manual Compass/database step needed.
                self._runs.create_index([("completed_at", DESCENDING)])
                self._runs.create_index([("product_name", ASCENDING)])
                self._rows.create_index([
                    ("product_key", ASCENDING),
                    ("run_id", ASCENDING),
                    ("position", ASCENDING),
                ])
                self._rows.create_index([("run_id", ASCENDING)])
                logger.info(
                    "MongoDB evaluation history ready: %s.%s and %s.%s",
                    self.database_name,
                    self.runs_collection_name,
                    self.database_name,
                    self.rows_collection_name,
                )
            except PyMongoError as exc:
                self._client = self._runs = self._rows = None
                logger.error("MongoDB evaluation history connection failed: %s", exc)
                raise EvaluationHistoryUnavailableError(
                    "MongoDB evaluation history connection failed"
                ) from exc

    @classmethod
    def _summary(cls, results: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        rows = list(results)
        completed = [row for row in rows if row.get("status") == "completed"]
        dataset_rows = [row for row in completed if row.get("dataset_evaluated")]
        indicator_rows = [row for row in completed if row.get("indicator_evaluated")]
        filter_rows = [row for row in completed if row.get("filter_evaluated")]
        dataset_matches = sum(row.get("dataset_top_3_match") is True for row in dataset_rows)
        indicator_matches = sum(row.get("indicator_match") is True for row in indicator_rows)
        filter_matches = sum(row.get("filter_match") is True for row in filter_rows)
        return {
            "completed_rows": len(rows),
            "valid_rows": len(completed),
            "error_rows": len(rows) - len(completed),
            "dataset_top_3_evaluated_rows": len(dataset_rows),
            "indicator_evaluated_rows": len(indicator_rows),
            "filter_evaluated_rows": len(filter_rows),
            "dataset_top_3_missing_expected_rows": sum(not row.get("expected_dataset") for row in rows),
            "indicator_missing_expected_rows": sum(not row.get("expected_indicator") for row in rows),
            "filter_missing_expected_rows": sum(not row.get("expected_filters") for row in rows),
            "dataset_top_3_matches": dataset_matches,
            "indicator_matches": indicator_matches,
            "filter_matches": filter_matches,
            "dataset_top_3_accuracy": round(100 * dataset_matches / len(dataset_rows), 2) if dataset_rows else None,
            "indicator_accuracy": round(100 * indicator_matches / len(indicator_rows), 2) if indicator_rows else None,
            "filter_accuracy": round(100 * filter_matches / len(filter_rows), 2) if filter_rows else None,
        }

    @staticmethod
    def _safe_actor(actor: Any) -> Optional[Dict[str, Any]]:
        """Keep useful audit attribution without copying session/token fields."""
        if not isinstance(actor, dict):
            return None
        result = {
            "id": str(actor.get("id") or actor.get("_id") or ""),
            "username": str(actor.get("username") or ""),
            "roles": list(actor.get("roles") or ([actor["role"]] if actor.get("role") else [])),
        }
        return {key: value for key, value in result.items() if value not in ("", [])}

    @classmethod
    def _group_by_product(cls, results: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        grouped: Dict[str, Dict[str, Any]] = {}
        for result in results:
            product_name = cls._text(result.get("expected_dataset"))
            # Invalid source rows have no product to replace, so they are kept
            # in the per-job CSV only and not mixed into product history.
            if not product_name:
                continue
            product_key = cls._product_key(product_name)
            product = grouped.setdefault(product_key, {"product_name": product_name, "results": []})
            product["results"].append(deepcopy(result))
        return grouped

    def save_latest_by_product(self, job: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Replace saved rows only for products included in this completed job.

        New rows are inserted first, then the current-run pointer is updated,
        and finally old rows for that product are deleted.  A read always uses
        the pointer's ``run_id``, so it never returns a mixture of old and new
        evaluation rows.
        """
        if job.get("status") != "completed":
            raise ValueError("Only completed evaluations can be saved to history")
        self.ensure_collections()
        grouped = self._group_by_product(job.get("results", []))
        if not grouped:
            logger.info("Evaluation history skipped: no rows with Expected Dataset")
            return []

        saved: List[Dict[str, Any]] = []
        completed_at = self._now()
        actor = self._safe_actor(job.get("actor"))
        run_id = str(job.get("job_id") or uuid.uuid4())

        try:
            with self._lock:
                for product_key, product in grouped.items():
                    rows = product["results"]
                    row_documents = [
                        {
                            "_id": str(uuid.uuid4()),
                            "product_key": product_key,
                            "run_id": run_id,
                            "position": position,
                            "result": result,
                            "saved_at": completed_at,
                        }
                        for position, result in enumerate(rows, start=1)
                    ]
                    if row_documents:
                        self._rows.insert_many(row_documents, ordered=True)

                    summary = self._summary(rows)
                    run_document = {
                        "product_key": product_key,
                        "product_name": product["product_name"],
                        "run_id": run_id,
                        "file_name": self._text(job.get("file_name")),
                        "completed_at": completed_at,
                        "total_rows": len(rows),
                        "summary": summary,
                        "actor": actor,
                        "updated_at": completed_at,
                    }
                    self._runs.update_one(
                        {"_id": product_key},
                        {"$set": run_document, "$setOnInsert": {"created_at": completed_at}},
                        upsert=True,
                    )
                    # Delete only prior rows for this product. Other product
                    # histories from the same sheet (or other sheets) remain.
                    self._rows.delete_many({
                        "product_key": product_key,
                        "run_id": {"$ne": run_id},
                    })
                    saved.append(self._public_run({"_id": product_key, **run_document}))

        except PyMongoError as exc:
            logger.error("MongoDB evaluation history save failed: %s", exc)
            raise EvaluationHistoryUnavailableError("Could not save evaluation history to MongoDB") from exc

        logger.info(
            "MongoDB evaluation history saved: job=%s, products=%s",
            run_id,
            ", ".join(item["product_name"] for item in saved),
        )
        return saved

    @classmethod
    def _public_run(cls, document: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "product_key": cls._text(document.get("product_key") or document.get("_id")),
            "product_name": cls._text(document.get("product_name")),
            "run_id": cls._text(document.get("run_id")),
            "file_name": cls._text(document.get("file_name")),
            "completed_at": cls._iso(document.get("completed_at")),
            "total_rows": int(document.get("total_rows") or 0),
            "summary": deepcopy(document.get("summary") or {}),
            "actor": deepcopy(document.get("actor")) if document.get("actor") else None,
        }

    def list_latest(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Return the current saved run for each product, newest first."""
        if not 1 <= limit <= 500:
            raise ValueError("limit must be between 1 and 500")
        self.ensure_collections()
        try:
            documents = self._runs.find({}).sort("completed_at", DESCENDING).limit(limit)
            return [self._public_run(document) for document in documents]
        except PyMongoError as exc:
            logger.error("MongoDB evaluation history list failed: %s", exc)
            raise EvaluationHistoryUnavailableError("Could not load evaluation history from MongoDB") from exc

    def list_latest_page(
        self,
        *,
        page: int = 1,
        page_size: int = 10,
        search: str = "",
    ) -> Dict[str, Any]:
        """Return a searchable, paginated list of current product evaluations."""
        if page < 1:
            raise ValueError("page must be at least 1")
        if not 1 <= page_size <= 100:
            raise ValueError("page_size must be between 1 and 100")

        self.ensure_collections()
        search_text = self._text(search)
        query: Dict[str, Any] = {}
        if search_text:
            query = {
                "$or": [
                    {"product_name": {"$regex": re.escape(search_text), "$options": "i"}},
                    {"product_key": {"$regex": re.escape(search_text), "$options": "i"}},
                ]
            }

        try:
            total = self._runs.count_documents(query)
            total_pages = max(1, (total + page_size - 1) // page_size)
            if page > total_pages:
                page = total_pages
            documents = (
                self._runs.find(query)
                .sort("completed_at", DESCENDING)
                .skip((page - 1) * page_size)
                .limit(page_size)
            )
            return {
                "history": [self._public_run(document) for document in documents],
                "page": page,
                "page_size": page_size,
                "total": total,
                "total_pages": total_pages,
                "search": search_text,
            }
        except PyMongoError as exc:
            logger.error("MongoDB evaluation history page failed: %s", exc)
            raise EvaluationHistoryUnavailableError("Could not load evaluation history from MongoDB") from exc

    def get_latest_rows(self, product_key: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Read only the rows belonging to a product's current saved run."""
        self.ensure_collections()
        key = self._product_key(product_key)
        try:
            run = self._runs.find_one({"_id": key})
            if run is None:
                raise KeyError(key)
            documents = self._rows.find({
                "product_key": key,
                "run_id": run["run_id"],
            }).sort("position", ASCENDING)
            return self._public_run(run), [deepcopy(document.get("result") or {}) for document in documents]
        except PyMongoError as exc:
            logger.error("MongoDB evaluation history read failed: %s", exc)
            raise EvaluationHistoryUnavailableError("Could not load evaluation history from MongoDB") from exc

    @staticmethod
    def _csv_value(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        return str(value)

    @staticmethod
    def _csv_match(value: Any) -> str:
        if value is None:
            return ""
        return "TRUE" if bool(value) else "FALSE"

    def latest_csv(self, product_key: str) -> Tuple[bytes, str]:
        """Build a spreadsheet-friendly CSV from the product's latest rows."""
        run, results = self.get_latest_rows(product_key)
        fieldnames = (
            "Row Number", "Prompt", "Predicted Datasets", "Expected Dataset",
            "Dataset Match", "Predicted Indicator", "Expected Indicator",
            "Indicator Match", "Predicted Filters", "Expected Filter",
            "Filters Match", "Status", "Reason",
        )
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({
                "Row Number": result.get("row_number", ""),
                "Prompt": self._csv_value(result.get("prompt")),
                "Predicted Datasets": self._csv_value(result.get("predicted_datasets")),
                "Expected Dataset": self._csv_value(result.get("expected_dataset")),
                "Dataset Match": self._csv_match(result.get("dataset_top_3_match")),
                "Predicted Indicator": self._csv_value(result.get("predicted_indicator")),
                "Expected Indicator": self._csv_value(result.get("expected_indicator")),
                "Indicator Match": self._csv_match(result.get("indicator_match")),
                "Predicted Filters": self._csv_value(result.get("predicted_filters")),
                "Expected Filter": self._csv_value(result.get("expected_filters")),
                "Filters Match": self._csv_match(result.get("filter_match")),
                "Status": self._csv_value(result.get("status")),
                "Reason": self._csv_value(result.get("reason")),
            })
        filename = f"evaluation_{run['product_key']}_latest.csv"
        return ("\ufeff" + output.getvalue()).encode("utf-8"), filename
