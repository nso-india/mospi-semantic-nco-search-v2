from __future__ import annotations

import csv
import io
import json
import logging
import os
import threading
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Tuple

if TYPE_CHECKING:
    from evaluation_history_manager import EvaluationHistoryManager

try:
    from openpyxl import load_workbook
except ImportError:  # pragma: no cover - reported clearly if xlsx is uploaded
    load_workbook = None


logger = logging.getLogger(__name__)


class EvaluationValidationError(ValueError):
    """The uploaded evaluation file does not meet the required template."""


class EvaluationRunner:
    """Run one uploaded test sheet at a time in a background thread.

    Results stay in memory while a job is running and are also written to an
    individual CSV file.  The CSV is the download artifact and avoids storing
    test prompts/results in user-interaction analytics.
    """

    COLUMN_ALIASES = {
        "prompt": ("Prompts", "Prompt", "Query", "Queries"),
        "expected_dataset": (
            "Expected Dataset", "Expected Datasets", "Dataset", "Datasets",
            "Expected Product", "Expected Products",
        ),
        "expected_indicator": (
            "Expected Indicator", "Expected Indicators", "Indicator", "Indicators",
        ),
        "expected_filter": (
            "Expected Filter", "Expected Filters", "Filter", "Filters", "Expected Truth",
        ),
    }
    REQUIRED_FIELD = "prompt"
    ALLOWED_EXTENSIONS = {".csv", ".xlsx"}
    MAX_ROWS = 1_000

    def __init__(
        self,
        output_dir: str | Path,
        history_manager: Optional["EvaluationHistoryManager"] = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Mongo history is optional here so an evaluation result can still be
        # downloaded if history storage has a temporary outage.
        self.history_manager = history_manager
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _as_text(value: Any) -> str:
        return "" if value is None else str(value).strip()

    @staticmethod
    def _normalise_text(value: Any) -> str:
        """Compare labels without case or spacing-only differences."""
        return " ".join(str(value or "").casefold().split())

    @classmethod
    def _normalise_header(cls, value: Any) -> str:
        return cls._normalise_text(value)

    @classmethod
    def _validate_headers(cls, headers: Iterable[Any]) -> Dict[str, str]:
        aliases = {
            cls._normalise_header(alias): field
            for field, names in cls.COLUMN_ALIASES.items()
            for alias in names
        }
        available: Dict[str, str] = {}
        for header in headers:
            header_text = cls._as_text(header)
            if header_text:
                canonical_field = aliases.get(cls._normalise_header(header_text))
                if not canonical_field:
                    continue
                if canonical_field in available:
                    raise EvaluationValidationError(
                        f"Duplicate column for {canonical_field.replace('_', ' ')}: "
                        f"{available[canonical_field]} and {header_text}"
                    )
                available[canonical_field] = header_text

        if cls.REQUIRED_FIELD not in available:
            raise EvaluationValidationError(
                "Missing required prompt column. Accepted names: Prompts, Prompt, Query, Queries"
            )
        return available

    @classmethod
    def _row_from_values(
        cls,
        values: Iterable[Any],
        headers: Iterable[Any],
    ) -> Dict[str, str]:
        return {
            cls._as_text(header): cls._as_text(value)
            for header, value in zip(headers, values)
            if cls._as_text(header)
        }

    @classmethod
    def _read_csv(cls, content: bytes) -> List[Dict[str, str]]:
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise EvaluationValidationError("CSV must be UTF-8 encoded") from exc

        reader = csv.DictReader(io.StringIO(text))
        if not reader.fieldnames:
            raise EvaluationValidationError("CSV is empty or has no header row")
        cls._validate_headers(reader.fieldnames)
        return [
            {cls._as_text(key): cls._as_text(value) for key, value in row.items()}
            for row in reader
        ]

    @classmethod
    def _read_xlsx(cls, content: bytes) -> List[Dict[str, str]]:
        if load_workbook is None:
            raise EvaluationValidationError(
                "Excel support is unavailable. Install openpyxl and restart the service."
            )
        try:
            workbook = load_workbook(io.BytesIO(content), read_only=True, data_only=True)
            worksheet = workbook.active
            rows = worksheet.iter_rows(values_only=True)
            headers = next(rows, None)
            if not headers:
                raise EvaluationValidationError("Excel file is empty or has no header row")
            cls._validate_headers(headers)
            return [cls._row_from_values(values, headers) for values in rows]
        except EvaluationValidationError:
            raise
        except Exception as exc:
            raise EvaluationValidationError("Could not read the Excel file") from exc

    @classmethod
    def parse_upload(cls, filename: str, content: bytes) -> List[Dict[str, str]]:
        extension = Path(filename or "").suffix.lower()
        if extension not in cls.ALLOWED_EXTENSIONS:
            raise EvaluationValidationError("Upload a .csv or .xlsx evaluation file")
        if not content:
            raise EvaluationValidationError("The uploaded file is empty")

        rows = cls._read_csv(content) if extension == ".csv" else cls._read_xlsx(content)
        if not rows:
            raise EvaluationValidationError("The uploaded file has no evaluation rows")
        if len(rows) > cls.MAX_ROWS:
            raise EvaluationValidationError(
                f"Evaluation files are limited to {cls.MAX_ROWS} rows"
            )
        invalid_filter_rows = []
        for row_number, row in enumerate(rows, start=2):
            raw_filter = cls._expected_filters_value(row)
            if not raw_filter:
                continue
            try:
                cls._parse_expected_filters(raw_filter)
            except EvaluationValidationError:
                invalid_filter_rows.append(str(row_number))
        if invalid_filter_rows:
            raise EvaluationValidationError(
                "Expected Filter must be a valid JSON object in row(s): "
                + ", ".join(invalid_filter_rows)
            )
        return rows

    def start(
        self,
        *,
        filename: str,
        content: bytes,
        prediction_fn: Callable[[str], Dict[str, Any]],
        actor: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Validate an upload, create a job, and begin sequential evaluation."""
        rows = self.parse_upload(filename, content)
        job_id = str(uuid.uuid4())
        job = {
            "job_id": job_id,
            "file_name": Path(filename).name,
            "status": "queued",
            "created_at": self._now(),
            "started_at": None,
            "completed_at": None,
            "total_rows": len(rows),
            "completed_rows": 0,
            "results": [],
            "error": None,
            # Cancellation is cooperative: the active LLM request is allowed
            # to finish, then no further prompt is started.
            "cancel_requested": False,
            "cancelled_at": None,
            "actor": deepcopy(actor) if isinstance(actor, dict) else None,
            "history_status": "pending" if self.history_manager is not None else "not_configured",
            "history_products": [],
            "output_path": str(self.output_dir / f"evaluation_{job_id}.csv"),
        }
        with self._lock:
            self._jobs[job_id] = job

        thread = threading.Thread(
            target=self._run_job,
            args=(job_id, rows, prediction_fn),
            name=f"evaluation-{job_id[:8]}",
            daemon=True,
        )
        thread.start()
        return self.get_page(job_id, page=1, page_size=20)

    @classmethod
    def _expected_value(cls, row: Dict[str, str], column: str) -> str:
        aliases = cls.COLUMN_ALIASES.get(column, (column,))
        targets = {cls._normalise_header(alias) for alias in aliases}
        for key, value in row.items():
            if cls._normalise_header(key) in targets:
                return cls._as_text(value)
        return ""

    @classmethod
    def _expected_filters_value(cls, row: Dict[str, str]) -> str:
        """Read an optional expected-filter value using its accepted aliases."""
        return cls._expected_value(row, "expected_filter")

    @classmethod
    def _parse_expected_filters(cls, raw_value: str) -> Dict[str, Any]:
        if not raw_value:
            return {}
        try:
            value = json.loads(raw_value)
        except json.JSONDecodeError as exc:
            raise EvaluationValidationError("Expected Filter must be a valid JSON object") from exc
        if not isinstance(value, dict):
            raise EvaluationValidationError("Expected Filter must be a JSON object")
        return value

    @classmethod
    def _actual_filter_map(cls, result: Dict[str, Any]) -> Dict[str, str]:
        filters: Dict[str, str] = {}
        for item in result.get("filters", []) or []:
            if not isinstance(item, dict):
                continue
            name = cls._as_text(item.get("filter_name"))
            if name:
                filters[name] = cls._as_text(item.get("option"))
        return filters

    @classmethod
    def _normalise_filter_name(cls, value: Any) -> List[str]:
        """Return filter-label words, treating ``/`` and ``_`` as separators.

        Metadata does not always use one spelling for the same filter. For
        example, a sheet can say ``State`` while the product metadata calls it
        ``State/UT``. This normalisation is used for *filter names only*;
        selected filter values are still compared exactly below.
        """
        label = cls._as_text(value).casefold()
        separator_normalised = "".join(
            character if character.isalnum() else " "
            for character in label
        )
        return separator_normalised.split()

    @classmethod
    def _filter_names_match(cls, actual_name: Any, expected_name: Any) -> bool:
        """Match an exact label first, then a safe word-boundary shorthand.

        The shorthand rule only accepts a complete leading sequence of words,
        never an arbitrary substring. Therefore ``State`` matches ``State/UT``
        and ``Age`` matches ``Age Group``, while unrelated names such as
        ``Rate`` and ``Birth Rate`` do not match.
        """
        actual_words = cls._normalise_filter_name(actual_name)
        expected_words = cls._normalise_filter_name(expected_name)
        if not actual_words or not expected_words:
            return False
        if actual_words == expected_words:
            return True

        shorter, longer = (
            (actual_words, expected_words)
            if len(actual_words) < len(expected_words)
            else (expected_words, actual_words)
        )
        return len(shorter) < len(longer) and longer[: len(shorter)] == shorter

    @classmethod
    def _evaluate_row(
        cls,
        row_number: int,
        row: Dict[str, str],
        prediction_fn: Callable[[str], Dict[str, Any]],
    ) -> Dict[str, Any]:
        prompt = cls._expected_value(row, "prompt")
        expected_dataset = cls._expected_value(row, "expected_dataset")
        expected_indicator = cls._expected_value(row, "expected_indicator")
        expected_filters_text = cls._expected_filters_value(row)

        base = {
            "row_number": row_number,
            "prompt": prompt,
            "expected_dataset": expected_dataset,
            "expected_indicator": expected_indicator,
            "expected_filters": {},
            "predicted_datasets": [],
            "dataset_evaluated": False,
            "dataset_top_3_match": None,
            "predicted_indicator": None,
            "indicator_evaluated": False,
            "indicator_match": None,
            "predicted_filters": {},
            "filter_evaluated": False,
            "filter_match": None,
            "reason": None,
            "status": "completed",
        }

        if not prompt:
            base.update({"status": "error", "reason": "Prompts is required"})
            return base
        try:
            expected_filters = cls._parse_expected_filters(expected_filters_text)
            base["expected_filters"] = expected_filters
            response = prediction_fn(prompt)
            predicted_results = response.get("results", []) if isinstance(response, dict) else []
            if not isinstance(predicted_results, list):
                raise RuntimeError("Prediction returned an invalid results payload")

            base["predicted_datasets"] = [
                cls._as_text(result.get("dataset"))
                for result in predicted_results
                if isinstance(result, dict) and cls._as_text(result.get("dataset"))
            ]
            if not expected_dataset:
                base["reason"] = "No Expected Dataset supplied; dataset, indicator, and filters were not evaluated"
                return base

            base["dataset_evaluated"] = True
            expected_dataset_key = cls._normalise_text(expected_dataset)
            expected_result = next(
                (
                    result for result in predicted_results
                    if isinstance(result, dict)
                    and cls._normalise_text(result.get("dataset")) == expected_dataset_key
                ),
                None,
            )
            base["dataset_top_3_match"] = expected_result is not None
            if expected_result is None:
                base["reason"] = "Expected dataset was not returned in the top 3 results"
                return base

            predicted_indicator = cls._as_text(expected_result.get("indicator"))
            predicted_filters = cls._actual_filter_map(expected_result)
            base["predicted_indicator"] = predicted_indicator
            base["predicted_filters"] = predicted_filters
            if expected_indicator:
                base["indicator_evaluated"] = True
                base["indicator_match"] = (
                    cls._normalise_text(predicted_indicator)
                    == cls._normalise_text(expected_indicator)
                )

            # Expected filters are intentionally compared as a subset. A normal
            # prediction may add useful defaults such as State=Select All;
            # those extra filters do not make the supplied expectation wrong.
            mismatches = []
            if expected_filters:
                base["filter_evaluated"] = True
                for expected_name, expected_value in expected_filters.items():
                    actual_value = next(
                        (
                            value for name, value in predicted_filters.items()
                            # Filter labels may be a clear shorthand in the
                            # uploaded sheet, e.g. State instead of State/UT.
                            if cls._filter_names_match(name, expected_name)
                        ),
                        None,
                    )
                    if cls._normalise_text(actual_value) != cls._normalise_text(expected_value):
                        mismatches.append(str(expected_name))
                base["filter_match"] = not mismatches

            failures = []
            if base["indicator_evaluated"] and base["indicator_match"] is False:
                failures.append("Indicator did not match")
            if base["filter_evaluated"] and mismatches:
                failures.append("Filter mismatch: " + ", ".join(mismatches))
            if failures:
                base["reason"] = "; ".join(failures)
            else:
                evaluated = ["dataset"]
                if base["indicator_evaluated"]:
                    evaluated.append("indicator")
                if base["filter_evaluated"]:
                    evaluated.append("filters")
                base["reason"] = "Matched expected " + ", ".join(evaluated)
            return base
        except Exception as exc:
            logger.exception("Evaluation row %s failed", row_number)
            base.update({"status": "error", "reason": str(exc)})
            return base

    @classmethod
    def _summary(cls, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        completed = [item for item in results if item.get("status") == "completed"]
        dataset_rows = [item for item in completed if item.get("dataset_evaluated")]
        indicator_rows = [item for item in completed if item.get("indicator_evaluated")]
        filter_rows = [item for item in completed if item.get("filter_evaluated")]
        dataset_matches = sum(item.get("dataset_top_3_match") is True for item in dataset_rows)
        indicator_matches = sum(item.get("indicator_match") is True for item in indicator_rows)
        filter_matches = sum(item.get("filter_match") is True for item in filter_rows)
        return {
            "completed_rows": len(results),
            "valid_rows": len(completed),
            "error_rows": len(results) - len(completed),
            "dataset_top_3_evaluated_rows": len(dataset_rows),
            "indicator_evaluated_rows": len(indicator_rows),
            "filter_evaluated_rows": len(filter_rows),
            "dataset_top_3_missing_expected_rows": sum(not item.get("expected_dataset") for item in results),
            "indicator_missing_expected_rows": sum(not item.get("expected_indicator") for item in results),
            "filter_missing_expected_rows": sum(not item.get("expected_filters") for item in results),
            "dataset_top_3_matches": dataset_matches,
            "indicator_matches": indicator_matches,
            "filter_matches": filter_matches,
            "dataset_top_3_accuracy": round(100 * dataset_matches / len(dataset_rows), 2) if dataset_rows else None,
            "indicator_accuracy": round(100 * indicator_matches / len(indicator_rows), 2) if indicator_rows else None,
            "filter_accuracy": round(100 * filter_matches / len(filter_rows), 2) if filter_rows else None,
        }

    @staticmethod
    def _csv_value(value: Any) -> str:
        """Return a spreadsheet-safe text value without losing filter maps."""
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

    def _write_result_file(self, job: Dict[str, Any]) -> None:
        """Write one completed evaluation as a spreadsheet-friendly CSV file."""
        output_path = Path(job["output_path"])
        temporary_path = output_path.with_suffix(".tmp")
        fieldnames = (
            "Row Number",
            "Prompt",
            "Predicted Datasets",
            "Expected Dataset",
            "Dataset Match",
            "Predicted Indicator",
            "Expected Indicator",
            "Indicator Match",
            "Predicted Filters",
            "Expected Filter",
            "Filters Match",
            "Status",
            "Reason",
        )
        # utf-8-sig lets Excel open Indian-language text and symbols correctly.
        with temporary_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for result in job["results"]:
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
        os.replace(temporary_path, output_path)

    def _mark_cancelled_locked(self, job: Dict[str, Any]) -> None:
        """Mark a job stopped while its lock is already held."""
        job["status"] = "cancelled"
        job["cancelled_at"] = self._now()
        job["completed_at"] = job["cancelled_at"]
        # A stopped partial run must not replace a product's completed history.
        job["history_status"] = "cancelled"

    def request_cancel(self, job_id: str) -> Dict[str, Any]:
        """Request a safe stop after the current prompt, if one is running."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if job["status"] not in {"queued", "running"}:
                raise EvaluationValidationError("Only a queued or running evaluation can be stopped")
            job["cancel_requested"] = True
        return self.get_page(job_id, page=1, page_size=10)

    def _run_job(
        self,
        job_id: str,
        rows: List[Dict[str, str]],
        prediction_fn: Callable[[str], Dict[str, Any]],
    ) -> None:
        with self._lock:
            job = self._jobs[job_id]
            if job.get("cancel_requested"):
                self._mark_cancelled_locked(job)
                return
            job["status"] = "running"
            job["started_at"] = self._now()

        for row_number, row in enumerate(rows, start=2):
            # Do not start another LLM/semantic-search request once Stop was
            # pressed. A request already in progress cannot be killed safely.
            with self._lock:
                job = self._jobs[job_id]
                if job.get("cancel_requested"):
                    self._mark_cancelled_locked(job)
                    return
            result = self._evaluate_row(row_number, row, prediction_fn)
            with self._lock:
                job = self._jobs[job_id]
                if job.get("cancel_requested"):
                    self._mark_cancelled_locked(job)
                    return
                job["results"].append(result)
                job["completed_rows"] = len(job["results"])

        with self._lock:
            job = self._jobs[job_id]
            if job.get("cancel_requested"):
                self._mark_cancelled_locked(job)
                return
            job["status"] = "completed"
            job["completed_at"] = self._now()
            try:
                self._write_result_file(job)
            except Exception as exc:
                logger.exception("Could not write evaluation result file for %s", job_id)
                job["status"] = "failed"
                job["error"] = f"Could not save the CSV download: {exc}"

            # Take an immutable snapshot while holding the lock. MongoDB work
            # happens after the lock so a slow database cannot block the UI
            # from reading progress for another evaluation job.
            history_snapshot = deepcopy(job) if job["status"] == "completed" else None

        if history_snapshot is not None and self.history_manager is not None:
            try:
                products = self.history_manager.save_latest_by_product(history_snapshot)
                with self._lock:
                    job = self._jobs[job_id]
                    job["history_status"] = "saved"
                    job["history_products"] = products
            except Exception as exc:
                # The completed per-job CSV remains usable even if MongoDB is
                # unavailable. The UI makes this status visible for tracing.
                logger.exception("Could not save MongoDB evaluation history for %s", job_id)
                with self._lock:
                    job = self._jobs[job_id]
                    job["history_status"] = "failed"
                    job["history_error"] = str(exc)

    def get_page(self, job_id: str, *, page: int, page_size: int) -> Dict[str, Any]:
        """Return a safe paginated copy of one evaluation job."""
        if page < 1:
            raise EvaluationValidationError("page must be at least 1")
        if not 1 <= page_size <= 100:
            raise EvaluationValidationError("page_size must be between 1 and 100")

        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            results = deepcopy(job["results"])
            total_results = len(results)
            start = (page - 1) * page_size
            return {
                "job_id": job["job_id"],
                "file_name": job["file_name"],
                "status": job["status"],
                "created_at": job["created_at"],
                "started_at": job["started_at"],
                "completed_at": job["completed_at"],
                "cancelled_at": job.get("cancelled_at"),
                "cancel_requested": bool(job.get("cancel_requested")),
                "error": job["error"],
                "history_status": job.get("history_status"),
                "history_error": job.get("history_error"),
                "history_products": deepcopy(job.get("history_products", [])),
                "total_rows": job["total_rows"],
                "completed_rows": job["completed_rows"],
                "summary": self._summary(results),
                "page": page,
                "page_size": page_size,
                "total_results": total_results,
                "total_pages": max(1, (total_results + page_size - 1) // page_size),
                "results": results[start:start + page_size],
                "download_ready": job["status"] == "completed" and Path(job["output_path"]).is_file(),
            }

    def get_download_path(self, job_id: str) -> Path:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            path = Path(job["output_path"])
            if job["status"] != "completed" or not path.is_file():
                raise EvaluationValidationError("Evaluation CSV is not ready yet")
            return path
