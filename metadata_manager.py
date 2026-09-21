from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
import re
import uuid
from copy import deepcopy
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.errors import DuplicateKeyError, PyMongoError


logger = logging.getLogger(__name__)


class MetadataConflictError(ValueError):
    """Raised when a user attempts to save a stale copy of a product."""


class MetadataManager:
    """Mongo-backed published metadata and user-owned JSON drafts.

    Published product JSON is stored only in MongoDB.  The JSON itself stays
    in ``source_json`` unchanged in shape; document metadata such as versions
    lives outside that source JSON and is never sent to the embedding model.
    """

    def __init__(self, catalog_file_path: str, backup_dir: str | None = None):
        self.catalog_file = Path(catalog_file_path).resolve()
        self.catalog_dir = self.catalog_file.parent
        self.product_dir = self.catalog_dir / "products"
    
        self.drafts_dir = self.catalog_dir / "drafts"
        self.backup_dir = Path(backup_dir).resolve() if backup_dir else self.catalog_dir / "backups"
        self.audit_log_file = self.catalog_dir / "logs" / "metadata_changes.jsonl"
        self.audit_log_file.parent.mkdir(parents=True, exist_ok=True)
        self.drafts_dir.mkdir(parents=True, exist_ok=True)
        self.mongo_uri = os.environ.get("MONGO_URI")
        self.mongo_database_name = os.environ.get("MONGO_DB_NAME", "semantic_search")
        self.mongo_collection_name = os.environ.get("MONGO_PRODUCT_COLLECTION", "product_metadata")
        self.mongo_draft_collection_name = os.environ.get("MONGO_DRAFT_COLLECTION", "metadata_drafts")
        self._mongo_client = None
        self._product_collection = None
        self._draft_collection = None
        self._bootstrapped_local_products = False
        self._legacy_drafts_migrated = False
        self._embedding_model = None
        self._qdrant_client = None
        self._qdrant_models = None
        self._qdrant_collection_name = os.environ.get("QDRANT_COLLECTION", "product_metadata")
        self._qdrant_collection_ready = False

    @staticmethod
    def _actor_snapshot(actor: Any) -> Dict[str, str]:
        """Keep only safe user identity fields beside Mongo source_json.

        This deliberately excludes passwords, sessions, e-mail, and tokens.
        The returned object is stored in Mongo management fields only; it is
        never added to source_json or sent to Qdrant/the embedding model.
        """
        if not isinstance(actor, dict):
            return {"username": "system"}
        username = str(actor.get("username") or "").strip() or "unknown"
        user_id = str(actor.get("id") or actor.get("_id") or "").strip()
        snapshot = {"username": username}
        if user_id:
            snapshot["user_id"] = user_id
        return snapshot

    # ------------------------------------------------------------------
    # Mongo published-product access
    # ------------------------------------------------------------------
    @staticmethod
    def _read_json(path: Path) -> Dict[str, Any]:
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except FileNotFoundError:
            raise FileNotFoundError(f"Metadata file not found: {path}") from None
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path.name}: {exc}") from exc

        if not isinstance(data, dict):
            raise ValueError(f"Metadata file must contain a JSON object: {path.name}")
        return data

    def _load_catalog(self) -> Dict[str, str]:
        catalog = self._read_json(self.catalog_file)
        # Support a future {"products": {...}} wrapper without changing the API.
        catalog = catalog.get("products", catalog)
        if not isinstance(catalog, dict) or not catalog:
            raise ValueError("Product catalog must be a non-empty object")

        validated: Dict[str, str] = {}
        for product_code, relative_path in catalog.items():
            if not isinstance(product_code, str) or not product_code.strip():
                raise ValueError("Product catalog contains an invalid product code")
            if not isinstance(relative_path, str) or not relative_path.strip():
                raise ValueError(f"Product catalog entry for {product_code!r} has an invalid path")
            validated[product_code] = relative_path
        return validated

    def _get_product_collection(self):
        """Return the one Mongo collection that holds all published products."""
        if self._product_collection is not None:
            return self._product_collection
        if not self.mongo_uri:
            raise RuntimeError("MONGO_URI is required for published product metadata")
        try:
            self._mongo_client = MongoClient(
                self.mongo_uri,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
                tz_aware=True,
            )
            self._mongo_client.admin.command("ping")
            collection = self._mongo_client[self.mongo_database_name][self.mongo_collection_name]
            # product_key prevents duplicate products such as NAS/nas.
            # sequence preserves the order shown in the product selector.
            collection.create_index("product_key", unique=True)
            collection.create_index("sequence")
            self._product_collection = collection
            logger.info(
                "MongoDB product metadata connected: database=%s, collection=%s",
                self.mongo_database_name,
                self.mongo_collection_name,
            )
            self._bootstrap_local_products_once()
            return collection
        except PyMongoError as exc:
            logger.error("MongoDB product metadata connection failed: %s", exc)
            raise RuntimeError("MongoDB product metadata connection failed") from exc

    def _get_draft_collection(self):
        """Return the separate Mongo collection for unfinished user drafts.

        Creating its indexes also creates the collection automatically in
        MongoDB.  A draft can therefore never appear in ``product_metadata``
        or in the Qdrant search index before its explicit publish action.
        """
        if self._draft_collection is not None:
            return self._draft_collection
        # Reuse the already configured Mongo client and its connection checks.
        self._get_product_collection()
        if self._mongo_client is None:
            raise RuntimeError("MongoDB product metadata connection failed")
        try:
            collection = self._mongo_client[self.mongo_database_name][self.mongo_draft_collection_name]
            # One owner can have one editable draft for a given product name;
            # different users may draft the same future product independently.
            collection.create_index([("owner_key", ASCENDING), ("product_key", ASCENDING)], unique=True)
            collection.create_index([("owner.user_id", ASCENDING), ("updated_at", DESCENDING)])
            collection.create_index([("updated_at", DESCENDING)])
            self._draft_collection = collection
            logger.info(
                "MongoDB metadata drafts connected: database=%s, collection=%s",
                self.mongo_database_name,
                self.mongo_draft_collection_name,
            )
            return collection
        except PyMongoError as exc:
            logger.error("MongoDB metadata drafts connection failed: %s", exc)
            raise RuntimeError("MongoDB metadata drafts connection failed") from exc

    def _bootstrap_local_products_once(self) -> None:
        """One-time migration of separate legacy product files when Mongo is empty.

        This never reads the combined ``products.json`` file.  Once documents
        exist in MongoDB, local product JSON files are no longer used by the
        application.
        """
        if self._bootstrapped_local_products:
            return
        self._bootstrapped_local_products = True
        collection = self._product_collection
        if collection is None:
            return
        existing_count = collection.estimated_document_count()
        if existing_count > 0:
            logger.info(
                "MongoDB product migration skipped: %s published products already exist in %s",
                existing_count,
                self.mongo_collection_name,
            )
            return
        logger.info("MongoDB product_metadata is empty; starting one-time migration from separate product JSON files")
        catalog = self._load_catalog()
        records = []
        migration_errors = []
        now = datetime.now(timezone.utc)
        for sequence, (product_name, relative_path) in enumerate(catalog.items()):
            source_path = (self.catalog_dir / relative_path).resolve()
            if not source_path.exists():
                migration_errors.append(f"{product_name}: missing file {source_path}")
                continue
            raw_bytes = source_path.read_bytes()
            try:
                document = json.loads(raw_bytes.decode("utf-8"))
                dataset = document.get("datasets", {}).get(product_name)
            except (UnicodeDecodeError, json.JSONDecodeError):
                migration_errors.append(f"{product_name}: invalid JSON in {source_path}")
                continue
            if not isinstance(dataset, dict):
                migration_errors.append(f"{product_name}: datasets.{product_name} is missing in {source_path}")
                continue
            # Keep the original file text in source_json. This preserves the
            # original JSON key order before any user edits are made.
            records.append({
                "_id": product_name,
                "product_name": product_name,
                "product_key": product_name.strip().upper(),
                "source_json": raw_bytes.decode("utf-8"),
                "version": self._version(raw_bytes),
                "sequence": sequence,
                "created_at": now,
                "updated_at": now,
                "created_by": {"username": "system_migration"},
                "updated_by": {"username": "system_migration"},
            })
        
        if migration_errors:
            logger.error("MongoDB product migration stopped: %s", "; ".join(migration_errors))
            raise RuntimeError("Could not migrate every legacy product JSON file to MongoDB")
        if not records:
            raise RuntimeError("No legacy product JSON files were available for MongoDB migration")
        try:
            collection.insert_many(records, ordered=True)
            logger.info("MongoDB product migration complete: migrated_products=%s", len(records))
        except PyMongoError as exc:
            logger.error("Could not migrate local product JSON files to MongoDB: %s", exc)
            raise RuntimeError("Could not migrate product metadata to MongoDB") from exc

    @staticmethod
    def _serialize_source_document(document: Dict[str, Any]) -> bytes:
        """Serialize without sorting so existing and nested key order survives.

        Never set ``sort_keys=True`` here: indicator/filter order affects the
        intended structure sent to the embedding process.
        """
        return (json.dumps(document, indent=2, ensure_ascii=False, sort_keys=False) + "\n").encode("utf-8")

    def _find_product_record(self, dataset_name: str) -> Dict[str, Any]:
        if not isinstance(dataset_name, str) or not dataset_name.strip():
            raise KeyError("Unknown product")
        collection = self._get_product_collection()
        product_key = dataset_name.strip().upper()
        record = collection.find_one({"product_key": product_key})
        if record is None:
            raise KeyError(f"Unknown product: {dataset_name}")
        return record

    def _read_mongo_product_document(self, dataset_name: str) -> Tuple[Dict[str, Any], bytes, Dict[str, Any], Dict[str, Any]]:
        record = self._find_product_record(dataset_name)
        raw_source = record.get("source_json")
        if not isinstance(raw_source, str):
            raise ValueError(f"Published product {record.get('_id')} has no source_json")
        raw_bytes = raw_source.encode("utf-8")
        try:
            document = json.loads(raw_source)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Published product {record.get('_id')} contains invalid JSON") from exc
        product_name = record["product_name"]
        dataset = document.get("datasets", {}).get(product_name) if isinstance(document, dict) else None
        if not isinstance(dataset, dict):
            raise ValueError(f"Published product {product_name} must contain datasets.{product_name}")
        return record, raw_bytes, document, dataset

    def _product_path(self, dataset_name: str) -> Path:
        catalog = self._load_catalog()
        if dataset_name not in catalog:
            raise KeyError(f"Unknown product: {dataset_name}")

        product_path = (self.catalog_dir / catalog[dataset_name]).resolve()
        # The catalog must never point outside its own products directory.
        if self.catalog_dir not in product_path.parents or product_path.suffix.lower() != ".json":
            raise ValueError(f"Invalid catalog path for product {dataset_name}")
        return product_path

    @staticmethod
    def _version(raw_bytes: bytes) -> str:
        return hashlib.sha256(raw_bytes).hexdigest()

    def _load_product_document(self, dataset_name: str) -> Tuple[Path, bytes, Dict[str, Any], Dict[str, Any]]:
        path = self._product_path(dataset_name)
        try:
            raw_bytes = path.read_bytes()
            document = json.loads(raw_bytes.decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise ValueError(f"Product file is not UTF-8: {path.name}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path.name}: {exc}") from exc

        datasets = document.get("datasets") if isinstance(document, dict) else None
        if not isinstance(datasets, dict) or dataset_name not in datasets:
            raise ValueError(f"{path.name} must contain datasets.{dataset_name}")
        dataset = datasets[dataset_name]
        if not isinstance(dataset, dict):
            raise ValueError(f"datasets.{dataset_name} in {path.name} must be an object")
        return path, raw_bytes, document, dataset

    def list_datasets(self) -> List[str]:
        """Return published Mongo products in their stable creation order."""
        collection = self._get_product_collection()
        return [
            record["product_name"]
            for record in collection.find({}, {"product_name": 1}).sort([( "sequence", 1), ("product_name", 1)])
            if isinstance(record.get("product_name"), str)
        ]

    def load_all_datasets(self) -> Dict[str, Dict[str, Any]]:
        """Load every published Mongo product in its stable order."""
        datasets: Dict[str, Dict[str, Any]] = {}
        for dataset_name in self.list_datasets():
            _, _, _, dataset = self._read_mongo_product_document(dataset_name)
            datasets[dataset_name] = dataset
        return datasets

    def get_dataset(self, dataset_name: str) -> Dict[str, Any]:
        """Return one published Mongo product plus its optimistic-lock version."""
        record, raw_bytes, _, dataset = self._read_mongo_product_document(dataset_name)
        return {
            "data": deepcopy(dataset),
            "version": record.get("version") or self._version(raw_bytes),
            "product_name": record["product_name"],
            "created_at": record.get("created_at"),
            "updated_at": record.get("updated_at"),
            "created_by": deepcopy(record.get("created_by")),
            "updated_by": deepcopy(record.get("updated_by")),
        }

    def load_single_dataset(self, dataset_name: str) -> Dict[str, Any]:
        """Compatibility helper returning only the selected dataset object."""
        return self.get_dataset(dataset_name)["data"]

    def load_metadata(self) -> Dict[str, Any]:
        """Compatibility helper that returns an in-memory aggregate only.

        It never writes a combined products.json file.
        """
        return {"datasets": self.load_all_datasets()}

    # ------------------------------------------------------------------
    # Validation and order-preserving write support
    # ------------------------------------------------------------------
    @staticmethod
    def _merge_preserving_order(old_value: Any, new_value: Any) -> Any:
        """Apply new values while retaining existing object-key order.

        Lists always use the order supplied by the UI.  This preserves the
        existing order for ordinary edits and permits intentional add/remove
        operations.  Existing dictionary keys retain their original order;
        new keys are appended at that exact nested level.
        """
        if isinstance(old_value, dict) and isinstance(new_value, dict):
            # First retain every existing key in its original position.
            # Any new UI key is appended only after the old keys.
            merged: Dict[str, Any] = {}
            for key, old_child in old_value.items():
                if key in new_value:
                    merged[key] = MetadataManager._merge_preserving_order(old_child, new_value[key])
            for key, new_child in new_value.items():
                if key not in old_value:
                    merged[key] = deepcopy(new_child)
            return merged

        if isinstance(old_value, list) and isinstance(new_value, list):
            # Lists are intentionally never sorted. The UI order is the
            # product order for indicators, filters, and filter values.
            merged_list: List[Any] = []
            for index, new_child in enumerate(new_value):
                if index < len(old_value):
                    merged_list.append(MetadataManager._merge_preserving_order(old_value[index], new_child))
                else:
                    merged_list.append(deepcopy(new_child))
            return merged_list

        return deepcopy(new_value)

    @staticmethod
    def _validate_filter_value(value: Any, context: str) -> Tuple[bool, str]:
        if isinstance(value, list):
            if not value:
                return False, f"{context} has no values"
            for index, child in enumerate(value):
                valid, message = MetadataManager._validate_filter_value(child, f"{context}[{index}]")
                if not valid:
                    return valid, message
            return True, ""

        if isinstance(value, dict):
            if not value:
                return False, f"{context} is an empty nested object"
            for key, child in value.items():
                if not isinstance(key, str) or not key.strip():
                    return False, f"{context} contains an empty nested filter name"
                valid, message = MetadataManager._validate_filter_value(child, f"{context}.{key}")
                if not valid:
                    return valid, message
            return True, ""

        if value is None or (isinstance(value, str) and not value.strip()):
            return False, f"{context} contains an empty value"
        return True, ""

    def validate_dataset_structure(self, dataset_data: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate flat and arbitrarily nested filter structures safely."""
        if not isinstance(dataset_data, dict):
            return False, "Dataset must be an object"

        indicators = dataset_data.get("indicators", [])
        if not isinstance(indicators, list):
            return False, "Indicators must be a list"

        for index, indicator in enumerate(indicators):
            if not isinstance(indicator, dict):
                return False, f"Indicator #{index + 1} must be an object"
            if not isinstance(indicator.get("name"), str) or not indicator["name"].strip():
                return False, f"Indicator #{index + 1} must have a name"

            filters = indicator.get("filters", [])
            if not isinstance(filters, list):
                return False, f"Filters in indicator '{indicator['name']}' must be a list"
            for filter_index, filter_object in enumerate(filters):
                if not isinstance(filter_object, dict) or not filter_object:
                    return False, f"Filter #{filter_index + 1} in '{indicator['name']}' must be a non-empty object"
                for filter_name, filter_value in filter_object.items():
                    if not isinstance(filter_name, str) or not filter_name.strip() or filter_name == "New_Filter":
                        return False, f"Filter #{filter_index + 1} in '{indicator['name']}' has an invalid name"
                    valid, message = self._validate_filter_value(filter_value, f"Filter '{filter_name}'")
                    if not valid:
                        return valid, message
        return True, ""

    @staticmethod
    def _normalise_product_name(product_name: Any) -> str:
        """Return a safe upper-case product name suitable for files and catalog keys."""
        if not isinstance(product_name, str):
            raise ValueError("product_name is required")
        name = product_name.strip().upper()
        if not re.fullmatch(r"[A-Z][A-Z0-9_]{1,49}", name):
            raise ValueError(
                "Product name must start with a letter and contain only upper-case letters, numbers, or underscores"
            )
        return name

    @staticmethod
    def _new_product_dataset(dataset_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build a new source dataset with a predictable key order."""
        return {
            "description": deepcopy(dataset_data.get("description", "")),
            "indicators": deepcopy(dataset_data.get("indicators", [])),
        }

    def _validate_publishable_dataset(self, dataset_data: Dict[str, Any]) -> Tuple[bool, str]:
        valid, message = self.validate_dataset_structure(dataset_data)
        if not valid:
            return valid, message
        if not isinstance(dataset_data.get("description"), str) or not dataset_data["description"].strip():
            return False, "Product description is required before publishing"
        if not dataset_data.get("indicators"):
            return False, "Add at least one indicator before publishing"
        for index, indicator in enumerate(dataset_data["indicators"]):
            if not isinstance(indicator.get("description"), str) or not indicator["description"].strip():
                return False, f"Indicator #{index + 1} must have a description before publishing"
        return True, ""

    # ------------------------------------------------------------------
    # Draft operations. Draft documents are in a separate Mongo collection,
    # are scoped to their owner, and are never read by load_all_datasets().
    # Therefore unfinished metadata cannot affect AI search.
    # ------------------------------------------------------------------
    @staticmethod
    def _actor_is_admin(actor: Any) -> bool:
        """Return whether a safe auth user has administrator access."""
        if not isinstance(actor, dict):
            return False
        roles = actor.get("roles", actor.get("role", []))
        if isinstance(roles, str):
            roles = [roles]
        return isinstance(roles, (list, tuple, set)) and "admin" in roles

    @classmethod
    def _draft_owner(cls, actor: Any) -> Tuple[Dict[str, str], str]:
        """Create a safe owner snapshot and stable key for unique drafts."""
        owner = cls._actor_snapshot(actor)
        owner_key = owner.get("user_id") or owner["username"].casefold()
        return owner, owner_key

    def _assert_draft_access(self, draft: Dict[str, Any], actor: Any) -> None:
        """Allow the owner or an admin; local auth-disabled mode stays usable."""
        if actor is None or self._actor_is_admin(actor):
            return
        owner_id = str((draft.get("owner") or {}).get("user_id") or "")
        actor_id = str(actor.get("id") or actor.get("_id") or "") if isinstance(actor, dict) else ""
        if not owner_id or owner_id != actor_id:
            raise PermissionError("You can access only your own drafts")

    def _read_draft_record(self, draft_id: str) -> Tuple[Dict[str, Any], bytes, str, Dict[str, Any]]:
        if not isinstance(draft_id, str) or not draft_id.strip():
            raise KeyError("Unknown draft")
        draft = self._get_draft_collection().find_one({"draft_id": draft_id.strip()})
        if draft is None:
            raise KeyError("Unknown draft")
        raw_source = draft.get("source_json")
        if not isinstance(raw_source, str):
            raise ValueError("Draft has no source_json")
        raw_bytes = raw_source.encode("utf-8")
        try:
            document = json.loads(raw_source)
        except json.JSONDecodeError as exc:
            raise ValueError("Draft contains invalid JSON") from exc
        product_name = str(draft.get("product_name") or "")
        datasets = document.get("datasets") if isinstance(document, dict) else None
        dataset_data = datasets.get(product_name) if isinstance(datasets, dict) else None
        if not isinstance(dataset_data, dict):
            raise ValueError("Draft JSON must contain one matching product")
        return draft, raw_bytes, product_name, dataset_data

    def _draft_response(self, draft: Dict[str, Any], raw_bytes: bytes, dataset_data: Dict[str, Any]) -> Dict[str, Any]:
        """Return editor data without exposing internal MongoDB fields."""
        return {
            "draft_id": draft["draft_id"],
            "status": "draft",
            "product_name": draft["product_name"],
            "dataset_data": deepcopy(dataset_data),
            "created_at": draft.get("created_at"),
            "updated_at": draft.get("updated_at"),
            "version": draft.get("version") or self._version(raw_bytes),
            "created_by": deepcopy(draft.get("created_by")),
            "updated_by": deepcopy(draft.get("updated_by")),
            "owner": deepcopy(draft.get("owner")),
        }

    def _migrate_legacy_drafts_once(self) -> None:
        """Copy old local drafts to MongoDB without deleting the originals.

        Historic files have no authenticated owner. They are labelled
        ``system_migration`` and, consequently, are visible to administrators
        for reassignment or publishing. This one-way safe copy prevents an
        upgrade from silently hiding existing drafts.
        """
        if self._legacy_drafts_migrated:
            return
        self._legacy_drafts_migrated = True
        collection = self._get_draft_collection()
        owner = {"username": "system_migration"}
        imported = 0
        for path in self.drafts_dir.glob("*.json"):
            try:
                raw_document = self._read_json(path)
                if raw_document.get("status") == "draft":
                    product_name = self._normalise_product_name(
                        raw_document.get("product_code") or raw_document.get("product_name")
                    )
                    dataset_data = raw_document.get("dataset_data")
                else:
                    datasets = raw_document.get("datasets") if isinstance(raw_document, dict) else None
                    if not isinstance(datasets, dict) or len(datasets) != 1:
                        raise ValueError("not a source-shaped draft")
                    product_name = self._normalise_product_name(next(iter(datasets)))
                    dataset_data = datasets[product_name]
                self._validate_draft_shape(dataset_data)
                cleaned_dataset = self._clean_draft_dataset(dataset_data)
                raw_bytes = self._serialize_source_document({"datasets": {product_name: cleaned_dataset}})
                stat = path.stat()
                document = {
                    "_id": str(uuid.uuid4()),
                    "draft_id": str(uuid.uuid4()),
                    "status": "draft",
                    "product_name": product_name,
                    "product_key": product_name,
                    "owner": owner,
                    "owner_key": "system_migration",
                    "source_json": raw_bytes.decode("utf-8"),
                    "version": self._version(raw_bytes),
                    "created_at": datetime.fromtimestamp(stat.st_ctime, timezone.utc),
                    "updated_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc),
                    "created_by": owner,
                    "updated_by": owner,
                }
                try:
                    collection.insert_one(document)
                    imported += 1
                except DuplicateKeyError:
                    # Repeated application starts must not create a second
                    # copy of the same old local draft.
                    continue
            except (OSError, ValueError, TypeError):
                logger.warning("Skipping malformed legacy metadata draft: %s", path.name)
        if imported:
            logger.info("Migrated legacy metadata drafts to MongoDB: count=%s", imported)

    @staticmethod
    def _validate_draft_shape(dataset_data: Any) -> None:
        """Drafts may be incomplete, but must remain valid JSON editor data."""
        if not isinstance(dataset_data, dict):
            raise ValueError("dataset_data must be an object")
        indicators = dataset_data.get("indicators", [])
        if not isinstance(indicators, list):
            raise ValueError("Indicators must be a list")
        for index, indicator in enumerate(indicators):
            if not isinstance(indicator, dict):
                raise ValueError(f"Indicator #{index + 1} must be an object")
            if "filters" in indicator and not isinstance(indicator["filters"], list):
                raise ValueError(f"Filters in indicator #{index + 1} must be a list")

    @staticmethod
    def _clean_draft_dataset(dataset_data: Dict[str, Any]) -> Dict[str, Any]:
        """Remove empty editor rows before a draft is written to disk.

        Drafts can be incomplete, but a click on “Add indicator” or “Add
        filter” must not leave placeholder objects in the saved JSON.
        """
        cleaned_indicators: List[Dict[str, Any]] = []
        for indicator in dataset_data.get("indicators", []):
            name = str(indicator.get("name") or "").strip()
            description = str(indicator.get("description") or "").strip()
            cleaned_filters: List[Dict[str, Any]] = []
            for filter_object in indicator.get("filters", []):
                if not isinstance(filter_object, dict):
                    continue
                for filter_name, raw_values in filter_object.items():
                    clean_name = str(filter_name or "").strip()
                    if not clean_name or not isinstance(raw_values, list):
                        continue
                    clean_values = [
                        value.strip() if isinstance(value, str) else value
                        for value in raw_values
                        if value is not None and (not isinstance(value, str) or value.strip())
                    ]
                    if clean_values:
                        cleaned_filters.append({clean_name: clean_values})
            # Completely blank rows are UI placeholders, not metadata.
            if name or description or cleaned_filters:
                cleaned_indicators.append({
                    "name": name,
                    "description": description,
                    "filters": cleaned_filters,
                })
        return {
            "description": str(dataset_data.get("description") or "").strip(),
            "indicators": cleaned_indicators,
        }

    def list_drafts(self, actor: Any = None) -> List[Dict[str, Any]]:
        """List only the current user's drafts, unless they are an admin."""
        collection = self._get_draft_collection()
        self._migrate_legacy_drafts_once()
        query: Dict[str, Any] = {}
        if actor is not None and not self._actor_is_admin(actor):
            _, owner_key = self._draft_owner(actor)
            query["owner_key"] = owner_key
        drafts: List[Dict[str, Any]] = []
        for draft in collection.find(query).sort("updated_at", DESCENDING):
            try:
                _, raw_bytes, _, dataset_data = self._read_draft_record(str(draft.get("draft_id") or ""))
                drafts.append(self._draft_response(draft, raw_bytes, dataset_data))
            except ValueError:
                logger.warning("Skipping malformed Mongo metadata draft: %s", draft.get("draft_id"))
        return drafts

    def get_draft(self, draft_id: str, actor: Any = None) -> Dict[str, Any]:
        draft, raw_bytes, _, dataset_data = self._read_draft_record(draft_id)
        self._assert_draft_access(draft, actor)
        return self._draft_response(draft, raw_bytes, dataset_data)

    def create_draft(self, product_name: Any, dataset_data: Any, actor: Any = None) -> Dict[str, Any]:
        """Save a user-owned draft in MongoDB, outside published metadata."""
        name = self._normalise_product_name(product_name)
        self._validate_draft_shape(dataset_data)
        cleaned_dataset = self._clean_draft_dataset(dataset_data)
        raw_bytes = self._serialize_source_document({"datasets": {name: cleaned_dataset}})
        owner, owner_key = self._draft_owner(actor)
        now = datetime.now(timezone.utc)
        draft = {
            "_id": str(uuid.uuid4()),
            "draft_id": str(uuid.uuid4()),
            "status": "draft",
            "product_name": name,
            "product_key": name,
            "owner": owner,
            "owner_key": owner_key,
            "source_json": raw_bytes.decode("utf-8"),
            "version": self._version(raw_bytes),
            "created_at": now,
            "updated_at": now,
            "created_by": owner,
            "updated_by": owner,
        }
        try:
            self._get_draft_collection().insert_one(draft)
        except DuplicateKeyError as exc:
            raise ValueError(f"You already have a draft named {name}. Open it or choose another product name.") from exc
        except PyMongoError as exc:
            raise RuntimeError("Could not save metadata draft to MongoDB") from exc
        logger.info("MongoDB metadata draft created: product=%s owner=%s", name, owner.get("username"))
        return self._draft_response(draft, raw_bytes, cleaned_dataset)

    def update_draft(
        self,
        draft_id: str,
        product_name: Any,
        dataset_data: Any,
        expected_version: str | None = None,
        actor: Any = None,
    ) -> Dict[str, Any]:
        """Update one owner-authorized draft while retaining JSON list order."""
        name = self._normalise_product_name(product_name)
        self._validate_draft_shape(dataset_data)
        draft, raw_bytes, _, _ = self._read_draft_record(draft_id)
        self._assert_draft_access(draft, actor)
        current_version = draft.get("version") or self._version(raw_bytes)
        if expected_version and expected_version != current_version:
            raise MetadataConflictError("This draft was changed elsewhere. Reload it before saving.")
        cleaned_dataset = self._clean_draft_dataset(dataset_data)
        new_raw_bytes = self._serialize_source_document({"datasets": {name: cleaned_dataset}})
        new_version = self._version(new_raw_bytes)
        actor_snapshot = self._actor_snapshot(actor)
        try:
            write_result = self._get_draft_collection().update_one(
                {"_id": draft["_id"], "version": current_version},
                {"$set": {
                    "product_name": name,
                    "product_key": name,
                    "source_json": new_raw_bytes.decode("utf-8"),
                    "version": new_version,
                    "updated_at": datetime.now(timezone.utc),
                    "updated_by": actor_snapshot,
                }},
            )
        except DuplicateKeyError as exc:
            raise ValueError(f"You already have a draft named {name}. Open it or choose another product name.") from exc
        except PyMongoError as exc:
            raise RuntimeError("Could not save metadata draft to MongoDB") from exc
        if write_result.matched_count != 1:
            raise MetadataConflictError("This draft was changed elsewhere. Reload it before saving.")
        draft.update({
            "product_name": name,
            "product_key": name,
            "source_json": new_raw_bytes.decode("utf-8"),
            "version": new_version,
            "updated_at": datetime.now(timezone.utc),
            "updated_by": actor_snapshot,
        })
        logger.info("MongoDB metadata draft updated: product=%s owner=%s", name, (draft.get("owner") or {}).get("username"))
        return self._draft_response(draft, new_raw_bytes, cleaned_dataset)

    def delete_draft(self, draft_id: str, actor: Any = None) -> None:
        draft, _, _, _ = self._read_draft_record(draft_id)
        self._assert_draft_access(draft, actor)
        try:
            result = self._get_draft_collection().delete_one({"_id": draft["_id"]})
        except PyMongoError as exc:
            raise RuntimeError("Could not delete metadata draft") from exc
        if result.deleted_count != 1:
            raise KeyError("Unknown draft")
        logger.info("MongoDB metadata draft deleted: id=%s", draft_id)

    def publish_draft(
        self,
        draft_id: str,
        expected_version: str | None = None,
        user_ip: str | None = None,
        actor: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Publish one exact user-authorized draft into product_metadata."""
        draft, raw_bytes, product_name, dataset_data = self._read_draft_record(draft_id)
        self._assert_draft_access(draft, actor)
        current_version = draft.get("version") or self._version(raw_bytes)
        if expected_version and expected_version != current_version:
            raise MetadataConflictError("This draft was changed elsewhere. Reload it before publishing.")

        code = product_name
        valid, message = self._validate_publishable_dataset(dataset_data)
        if not valid:
            raise ValueError(message)

        collection = self._get_product_collection()
        if collection.find_one({"product_key": code}):
            raise ValueError(f"Product code {code} already exists. Use Update Metadata for that product.")
        # The JSON below is the ONLY part later used by AI search. Mongo's
        # product_name/version fields remain outside it and cannot confuse AI.
        product_document = {"datasets": {code: self._new_product_dataset(dataset_data)}}
        raw_source = self._serialize_source_document(product_document)
        now = datetime.now(timezone.utc)
        actor_snapshot = self._actor_snapshot(actor)
        try:
            last_record = next(collection.find({}, {"sequence": 1}).sort("sequence", -1).limit(1), None)
            sequence = int(last_record.get("sequence", -1)) + 1 if last_record else 0
            collection.insert_one({
                "_id": code,
                "product_name": code,
                "product_key": code,
                "source_json": raw_source.decode("utf-8"),
                "version": self._version(raw_source),
                "sequence": sequence,
                "created_at": now,
                "updated_at": now,
                "created_by": actor_snapshot,
                "updated_by": actor_snapshot,
            })
        except PyMongoError as exc:
            raise RuntimeError(f"Could not publish {code} to MongoDB") from exc

        try:
            self._get_draft_collection().delete_one({"_id": draft["_id"], "version": current_version})
        except PyMongoError as exc:
            logger.warning("Published %s but could not remove Mongo draft %s: %s", code, draft_id, exc)

        audit_entry = self._create_audit_entry(
            dataset_name=code,
            change_summary=f"Published new product {code}",
            changes_detected=[f"Created product: {code}", "Saved published JSON to MongoDB product_metadata"],
            indicators_changed=[
                {"action": "added", "name": indicator.get("name", "Unnamed indicator")}
                for indicator in dataset_data.get("indicators", [])
                if isinstance(indicator, dict)
            ],
            backup_file=None,
            user_ip=user_ip,
            actor=actor_snapshot,
        )
        self._append_audit_log(audit_entry)
        return {
            "success": True,
            "message": f"Published {code}",
            "product_code": code,
            "product_file": None,
            "indicator_count": len(dataset_data["indicators"]),
            "created_by": actor_snapshot,
            "updated_by": actor_snapshot,
        }

    def _create_backup(self, dataset_name: str, source_path: Path) -> str | None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        backup_path = self.backup_dir / f"{dataset_name}_{timestamp}.json"
        try:
            shutil.copy2(source_path, backup_path)
            return str(backup_path)
        except OSError as exc:
            logger.warning("Could not create metadata backup for %s: %s", dataset_name, exc)
            return None

    @staticmethod
    def _atomic_write(path: Path, data: Dict[str, Any]) -> None:
        """Write one product document atomically without sorting its keys."""
        temporary_name = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
            ) as temporary_file:
                temporary_name = temporary_file.name
                json.dump(data, temporary_file, indent=2, ensure_ascii=False, sort_keys=False)
                temporary_file.write("\n")
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_name, path)
        except Exception:
            if temporary_name and os.path.exists(temporary_name):
                os.unlink(temporary_name)
            raise

    # ------------------------------------------------------------------
    # Update and audit operations
    # ------------------------------------------------------------------
    def detect_changes(self, old_dataset: Dict[str, Any], new_dataset: Dict[str, Any]) -> Tuple[List[str], List[Dict[str, Any]]]:
        changes: List[str] = []
        indicators_changed: List[Dict[str, Any]] = []
        if old_dataset.get("description") != new_dataset.get("description"):
            changes.append("description")

        old_indicators = old_dataset.get("indicators", [])
        new_indicators = new_dataset.get("indicators", [])
        common_count = min(len(old_indicators), len(new_indicators))
        for index in range(common_count):
            old_indicator, new_indicator = old_indicators[index], new_indicators[index]
            if not isinstance(old_indicator, dict) or not isinstance(new_indicator, dict):
                continue
            name = new_indicator.get("name", f"Indicator #{index + 1}")
            if old_indicator.get("name") != new_indicator.get("name"):
                changes.append(f"Renamed indicator: {name}")
                indicators_changed.append({"action": "modified", "name": name, "field": "name"})
            if old_indicator.get("description") != new_indicator.get("description"):
                changes.append(f"Modified description in: {name}")
                indicators_changed.append({"action": "modified", "name": name, "field": "description"})
            if old_indicator.get("filters") != new_indicator.get("filters"):
                changes.append(f"Modified filters in: {name}")
                indicators_changed.append({"action": "modified", "name": name, "field": "filters"})

        for indicator in new_indicators[common_count:]:
            name = indicator.get("name", "Unnamed indicator") if isinstance(indicator, dict) else "Unnamed indicator"
            changes.append(f"Added indicator: {name}")
            indicators_changed.append({"action": "added", "name": name})
        for indicator in old_indicators[common_count:]:
            name = indicator.get("name", "Unnamed indicator") if isinstance(indicator, dict) else "Unnamed indicator"
            changes.append(f"Removed indicator: {name}")
            indicators_changed.append({"action": "removed", "name": name})
        return changes, indicators_changed

    def check_dataset_changes(
        self,
        dataset_name: str,
        new_dataset_data: Dict[str, Any],
        expected_version: str | None = None,
    ) -> Dict[str, Any]:
        """Check an editor save without writing MongoDB or refreshing vectors.

        The comparison is against the merged, order-preserving structure used
        by the real save path.  Therefore a form that is semantically and
        structurally unchanged returns ``changed=False`` and Qdrant/FAISS are
        never invoked.
        """
        valid, message = self.validate_dataset_structure(new_dataset_data)
        if not valid:
            raise ValueError(message)
        record, raw_bytes, _, old_dataset = self._read_mongo_product_document(dataset_name)
        current_version = record.get("version") or self._version(raw_bytes)
        if expected_version and expected_version != current_version:
            raise MetadataConflictError(
                f"{record['product_name']} was changed by another user. Reload the product before saving."
            )
        merged_dataset = self._merge_preserving_order(old_dataset, new_dataset_data)
        return {
            "changed": merged_dataset != old_dataset,
            "product_name": record["product_name"],
            "version": current_version,
        }

    def update_dataset_preserving_structure(
        self,
        dataset_name: str,
        new_dataset_data: Dict[str, Any],
        change_summary: str | None = None,
        user_ip: str | None = None,
        expected_version: str | None = None,
        actor: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Update one Mongo product while preserving all existing nested key order."""
        valid, message = self.validate_dataset_structure(new_dataset_data)
        if not valid:
            raise ValueError(message)
        record, raw_bytes, document, old_dataset = self._read_mongo_product_document(dataset_name)
        product_name = record["product_name"]
        current_version = record.get("version") or self._version(raw_bytes)
        if expected_version and expected_version != current_version:
            raise MetadataConflictError(
                f"{product_name} was changed by another user. Reload the product before saving."
            )

        merged_dataset = self._merge_preserving_order(old_dataset, new_dataset_data)
        changes, indicators_changed = self.detect_changes(old_dataset, merged_dataset)
        document["datasets"][product_name] = merged_dataset
        new_raw_bytes = self._serialize_source_document(document)
        new_version = self._version(new_raw_bytes)
        collection = self._get_product_collection()
        actor_snapshot = self._actor_snapshot(actor)
        created_by = record.get("created_by") or {"username": "system_migration"}
        try:
            write_result = collection.update_one(
                {"_id": record["_id"], "version": current_version},
                {"$set": {
                    "source_json": new_raw_bytes.decode("utf-8"),
                    "version": new_version,
                    "updated_at": datetime.now(timezone.utc),
                    "created_by": created_by,
                    "updated_by": actor_snapshot,
                }},
            )
        except PyMongoError as exc:
            raise RuntimeError(f"Could not save {product_name} to MongoDB") from exc
        if write_result.matched_count != 1:
            raise MetadataConflictError(f"{product_name} was changed by another user. Reload the product before saving.")

       
        audit_entry = self._create_audit_entry(
            dataset_name=product_name,
            change_summary=change_summary or f"Updated {product_name} metadata",
            changes_detected=changes or ["No changes detected"],
            indicators_changed=indicators_changed,
            backup_file=None,
            user_ip=user_ip,
            actor=actor_snapshot,
        )
        self._append_audit_log(audit_entry)

        return {
            "success": True,
            "message": f"Metadata updated successfully for {product_name}",
            "changes_detected": changes or ["No changes detected"],
            "indicators_modified": indicators_changed,
            "backup_file": None,
            "version": new_version,
            "updated_by": actor_snapshot,
        }

    def download_dataset_source(self, dataset_name: str) -> Dict[str, Any]:
        """Return one clean published product JSON for a user download.

        The response contains the same source-shaped JSON used for embedding.
        Mongo audit, ownership, and version fields are deliberately excluded.
        """
        record, raw_bytes, _, _ = self._read_mongo_product_document(dataset_name)
        return {
            "product_name": record["product_name"],
            "source_json": raw_bytes.decode("utf-8"),
            "version": record.get("version") or self._version(raw_bytes),
        }

    def replace_dataset_from_source_json(
        self,
        dataset_name: str,
        source_json: str,
        *,
        expected_version: str | None = None,
        user_ip: str | None = None,
        actor: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Safely replace one product from a downloaded-and-edited JSON file.

        The uploaded document must contain exactly ``datasets.<product>``. It
        is serialized without sorted keys, so the editor's indicator, filter,
        and option ordering remains exactly as supplied by the user.
        """
        if not isinstance(source_json, str) or not source_json.strip():
            raise ValueError("Upload a non-empty JSON metadata file")
        try:
            uploaded_document = json.loads(source_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Uploaded file is not valid JSON: {exc.msg}") from exc
        if not isinstance(uploaded_document, dict):
            raise ValueError("Uploaded JSON must contain an object")

        record, raw_bytes, _, old_dataset = self._read_mongo_product_document(dataset_name)
        product_name = record["product_name"]
        datasets = uploaded_document.get("datasets")
        if not isinstance(datasets, dict) or list(datasets.keys()) != [product_name]:
            raise ValueError(
                f"Uploaded JSON must contain exactly one product at datasets.{product_name}"
            )
        uploaded_dataset = datasets[product_name]
        valid, message = self.validate_dataset_structure(uploaded_dataset)
        if not valid:
            raise ValueError(message)

        current_version = record.get("version") or self._version(raw_bytes)
        if expected_version and expected_version != current_version:
            raise MetadataConflictError(
                f"{product_name} was changed by another user. Download a new copy before replacing it."
            )

        # Re-serialize the parsed document to reject duplicate JSON keys while
        # retaining the field/list order in the uploaded JSON object.
        new_raw_bytes = self._serialize_source_document(uploaded_document)
        new_version = self._version(new_raw_bytes)
        changes, indicators_changed = self.detect_changes(old_dataset, uploaded_dataset)
        actor_snapshot = self._actor_snapshot(actor)
        try:
            write_result = self._get_product_collection().update_one(
                {"_id": record["_id"], "version": current_version},
                {"$set": {
                    "source_json": new_raw_bytes.decode("utf-8"),
                    "version": new_version,
                    "updated_at": datetime.now(timezone.utc),
                    "updated_by": actor_snapshot,
                }},
            )
        except PyMongoError as exc:
            raise RuntimeError(f"Could not replace {product_name} in MongoDB") from exc
        if write_result.matched_count != 1:
            raise MetadataConflictError(
                f"{product_name} was changed by another user. Download a new copy before replacing it."
            )

        audit_entry = self._create_audit_entry(
            dataset_name=product_name,
            change_summary=f"Replaced {product_name} from uploaded JSON",
            changes_detected=changes or ["Replaced source JSON (no content changes detected)"],
            indicators_changed=indicators_changed,
            backup_file=None,
            user_ip=user_ip,
            actor=actor_snapshot,
        )
        self._append_audit_log(audit_entry)
        logger.info("Published metadata replaced from uploaded JSON: product=%s actor=%s", product_name, actor_snapshot["username"])
        return {
            "success": True,
            "message": f"Metadata replaced successfully for {product_name}",
            "changes_detected": changes or ["Replaced source JSON (no content changes detected)"],
            "indicators_modified": indicators_changed,
            "version": new_version,
            "updated_by": actor_snapshot,
        }

    def update_single_dataset_optimized(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Backward-compatible name; per-product writes are already optimized."""
        return self.update_dataset_preserving_structure(*args, **kwargs)

    def update_indicator(self, dataset_name: str, indicator_index: int, updated_indicator: Dict[str, Any]) -> Dict[str, Any]:
        dataset = self.load_single_dataset(dataset_name)
        indicators = dataset.get("indicators", [])
        if not 0 <= indicator_index < len(indicators):
            raise ValueError(f"Invalid indicator index: {indicator_index}")
        indicators[indicator_index] = updated_indicator
        return self.update_dataset_preserving_structure(dataset_name, dataset)

    def add_indicator(self, dataset_name: str, new_indicator: Dict[str, Any]) -> Dict[str, Any]:
        dataset = self.load_single_dataset(dataset_name)
        dataset.setdefault("indicators", []).append(new_indicator)
        return self.update_dataset_preserving_structure(dataset_name, dataset)

    def remove_indicator(self, dataset_name: str, indicator_index: int) -> Dict[str, Any]:
        dataset = self.load_single_dataset(dataset_name)
        indicators = dataset.get("indicators", [])
        if not 0 <= indicator_index < len(indicators):
            raise ValueError(f"Invalid indicator index: {indicator_index}")
        indicators.pop(indicator_index)
        return self.update_dataset_preserving_structure(dataset_name, dataset)

    def update_filter(self, dataset_name: str, indicator_index: int, filter_index: int, updated_filter: Dict[str, Any]) -> Dict[str, Any]:
        dataset = self.load_single_dataset(dataset_name)
        indicators = dataset.get("indicators", [])
        if not 0 <= indicator_index < len(indicators):
            raise ValueError(f"Invalid indicator index: {indicator_index}")
        filters = indicators[indicator_index].get("filters", [])
        if not 0 <= filter_index < len(filters):
            raise ValueError(f"Invalid filter index: {filter_index}")
        filters[filter_index] = updated_filter
        return self.update_dataset_preserving_structure(dataset_name, dataset)

    # ------------------------------------------------------------------
    # Product-specific Qdrant indexing.  This code intentionally lives in
    # MetadataManager so Flask routes remain transport-only.
    # ------------------------------------------------------------------
    def configure_indexing(self, embedding_model: Any, qdrant_client: Any, qdrant_models: Any, collection_name: str) -> None:
        # Keep Qdrant dependencies here, not in Flask routes. app.py only
        # supplies already-created clients and receives API requests.
        self._embedding_model = embedding_model
        self._qdrant_client = qdrant_client
        self._qdrant_models = qdrant_models
        self._qdrant_collection_name = collection_name
        self._qdrant_collection_ready = False
        logger.info("Qdrant product indexing configured for collection=%s", collection_name)

    @staticmethod
    def _clean_embedding_text(value: Any) -> str:
        cleaned = re.sub(r"[^a-z0-9 ]", " ", str(value or "").lower())
        return re.sub(r"\s+", " ", cleaned).strip()

    def _ensure_qdrant_collection(self) -> bool:
        """Create the shared Qdrant collection once, if it is missing.

        Returns ``True`` only when this call created the collection. Network
        and authentication errors are deliberately allowed to propagate: a
        failed Qdrant connection must stop indexing rather than look like an
        empty collection that can safely be recreated.
        """
        if not self._qdrant_client or not self._qdrant_models or not self._embedding_model:
            raise RuntimeError("Qdrant product indexing is not configured")
        if self._qdrant_collection_ready:
            return False
        if self._qdrant_client.collection_exists(self._qdrant_collection_name):
            self._qdrant_collection_ready = True
            logger.info("Qdrant collection already exists: %s", self._qdrant_collection_name)
            return False
        logger.info(
            "Creating Qdrant collection: collection=%s, vector_size=%s, distance=cosine",
            self._qdrant_collection_name,
            self._embedding_model.get_sentence_embedding_dimension(),
        )
        try:
            self._qdrant_client.create_collection(
                collection_name=self._qdrant_collection_name,
                vectors_config=self._qdrant_models.VectorParams(
                    size=self._embedding_model.get_sentence_embedding_dimension(),
                    distance=self._qdrant_models.Distance.COSINE,
                ),
            )
        except Exception:
            if not self._qdrant_client.collection_exists(self._qdrant_collection_name):
                logger.exception("Could not create Qdrant collection %s", self._qdrant_collection_name)
                raise
            logger.info("Qdrant collection was created by another application worker: %s", self._qdrant_collection_name)
        self._qdrant_collection_ready = True
        logger.info("Qdrant collection ready: %s", self._qdrant_collection_name)
        return True

    def index_product(self, product_name: str) -> Dict[str, Any]:
        """Replace vectors for one published product without touching other products."""
        record, _, _, dataset = self._read_mongo_product_document(product_name)
        indicators = dataset.get("indicators", [])
        if not isinstance(indicators, list) or not indicators:
            raise ValueError(f"{record['product_name']} has no indicators to index")
        self._ensure_qdrant_collection()
        logger.info(
            "Qdrant product indexing started: product=%s, indicators=%s, source_version=%s",
            record["product_name"],
            len(indicators),
            record.get("version"),
        )
        embedding_texts = []
        for indicator in indicators:
            if not isinstance(indicator, dict):
                embedding_texts.append("")
                continue
            name = self._clean_embedding_text(indicator.get("name"))
            description = self._clean_embedding_text(indicator.get("description"))
            embedding_texts.append(" ".join(part for part in (name, description) if part))

        vectors = self._embedding_model.encode(
            embedding_texts,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.clip(norms, 1e-12, None)
        updated_at = record.get("updated_at") or record.get("created_at")
        if isinstance(updated_at, datetime):
            last_updated_on = updated_at.isoformat()
        elif updated_at is None:
            last_updated_on = None
        else:
            last_updated_on = str(updated_at)

        product_filter = self._qdrant_models.Filter(
            must=[self._qdrant_models.FieldCondition(
                key="product",
                match=self._qdrant_models.MatchValue(value=record["product_name"]),
            )]
        )
        self._qdrant_client.delete(
            collection_name=self._qdrant_collection_name,
            points_selector=self._qdrant_models.FilterSelector(filter=product_filter),
            wait=True,
        )

        points = []
        for index, indicator in enumerate(indicators):
            if not isinstance(indicator, dict):
                continue
            points.append(self._qdrant_models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{record['product_key']}:{index}")),
                vector=vectors[index].astype("float32").tolist(),
                payload={
                    "product": record["product_name"],
                    "product_desc": dataset.get("description", ""),
                    "name": indicator.get("name", ""),
                    "description": indicator.get("description", ""),
                    "filters": deepcopy(indicator.get("filters", [])),
                    "last_updated_on": last_updated_on,
                },
            ))
        if not points:
            raise ValueError(f"{record['product_name']} has no valid indicators to index")
        self._qdrant_client.upsert(
            collection_name=self._qdrant_collection_name,
            points=points,
            wait=True,
        )
        logger.info(
            "Qdrant product indexing complete: product=%s, vectors=%s, collection=%s",
            record["product_name"],
            len(points),
            self._qdrant_collection_name,
        )
        return {
            "success": True,
            "product_name": record["product_name"],
            "indicator_count": len(points),
            "qdrant_collection": self._qdrant_collection_name,
            "status": "updated",
        }

    def index_all_products(self) -> List[Dict[str, Any]]:
        product_names = self.list_datasets()
        if not product_names:
            raise RuntimeError("MongoDB product_metadata contains no published products to index")
        self._ensure_qdrant_collection()
        logger.info(
            "Full Qdrant sync started: products=%s, collection=%s",
            len(product_names),
            self._qdrant_collection_name,
        )
        results: List[Dict[str, Any]] = []
        for position, product_name in enumerate(product_names, start=1):
            logger.info("Full Qdrant sync progress: %s/%s product=%s", position, len(product_names), product_name)
            try:
                results.append(self.index_product(product_name))
            except Exception:
                logger.exception("Full Qdrant sync failed at product %s", product_name)
                raise
        logger.info(
            "Full Qdrant sync complete: products=%s, vectors=%s, collection=%s",
            len(results),
            sum(result["indicator_count"] for result in results),
            self._qdrant_collection_name,
        )
        return results

    def get_metadata_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        if not self.audit_log_file.exists():
            return []
        changes: List[Dict[str, Any]] = []
        with self.audit_log_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    changes.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed metadata audit-log entry")
        changes.reverse()
        return changes[:max(0, limit)]

    @staticmethod
    def _create_audit_entry(
        dataset_name: str,
        change_summary: str,
        changes_detected: List[str],
        indicators_changed: List[Dict[str, Any]],
        backup_file: str | None,
        user_ip: str | None,
        actor: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        timestamp = datetime.now(timezone(timedelta(hours=5, minutes=30)))
        return {
            "timestamp": timestamp.isoformat(),
            "dataset": dataset_name,
            "summary": change_summary,
            "changes": changes_detected,
            "indicators_modified": indicators_changed,
            "backup_file": Path(backup_file).name if backup_file else None,
            "user_ip": user_ip or "unknown",
            "updated_by": MetadataManager._actor_snapshot(actor),
        }

    def _append_audit_log(self, audit_entry: Dict[str, Any]) -> None:
        with self.audit_log_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(audit_entry, ensure_ascii=False) + "\n")
