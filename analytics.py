"""
Analytics Module for eSankhyiki Semantic Search
Handles query log analysis and provides daily/weekly/monthly statistics
"""
import os
import logging
from datetime import datetime, timedelta, timezone
from collections import defaultdict, Counter
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

try:
    from pymongo import MongoClient
    from pymongo.errors import PyMongoError
except ImportError:  # Allows existing JSONL analytics until pymongo is installed.
    MongoClient = None
    PyMongoError = Exception


load_dotenv()
logger = logging.getLogger(__name__)


class AnalyticsMongoUnavailableError(RuntimeError):
    """Raised when MongoDB cannot provide analytics data."""


class AnalyticsEngine:
    """Read and write interaction analytics directly in MongoDB."""
    
    def __init__(self, mongo_uri: Optional[str] = None,
                 mongo_database: Optional[str] = None, mongo_collection: Optional[str] = None,
                 product_collection: Optional[str] = None, environment: str = "dev",
                 use_default_env: bool = True):
        # MONGO_* is the simple project configuration. The older MONGODB_*
        # names remain as a compatibility fallback.
        self.mongo_uri = (
            mongo_uri or os.environ.get("MONGO_URI") or os.environ.get("MONGODB_URI")
        ) if use_default_env else mongo_uri
        self.mongo_database = mongo_database or os.environ.get("MONGO_DB_NAME") or os.environ.get("MONGODB_DATABASE", "semantic_search")
        self.mongo_collection_name = mongo_collection or os.environ.get("MONGO_COLLECTION") or os.environ.get("MONGODB_COLLECTION", "interactions")
        self.product_collection_name = product_collection or os.environ.get("MONGO_PRODUCT_COLLECTION", "product_metadata")
        self.environment = environment
        self._mongo_client = None
        self._mongo_collection = None
        self._mongo_product_collection = None
        self._mongo_unavailable = False

    def _get_mongo_collection(self):
        """Connect once, on demand, and make the data source visible in logs."""
        if self._mongo_collection is not None:
            return self._mongo_collection
        if self._mongo_unavailable:
            raise AnalyticsMongoUnavailableError("MongoDB analytics connection is unavailable")
        if not self.mongo_uri:
            logger.error("MongoDB analytics connection failed: MONGO_URI is not configured")
            raise AnalyticsMongoUnavailableError("MONGO_URI is not configured")
        if MongoClient is None:
            logger.error("MongoDB analytics connection failed: pymongo is not installed")
            raise AnalyticsMongoUnavailableError("pymongo is not installed")
        try:
            safe_target = self.mongo_uri.rsplit("@", 1)[-1]
            logger.info("MongoDB analytics connecting to %s", safe_target)
            self._mongo_client = MongoClient(
                self.mongo_uri,
                serverSelectionTimeoutMS=3000,
                connectTimeoutMS=3000,
                tz_aware=True,
            )
            self._mongo_client.admin.command("ping")
            self._mongo_collection = self._mongo_client[self.mongo_database][self.mongo_collection_name]
            document_count = self._mongo_collection.estimated_document_count()
            logger.info(
                "MongoDB analytics connected: %s.%s (%s interaction documents)",
                self.mongo_database, self.mongo_collection_name, document_count,
            )
            return self._mongo_collection
        except PyMongoError as exc:
            self._mongo_unavailable = True
            logger.error("MongoDB analytics connection failed: %s", exc)
            raise AnalyticsMongoUnavailableError("MongoDB analytics connection failed") from exc

    def _get_product_collection(self):
        """Return the published product collection from the same environment database."""
        if self._mongo_product_collection is not None:
            return self._mongo_product_collection
        self._get_mongo_collection()
        self._mongo_product_collection = self._mongo_client[self.mongo_database][self.product_collection_name]
        return self._mongo_product_collection

    @staticmethod
    def _serialise_datetime(value: Any) -> Optional[str]:
        if not isinstance(value, datetime):
            return str(value) if value is not None else None
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()

    def get_product_activity(self, start_date: Optional[datetime] = None,
                             end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """List products created or updated in the requested UTC period."""
        collection = self._get_product_collection()
        query: Dict[str, Any] = {}
        if start_date and end_date:
            start = start_date.replace(tzinfo=timezone.utc) if start_date.tzinfo is None else start_date
            end = end_date.replace(tzinfo=timezone.utc) if end_date.tzinfo is None else end_date
            query = {"$or": [
                {"created_at": {"$gte": start, "$lt": end}},
                {"updated_at": {"$gte": start, "$lt": end}},
            ]}
        try:
            documents = collection.find(query, {
                "product_key": 1, "product_name": 1, "created_at": 1,
                "updated_at": 1, "created_by": 1, "updated_by": 1,
            }).sort([("updated_at", -1), ("created_at", -1)])
            products = []
            for index, document in enumerate(documents, 1):
                created_at = document.get("created_at")
                updated_at = document.get("updated_at") or created_at
                created_dt = self._parse_timestamp(created_at)
                updated_dt = self._parse_timestamp(updated_at)
                status = "updated" if created_dt and updated_dt and updated_dt > created_dt else "added"
                products.append({
                    "serial_number": index,
                    "product": document.get("product_name") or document.get("product_key") or str(document.get("_id", "")),
                    "created_at": self._serialise_datetime(created_at),
                    "updated_at": self._serialise_datetime(updated_at),
                    "status": status,
                })
            return {"environment": self.environment, "total": len(products), "products": products}
        except PyMongoError as exc:
            logger.error("MongoDB product activity read failed for %s: %s", self.environment, exc)
            raise AnalyticsMongoUnavailableError("MongoDB product activity read failed") from exc

    @staticmethod
    def _normalise_mongo_log(document: Dict[str, Any]) -> Dict[str, Any]:
        """Make BSON Date records match the existing JSONL record shape."""
        log = dict(document)
        log.pop("_id", None)
        timestamp = log.get("timestamp")
        if isinstance(timestamp, datetime):
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            log["timestamp"] = timestamp.astimezone(timezone.utc).isoformat()
        return log

    def _read_logs(self) -> List[Dict]:
        """Read interaction records from MongoDB only."""
        collection = self._get_mongo_collection()
        try:
            logs = [self._normalise_mongo_log(document) for document in collection.find({})]
            logger.info("MongoDB analytics read complete: %s interaction documents", len(logs))
            return logs
        except PyMongoError as exc:
            self._mongo_unavailable = True
            logger.error("MongoDB analytics read failed: %s", exc)
            raise AnalyticsMongoUnavailableError("MongoDB analytics read failed") from exc

    def get_available_periods(self, year: int = 2026) -> List[Dict[str, Any]]:
        """Return log-backed months from February onward for one year."""
        collection = self._get_mongo_collection()
        try:
            pipeline = [
                {
                    "$set": {
                        "_analytics_timestamp": {
                            "$convert": {
                                "input": "$timestamp",
                                "to": "date",
                                "onError": None,
                                "onNull": None,
                            }
                        }
                    }
                },
                {
                    "$match": {
                        "_analytics_timestamp": {
                            "$gte": datetime(year, 2, 1),
                            "$lt": datetime(year + 1, 1, 1),
                        }
                    }
                },
                {
                    "$group": {
                        "_id": {
                            "year": {"$year": "$_analytics_timestamp"},
                            "month": {"$month": "$_analytics_timestamp"},
                        }
                    }
                },
                {"$sort": {"_id.year": -1, "_id.month": 1}},
            ]
            periods_by_year: Dict[int, List[int]] = defaultdict(list)
            for item in collection.aggregate(pipeline):
                period = item.get("_id", {})
                period_year = period.get("year")
                month = period.get("month")
                if isinstance(period_year, int) and isinstance(month, int):
                    periods_by_year[period_year].append(month)
            return [
                {"year": year, "months": sorted(set(months))}
                for year, months in sorted(periods_by_year.items(), reverse=True)
            ]
        except PyMongoError as exc:
            logger.error("MongoDB analytics period read failed for %s: %s", self.environment, exc)
            raise AnalyticsMongoUnavailableError("MongoDB analytics period read failed") from exc

    def save_interaction(self, interaction: Dict[str, Any]) -> None:
        """Save one completed user interaction to the configured Mongo collection."""
        collection = self._get_mongo_collection()
        try:
            collection.insert_one(interaction)
            logger.info("MongoDB interaction saved: %s", interaction.get("id", "unknown"))
        except PyMongoError as exc:
            logger.error("MongoDB interaction save failed: %s", exc)
            raise AnalyticsMongoUnavailableError("MongoDB interaction save failed") from exc
    
    def _parse_timestamp(self, timestamp_str: Any) -> Optional[datetime]:
        """Parse JSON or BSON timestamps as a naive UTC datetime for analytics."""
        try:
            if isinstance(timestamp_str, datetime):
                parsed = timestamp_str
            else:
                parsed = datetime.fromisoformat(str(timestamp_str).replace('Z', '+00:00'))
            if parsed.tzinfo is not None:
                return parsed.astimezone(timezone.utc).replace(tzinfo=None)
            return parsed
        except (TypeError, ValueError):
            return None
    
    def _filter_logs_by_date_range(self, logs: List[Dict], start_date: datetime, end_date: datetime) -> List[Dict]:
        """Filter logs within a specific date range"""
        filtered = []
        for log in logs:
            timestamp = self._parse_timestamp(log.get('timestamp', ''))
            if timestamp and start_date <= timestamp < end_date:
                filtered.append(log)
        return filtered

    def get_interactions_for_range(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Return raw Mongo interaction records in a half-open UTC date range."""
        return self._filter_logs_by_date_range(self._read_logs(), start_date, end_date)
    
    def get_daily_analytics(self, target_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Get analytics for a specific day
        Args:
            target_date: Date in YYYY-MM-DD format. If None, uses today.
        Returns:
            Dictionary with daily statistics
        """
        logs = self._read_logs()
        
        if target_date:
            try:
                date_obj = datetime.strptime(target_date, '%Y-%m-%d')
            except:
                date_obj = datetime.now()
        else:
            date_obj = datetime.now()
        
        # Filter logs for the target day
        start_of_day = date_obj.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = start_of_day + timedelta(days=1)
        
        daily_logs = self._filter_logs_by_date_range(logs, start_of_day, end_of_day)
        
        # Calculate statistics
        total_queries = len(daily_logs)
        
        # Extract datasets and queries
        datasets = []
        indicators = []
        queries = []
        hourly_distribution = defaultdict(int)
        
        for log in daily_logs:
            queries.append(log.get('raw_query', ''))
            timestamp = self._parse_timestamp(log.get('timestamp', ''))
            if timestamp:
                hourly_distribution[timestamp.hour] += 1
            
            # Extract dataset and indicator from response
            response = log.get('response', {})
            results = response.get('results', [])
            if results and len(results) > 0:
                for result in results:
                    dataset = str(result.get('dataset') or '').strip()
                    if dataset:
                        datasets.append(dataset)
                    indicator = str(result.get('indicator') or '').strip()
                    if indicator:
                        indicators.append(indicator)
        
        # Top datasets
        dataset_counter = Counter(datasets)
        top_datasets = [
            {"name": name, "count": count}
            for name, count in dataset_counter.most_common(10)
        ]
        
        # Top indicators
        indicator_counter = Counter(indicators)
        top_indicators = [
            {"name": name, "count": count}
            for name, count in indicator_counter.most_common(10)
        ]
        
        # Top queries
        query_counter = Counter(queries)
        top_queries = [
            {"query": query, "count": count}
            for query, count in query_counter.most_common(10)
        ]
        
        # Hourly distribution (convert to list format)
        hourly_data = [
            {"hour": hour, "count": hourly_distribution.get(hour, 0)}
            for hour in range(24)
        ]
        
        return {
            "date": date_obj.strftime('%Y-%m-%d'),
            "total_queries": total_queries,
            "top_datasets": top_datasets,
            "top_indicators": top_indicators,
            "top_queries": top_queries,
            "hourly_distribution": hourly_data
        }
    
    def get_weekly_analytics(self, start_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Get analytics for a week (Monday to Sunday)
        Args:
            start_date: Any date in YYYY-MM-DD format. The week containing this date (Monday-Sunday) will be used.
                       If None, uses current week.
        Returns:
            Dictionary with weekly statistics
        
        Examples:
            - start_date="2024-06-29" (Saturday) → week: Monday June 24 to Sunday June 30
            - start_date="2024-07-05" (Friday) → week: Monday July 1 to Sunday July 7
            - start_date="2024-06-22" (Saturday) → week: Monday June 17 to Sunday June 23
        """
        logs = self._read_logs()
        
        if start_date:
            try:
                date_obj = datetime.strptime(start_date, '%Y-%m-%d')
            except:
                date_obj = datetime.now()
        else:
            date_obj = datetime.now()
        
        # Find the Monday of the week containing date_obj
        # weekday(): Monday=0, Tuesday=1, ..., Sunday=6
        days_since_monday = date_obj.weekday()  # 0 for Monday, 6 for Sunday
        monday = date_obj - timedelta(days=days_since_monday)
        
        # Week runs from Monday 00:00:00 to Sunday 23:59:59
        start_obj = monday.replace(hour=0, minute=0, second=0, microsecond=0)
        end_obj = start_obj + timedelta(days=7)  # Next Monday 00:00:00
        
        weekly_logs = self._filter_logs_by_date_range(logs, start_obj, end_obj)
        
        total_queries = len(weekly_logs)
        
        # Daily breakdown
        daily_counts = defaultdict(int)
        datasets = []
        indicators = []
        queries = []
        
        for log in weekly_logs:
            queries.append(log.get('raw_query', ''))
            timestamp = self._parse_timestamp(log.get('timestamp', ''))
            if timestamp:
                date_key = timestamp.strftime('%Y-%m-%d')
                daily_counts[date_key] += 1
            
            # Extract dataset and indicator
            response = log.get('response', {})
            results = response.get('results', [])
            if results and len(results) > 0:
                for result in results:
                    dataset = str(result.get('dataset') or '').strip()
                    if dataset:
                        datasets.append(dataset)
                    indicator = str(result.get('indicator') or '').strip()
                    if indicator:
                        indicators.append(indicator)
        
        # Generate daily data for all 7 days (Monday to Sunday)
        daily_data = []
        current_date = start_obj
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        for i in range(7):
            date_key = current_date.strftime('%Y-%m-%d')
            daily_data.append({
                "date": date_key,
                "day_name": day_names[i],
                "count": daily_counts.get(date_key, 0)
            })
            current_date += timedelta(days=1)
        
        # Top datasets
        dataset_counter = Counter(datasets)
        top_datasets = [
            {"name": name, "count": count}
            for name, count in dataset_counter.most_common(10)
        ]
        
        # Top indicators
        indicator_counter = Counter(indicators)
        top_indicators = [
            {"name": name, "count": count}
            for name, count in indicator_counter.most_common(10)
        ]
        
        # Top queries
        query_counter = Counter(queries)
        top_queries = [
            {"query": query, "count": count}
            for query, count in query_counter.most_common(10)
        ]
        
        # Calculate Sunday (last day of week)
        sunday = start_obj + timedelta(days=6)
        
        return {
            "week_start": start_obj.strftime('%Y-%m-%d'),  # Monday
            "week_end": sunday.strftime('%Y-%m-%d'),  # Sunday
            "week_label": f"{start_obj.strftime('%b %d')} - {sunday.strftime('%b %d, %Y')}",
            "total_queries": total_queries,
            "daily_breakdown": daily_data,
            "top_datasets": top_datasets,
            "top_indicators": top_indicators,
            "top_queries": top_queries
        }
    
    def get_monthly_analytics(self, year: Optional[int] = None, month: Optional[int] = None) -> Dict[str, Any]:
        """
        Get analytics for a specific month
        Args:
            year: Year (e.g., 2026). If None, uses current year.
            month: Month (1-12). If None, uses current month.
        Returns:
            Dictionary with monthly statistics
        """
        logs = self._read_logs()
        
        now = datetime.now()
        target_year = year if year else now.year
        target_month = month if month else now.month
        
        # Calculate start and end of month
        start_of_month = datetime(target_year, target_month, 1)
        if target_month == 12:
            end_of_month = datetime(target_year + 1, 1, 1)
        else:
            end_of_month = datetime(target_year, target_month + 1, 1)
        
        monthly_logs = self._filter_logs_by_date_range(logs, start_of_month, end_of_month)
        
        total_queries = len(monthly_logs)
        
        # Daily breakdown for the month
        daily_counts = defaultdict(int)
        datasets = []
        indicators = []
        queries = []
        
        for log in monthly_logs:
            queries.append(log.get('raw_query', ''))
            timestamp = self._parse_timestamp(log.get('timestamp', ''))
            if timestamp:
                day = timestamp.day
                daily_counts[day] += 1
            
            # Extract dataset and indicator
            response = log.get('response', {})
            results = response.get('results', [])
            if results and len(results) > 0:
                for result in results:
                    dataset = str(result.get('dataset') or '').strip()
                    if dataset:
                        datasets.append(dataset)
                    indicator = str(result.get('indicator') or '').strip()
                    if indicator:
                        indicators.append(indicator)
        
        # Generate daily data for all days in month
        from calendar import monthrange
        days_in_month = monthrange(target_year, target_month)[1]
        
        daily_data = []
        for day in range(1, days_in_month + 1):
            daily_data.append({
                "day": day,
                "count": daily_counts.get(day, 0)
            })
        
        # Top datasets
        dataset_counter = Counter(datasets)
        top_datasets = [
            {"name": name, "count": count}
            for name, count in dataset_counter.most_common(10)
        ]
        
        # Top indicators
        indicator_counter = Counter(indicators)
        top_indicators = [
            {"name": name, "count": count}
            for name, count in indicator_counter.most_common(10)
        ]
        
        # Top queries
        query_counter = Counter(queries)
        top_queries = [
            {"query": query, "count": count}
            for query, count in query_counter.most_common(10)
        ]
        
        return {
            "year": target_year,
            "month": target_month,
            "month_name": start_of_month.strftime('%B'),
            "total_queries": total_queries,
            "daily_breakdown": daily_data,
            "top_datasets": top_datasets,
            "top_indicators": top_indicators,
            "top_queries": top_queries,
            "avg_queries_per_day": round(total_queries / days_in_month, 2) if days_in_month > 0 else 0
        }
    
    def get_custom_range_analytics(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Get analytics for a custom date range
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
        Returns:
            Dictionary with range statistics
        """
        logs = self._read_logs()
        
        try:
            start_obj = datetime.strptime(start_date, '%Y-%m-%d')
            end_obj = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)  # Include end date
        except:
            return {"error": "Invalid date format. Use YYYY-MM-DD"}
        
        range_logs = self._filter_logs_by_date_range(logs, start_obj, end_obj)
        
        total_queries = len(range_logs)
        
        datasets = []
        indicators = []
        queries = []
        daily_counts = defaultdict(int)
        
        for log in range_logs:
            queries.append(log.get('raw_query', ''))
            timestamp = self._parse_timestamp(log.get('timestamp', ''))
            if timestamp:
                date_key = timestamp.strftime('%Y-%m-%d')
                daily_counts[date_key] += 1
            
            # Extract dataset and indicator
            response = log.get('response', {})
            results = response.get('results', [])
            if results and len(results) > 0:
                for result in results:
                    dataset = str(result.get('dataset') or '').strip()
                    if dataset:
                        datasets.append(dataset)
                    indicator = str(result.get('indicator') or '').strip()
                    if indicator:
                        indicators.append(indicator)
        
        # Daily data
        daily_data = []
        current_date = start_obj
        while current_date < end_obj:
            date_key = current_date.strftime('%Y-%m-%d')
            daily_data.append({
                "date": date_key,
                "count": daily_counts.get(date_key, 0)
            })
            current_date += timedelta(days=1)
        
        # Top datasets
        dataset_counter = Counter(datasets)
        top_datasets = [
            {"name": name, "count": count}
            for name, count in dataset_counter.most_common(10)
        ]
        
        # Top indicators
        indicator_counter = Counter(indicators)
        top_indicators = [
            {"name": name, "count": count}
            for name, count in indicator_counter.most_common(10)
        ]
        
        # Top queries
        query_counter = Counter(queries)
        top_queries = [
            {"query": query, "count": count}
            for query, count in query_counter.most_common(10)
        ]
        
        total_days = (end_obj - start_obj).days
        
        return {
            "start_date": start_date,
            "end_date": end_date,
            "total_days": total_days,
            "total_queries": total_queries,
            "daily_breakdown": daily_data,
            "top_datasets": top_datasets,
            "top_indicators": top_indicators,
            "top_queries": top_queries,
            "avg_queries_per_day": round(total_queries / total_days, 2) if total_days > 0 else 0
        }
