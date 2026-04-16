import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import BackendConfig

try:
    import firebase_admin
    from firebase_admin import credentials, firestore
except ImportError:
    firebase_admin = None
    credentials = None
    firestore = None


class JsonlReportStore:
    store_type = "jsonl"

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write_report(self, report: Dict[str, Any]) -> None:
        with self._lock:
            with self.path.open("a", encoding="utf-8") as output_file:
                output_file.write(json.dumps(report, ensure_ascii=True) + "\n")

    def _read_all(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []

        reports: List[Dict[str, Any]] = []
        with self.path.open(encoding="utf-8") as input_file:
            for line in input_file:
                line = line.strip()
                if not line:
                    continue
                try:
                    reports.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return reports

    def list_recent(
        self,
        limit: int = 20,
        route: Optional[str] = None,
        zone: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        reports = self._read_all()
        if route:
            route = route.upper()
            reports = [report for report in reports if (report.get("likely_route") or "").upper() == route]
        if zone:
            zone = zone.upper()
            reports = [report for report in reports if (report.get("zone_name") or "").upper() == zone]
        reports.sort(key=lambda item: item.get("detected_at", ""), reverse=True)
        return reports[:limit]

    def get_report(self, report_id: str) -> Optional[Dict[str, Any]]:
        for report in reversed(self._read_all()):
            if report.get("id") == report_id:
                return report
        return None


class FirestoreReportStore:
    store_type = "firestore"

    def __init__(
        self,
        collection_name: str,
        credentials_path: Optional[str] = None,
        project_id: Optional[str] = None,
    ):
        if firebase_admin is None or credentials is None or firestore is None:
            raise RuntimeError(
                "firebase-admin is not installed. Install it with "
                "'python -m pip install firebase-admin'."
            )

        if not firebase_admin._apps:
            if credentials_path:
                credential = credentials.Certificate(credentials_path)
                init_options = {"projectId": project_id} if project_id else None
                firebase_admin.initialize_app(credential, init_options)
            else:
                firebase_admin.initialize_app()

        self.collection = firestore.client().collection(collection_name)

    @staticmethod
    def _coerce_firestore_timestamp(value: Any):
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value

        text = str(value or "").strip()
        if not text:
            return firestore.SERVER_TIMESTAMP
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"

        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return firestore.SERVER_TIMESTAMP

        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed

    @staticmethod
    def _json_friendly(value: Any) -> Any:
        if isinstance(value, datetime):
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone.utc)
            return value.isoformat()
        if isinstance(value, list):
            return [FirestoreReportStore._json_friendly(item) for item in value]
        if isinstance(value, dict):
            return {
                key: FirestoreReportStore._json_friendly(item)
                for key, item in value.items()
            }
        return value

    def _prepare_firestore_report(self, report: Dict[str, Any]) -> Dict[str, Any]:
        allowed_fields = (
            "id",
            "source",
            "camera_id",
            "stream_name",
            "track_id",
            "zone_name",
            "bus_confidence",
            "likely_route",
            "route_confidence",
            "direction",
            "destination",
            "matched_stop",
            "schedule_instance_id",
            "trip_start_time",
            "trip_end_time",
            "model_version",
            "status",
            "predictions",
        )
        document = {
            field_name: report[field_name]
            for field_name in allowed_fields
            if field_name in report
        }
        document["detected_at"] = self._coerce_firestore_timestamp(
            report.get("detected_at")
        )
        return document

    def write_report(self, report: Dict[str, Any]) -> None:
        document = self._prepare_firestore_report(report)
        self.collection.document(str(report["id"])).set(document)

    def list_recent(
        self,
        limit: int = 20,
        route: Optional[str] = None,
        zone: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        query = self.collection.order_by(
            "detected_at",
            direction=firestore.Query.DESCENDING,
        ).limit(max(limit * 5, 50))

        reports = [
            self._json_friendly(document.to_dict() or {})
            for document in query.stream()
        ]
        if route:
            route = route.upper()
            reports = [report for report in reports if (report.get("likely_route") or "").upper() == route]
        if zone:
            zone = zone.upper()
            reports = [report for report in reports if (report.get("zone_name") or "").upper() == zone]
        return reports[:limit]

    def get_report(self, report_id: str) -> Optional[Dict[str, Any]]:
        document = self.collection.document(report_id).get()
        if not document.exists:
            return None
        return self._json_friendly(document.to_dict() or {})


def build_report_store(config: BackendConfig):
    if config.report_store == "firestore":
        return FirestoreReportStore(
            collection_name=config.firestore_collection,
            credentials_path=config.firebase_credentials_path,
            project_id=config.firebase_project_id,
        )
    return JsonlReportStore(config.reports_jsonl_path)
