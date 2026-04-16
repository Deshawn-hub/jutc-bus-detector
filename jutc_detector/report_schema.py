from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_report_id() -> str:
    return f"ai_{uuid4().hex}"


@dataclass
class RoutePredictionSummary:
    route_number: str
    probability: int
    direction: str
    destination: str
    matched_stop: str
    scheduled_time: str
    delta_label: str
    range_name: str
    schedule_instance_id: str = ""
    trip_start_time: str = ""
    trip_end_time: str = ""

    @classmethod
    def from_prediction(cls, prediction: Dict[str, Any]) -> "RoutePredictionSummary":
        return cls(
            route_number=str(prediction.get("route_number") or ""),
            probability=int(prediction.get("probability") or 0),
            direction=str(prediction.get("direction") or ""),
            destination=str(prediction.get("destination") or ""),
            matched_stop=str(prediction.get("matched_stop") or ""),
            scheduled_time=str(prediction.get("scheduled_time") or ""),
            delta_label=str(prediction.get("delta_label") or ""),
            range_name=str(prediction.get("range_name") or ""),
            schedule_instance_id=str(prediction.get("schedule_instance_id") or ""),
            trip_start_time=str(prediction.get("trip_start_time") or ""),
            trip_end_time=str(prediction.get("trip_end_time") or ""),
        )


@dataclass
class AIReport:
    id: str
    source: str
    camera_id: str
    stream_name: str
    detected_at: str
    track_id: int
    zone_name: str
    bus_confidence: float
    likely_route: Optional[str] = None
    route_confidence: Optional[int] = None
    direction: Optional[str] = None
    destination: Optional[str] = None
    matched_stop: Optional[str] = None
    schedule_instance_id: Optional[str] = None
    trip_start_time: Optional[str] = None
    trip_end_time: Optional[str] = None
    model_version: Optional[str] = None
    snapshot_path: Optional[str] = None
    status: str = "active"
    predictions: List[RoutePredictionSummary] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
