import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_list(name: str, default: List[str]) -> List[str]:
    raw = os.environ.get(name)
    if not raw:
        return list(default)
    return [item.strip() for item in raw.split(",") if item.strip()]


def _env_detect_every_n_frames(name: str, default: str) -> Tuple[int, int]:
    raw = (os.environ.get(name, default) or default).strip()
    parts = [item.strip() for item in raw.split(",") if item.strip()]
    if not parts:
        parts = [default]
    if len(parts) == 1:
        value = max(1, int(parts[0]))
        return value, value
    if len(parts) == 2:
        first = max(1, int(parts[0]))
        second = max(1, int(parts[1]))
        return min(first, second), max(first, second)
    raise ValueError(
        f"{name} must be a single integer like '7' or a range like '3,7'."
    )


@dataclass
class BackendConfig:
    stream_url: str
    camera_id: str
    stream_name: str
    report_store: str
    model_path: Path
    routes_path: Path
    zone_config_path: Path
    reports_jsonl_path: Path
    snapshots_dir: Path
    firebase_credentials_path: Optional[str]
    firebase_project_id: Optional[str]
    firestore_collection: str
    confidence: float
    yolo_imgsz: int
    detect_every_n_frames: int
    detect_every_n_frames_min: int
    detect_every_n_frames_max: int
    dynamic_detect_every_n_frames: bool
    stable_hits_required: int
    report_cooldown_seconds: float
    top_route_predictions: int
    schedule_sigma_minutes: float
    schedule_early_tolerance_minutes: float
    schedule_late_tolerance_minutes: float
    service_day_override: Optional[str]
    stream_quality_preference: List[str]
    stream_live_edge: int
    stream_segment_threads: int
    chunk_duration_seconds: int
    raw_chunk_dir: Path
    annotated_dir: Path
    detection_json_dir: Path
    failed_chunk_dir: Path
    ffmpeg_binary: str
    ffprobe_binary: str
    clean_startup_chunks: bool
    delete_processed_chunks: bool
    keep_failed_chunks: bool
    save_annotated_video: bool
    show_debug: bool
    processor_poll_interval_seconds: float
    backlog_warning_chunks: int
    api_host: str
    api_port: int


def load_backend_config() -> BackendConfig:
    detect_every_n_frames_min, detect_every_n_frames_max = _env_detect_every_n_frames(
        "JUTC_DETECT_EVERY_N_FRAMES",
        "5,10",
    )
    return BackendConfig(
        stream_url=os.environ.get("JUTC_STREAM_URL", "").strip(),
        camera_id=os.environ.get("JUTC_CAMERA_ID", "camera_washington_blvd_1").strip(),
        stream_name=os.environ.get("JUTC_STREAM_NAME", "JUTC Live Feed").strip(),
        report_store=os.environ.get("JUTC_REPORT_STORE", "jsonl").strip().lower(),
        model_path=Path(os.environ.get("JUTC_MODEL_PATH", str(SCRIPT_DIR / "best.pt"))),
        routes_path=Path(os.environ.get("JUTC_ROUTES_PATH", str(PROJECT_ROOT / "busRoutes.json"))),
        zone_config_path=Path(
            os.environ.get("JUTC_ZONE_CONFIG_PATH", str(SCRIPT_DIR / "test2_zone_config.json"))
        ),
        reports_jsonl_path=Path(
            os.environ.get("JUTC_REPORTS_JSONL_PATH", str(SCRIPT_DIR / "ai_reports.jsonl"))
        ),
        snapshots_dir=Path(
            os.environ.get("JUTC_SNAPSHOTS_DIR", str(SCRIPT_DIR / "ai_report_snapshots"))
        ),
        firebase_credentials_path=os.environ.get("JUTC_FIREBASE_CREDENTIALS_PATH"),
        firebase_project_id=os.environ.get("JUTC_FIREBASE_PROJECT_ID"),
        firestore_collection=os.environ.get("JUTC_FIRESTORE_COLLECTION", "ai_reports").strip(),
        confidence=float(os.environ.get("JUTC_DETECT_CONFIDENCE", "0.60")),
        yolo_imgsz=int(os.environ.get("JUTC_YOLO_IMGSZ", "960")),
        detect_every_n_frames=detect_every_n_frames_min,
        detect_every_n_frames_min=detect_every_n_frames_min,
        detect_every_n_frames_max=detect_every_n_frames_max,
        dynamic_detect_every_n_frames=(detect_every_n_frames_min != detect_every_n_frames_max),
        stable_hits_required=max(1, int(os.environ.get("JUTC_STABLE_HITS_REQUIRED", "3"))),
        report_cooldown_seconds=float(os.environ.get("JUTC_REPORT_COOLDOWN_SECONDS", "180")),
        top_route_predictions=max(1, int(os.environ.get("JUTC_TOP_ROUTE_PREDICTIONS", "3"))),
        schedule_sigma_minutes=float(os.environ.get("JUTC_SCHEDULE_SIGMA_MINUTES", "12.0")),
        schedule_early_tolerance_minutes=float(
            os.environ.get("JUTC_SCHEDULE_EARLY_TOLERANCE_MINUTES", "4.0")
        ),
        schedule_late_tolerance_minutes=float(
            os.environ.get("JUTC_SCHEDULE_LATE_TOLERANCE_MINUTES", "8.0")
        ),
        service_day_override=os.environ.get("JUTC_SERVICE_DAY_OVERRIDE") or None,
        stream_quality_preference=_env_list(
            "JUTC_STREAM_QUALITY_PREFERENCE",
            ["best", "1080p", "720p", "480p"],
        ),
        stream_live_edge=max(1, int(os.environ.get("JUTC_STREAM_LIVE_EDGE", "10"))),
        stream_segment_threads=max(1, int(os.environ.get("JUTC_STREAM_SEGMENT_THREADS", "3"))),
        chunk_duration_seconds=max(5, int(os.environ.get("JUTC_CHUNK_DURATION", "10"))),
        raw_chunk_dir=Path(os.environ.get("JUTC_RAW_CHUNK_DIR", str(SCRIPT_DIR / "raw_chunks"))),
        annotated_dir=Path(
            os.environ.get("JUTC_ANNOTATED_DIR", str(SCRIPT_DIR / "annotated_chunks"))
        ),
        detection_json_dir=Path(
            os.environ.get("JUTC_DETECTION_JSON_DIR", str(SCRIPT_DIR / "detections"))
        ),
        failed_chunk_dir=Path(
            os.environ.get("JUTC_FAILED_CHUNK_DIR", str(SCRIPT_DIR / "failed_chunks"))
        ),
        ffmpeg_binary=os.environ.get("JUTC_FFMPEG_BINARY", "ffmpeg").strip() or "ffmpeg",
        ffprobe_binary=os.environ.get("JUTC_FFPROBE_BINARY", "ffprobe").strip() or "ffprobe",
        clean_startup_chunks=_env_bool("JUTC_CLEAN_STARTUP_CHUNKS", True),
        delete_processed_chunks=_env_bool("JUTC_DELETE_PROCESSED_CHUNKS", True),
        keep_failed_chunks=_env_bool("JUTC_KEEP_FAILED_CHUNKS", True),
        save_annotated_video=_env_bool("JUTC_SAVE_ANNOTATED_VIDEO", True),
        show_debug=_env_bool(
            "JUTC_SHOW_DEBUG",
            _env_bool("JUTC_DEBUG_WINDOW", False),
        ),
        processor_poll_interval_seconds=float(
            os.environ.get("JUTC_PROCESSOR_POLL_INTERVAL_SECONDS", "1.0")
        ),
        backlog_warning_chunks=max(1, int(os.environ.get("JUTC_BACKLOG_WARNING_CHUNKS", "4"))),
        api_host=os.environ.get("JUTC_API_HOST", "127.0.0.1").strip(),
        api_port=int(os.environ.get("JUTC_API_PORT", "8080")),
    )
