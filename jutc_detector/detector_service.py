import argparse
import json
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import streamlink
from ultralytics import YOLO

from .config import BackendConfig, load_backend_config
from .report_schema import AIReport, RoutePredictionSummary, new_report_id, utc_now_iso
from .report_store import build_report_store
from .route_schedule_matcher import (
    RouteScheduleMatcher,
    normalize_route_number,
    normalize_stop_name,
    normalized_text_matches,
)


Point = Tuple[int, int]
FRAME_SIZE = (1280, 720)
WINDOW_NAME = "JUTC Delayed Detector Debug"
ZONE_LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
RECORDING_STATE_FILENAME = "recording_state.json"
CHUNK_NAME_RE = re.compile(r"chunk_(\d+)")
FFMPEG_OPEN_RE = re.compile(r"Opening '([^']*chunk_(\d+)\.mp4)' for writing")


@dataclass
class ZoneDefinition:
    name: str
    polygon: List[Point]
    color: Tuple[int, int, int]
    route_targets: List[str]
    candidate_routes: List[Dict[str, str]]
    schedule_stop_keywords: List[str]


@dataclass
class TrackState:
    last_seen_frame: int = 0
    max_bus_confidence: float = 0.0
    zone_hits: Dict[str, int] = field(default_factory=dict)
    emitted_zones: set = field(default_factory=set)


@dataclass(frozen=True)
class ChunkPaths:
    index: int
    chunk_path: Path
    ready_marker_path: Path
    annotated_path: Path
    detection_json_path: Path
    ready_metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def chunk_name(self) -> str:
        return self.chunk_path.name


@dataclass
class VideoTimingInfo:
    source_fps: float
    output_fps: float
    total_frames_reported: int
    source_duration_seconds: float
    fps_source: str


def write_json_atomic(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = Path(f"{path}.tmp")
    with temp_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=True, indent=2)
    temp_path.replace(path)


def parse_ratio(value) -> Optional[float]:
    if value in (None, "", "0/0"):
        return None
    text = str(value).strip()
    if not text:
        return None
    if "/" in text:
        numerator_text, denominator_text = text.split("/", 1)
        try:
            numerator = float(numerator_text)
            denominator = float(denominator_text)
        except ValueError:
            return None
        if denominator == 0:
            return None
        return numerator / denominator
    try:
        return float(text)
    except ValueError:
        return None


def is_reasonable_fps(value: Optional[float]) -> bool:
    return value is not None and 1.0 <= value <= 240.0


def probe_video_timing(
    video_path: Path,
    capture,
    ffprobe_binary: str,
) -> VideoTimingInfo:
    cv2_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    cv2_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cv2_duration_seconds = 0.0
    if cv2_frame_count > 0 and cv2_fps > 0:
        cv2_duration_seconds = cv2_frame_count / cv2_fps

    source_fps = cv2_fps if is_reasonable_fps(cv2_fps) else 0.0
    fps_source = "opencv"
    total_frames_reported = cv2_frame_count
    source_duration_seconds = cv2_duration_seconds

    try:
        probe = subprocess.run(
            [
                ffprobe_binary,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=avg_frame_rate,r_frame_rate,nb_frames,duration",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                str(video_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(probe.stdout or "{}")
        stream = (payload.get("streams") or [{}])[0]
        format_info = payload.get("format") or {}

        ffprobe_avg_fps = parse_ratio(stream.get("avg_frame_rate"))
        ffprobe_real_fps = parse_ratio(stream.get("r_frame_rate"))
        ffprobe_frame_count = int(float(stream.get("nb_frames") or 0) or 0)
        ffprobe_duration = float(
            stream.get("duration")
            or format_info.get("duration")
            or 0.0
        )

        if is_reasonable_fps(ffprobe_avg_fps):
            source_fps = float(ffprobe_avg_fps)
            fps_source = "ffprobe_avg_frame_rate"
        elif is_reasonable_fps(ffprobe_real_fps):
            source_fps = float(ffprobe_real_fps)
            fps_source = "ffprobe_r_frame_rate"
        elif ffprobe_duration > 0 and ffprobe_frame_count > 0:
            derived_fps = ffprobe_frame_count / ffprobe_duration
            if is_reasonable_fps(derived_fps):
                source_fps = float(derived_fps)
                fps_source = "ffprobe_frame_count/duration"

        if ffprobe_frame_count > 0:
            total_frames_reported = ffprobe_frame_count
        if ffprobe_duration > 0:
            source_duration_seconds = ffprobe_duration
    except Exception:
        pass

    if not is_reasonable_fps(source_fps):
        source_fps = 30.0
        fps_source = "fallback_30fps"

    return VideoTimingInfo(
        source_fps=source_fps,
        output_fps=source_fps,
        total_frames_reported=total_frames_reported,
        source_duration_seconds=source_duration_seconds,
        fps_source=fps_source,
    )


def select_stream(streams, quality_preference: Sequence[str]):
    for quality_name in quality_preference:
        if quality_name in streams:
            return quality_name, streams[quality_name]
    for stream_name, stream in streams.items():
        if "audio" not in stream_name.lower():
            return stream_name, stream
    return next(iter(streams.items()))


def point_in_polygon(point: Point, polygon_points: Sequence[Point]) -> bool:
    if len(polygon_points) < 3:
        return False
    pts = np.array(polygon_points, dtype=np.int32)
    return cv2.pointPolygonTest(pts, (float(point[0]), float(point[1])), False) >= 0


def box_center(x1: int, y1: int, x2: int, y2: int) -> Point:
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def scale_point(point: Point, scale_x: float, scale_y: float) -> Point:
    return int(point[0] * scale_x), int(point[1] * scale_y)


def scale_box(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    scale_x: float,
    scale_y: float,
) -> Tuple[int, int, int, int]:
    return (
        int(x1 * scale_x),
        int(y1 * scale_y),
        int(x2 * scale_x),
        int(y2 * scale_y),
    )


def format_seconds(value: float) -> str:
    minutes = int(value // 60)
    seconds = value % 60
    return f"{minutes:02d}:{seconds:05.2f}"


def lookup_class_name(names, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def draw_polygon_zone(img, zone: ZoneDefinition) -> None:
    pts = np.array(zone.polygon, dtype=np.int32)
    if len(pts) >= 3:
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], zone.color)
        cv2.addWeighted(overlay, 0.15, img, 0.85, 0, img)
        cv2.polylines(img, [pts], isClosed=True, color=zone.color, thickness=2)
    if zone.polygon:
        cx = sum(point[0] for point in zone.polygon) // len(zone.polygon)
        cy = sum(point[1] for point in zone.polygon) // len(zone.polygon)
        cv2.putText(img, zone.name, (cx - 20, cy), ZONE_LABEL_FONT, 0.55, zone.color, 2)


def draw_detection_box(
    img,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    label: str,
    color: Tuple[int, int, int],
) -> None:
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, label, (x1, max(24, y1 - 10)), ZONE_LABEL_FONT, 0.55, color, 2)


def draw_debug_hud(
    img,
    chunk_name: str,
    frame_index: int,
    frame_time_seconds: float,
    bus_count: int,
    backlog_chunks: int,
    delay_seconds: float,
    reports_emitted: int,
    status: str,
) -> None:
    lines = [
        f"Chunk: {chunk_name}",
        f"Frame: {frame_index}",
        f"Chunk time: {format_seconds(frame_time_seconds)}",
        f"Buses in frame: {bus_count}",
        f"Backlog: {backlog_chunks} ready chunks",
        f"Approx live delay: {delay_seconds:.1f}s",
        f"Reports emitted: {reports_emitted}",
        f"Status: {status}",
        "Press Q to stop",
    ]
    for index, line in enumerate(lines):
        cv2.putText(img, line, (20, 30 + (index * 28)), ZONE_LABEL_FONT, 0.7, (255, 255, 0), 2)


def make_status_frame(message: str, detail: str) -> np.ndarray:
    blank = np.zeros((FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=np.uint8)
    cv2.putText(blank, message, (60, 320), ZONE_LABEL_FONT, 1.0, (0, 200, 255), 2)
    cv2.putText(blank, detail, (60, 370), ZONE_LABEL_FONT, 0.7, (255, 255, 255), 2)
    cv2.putText(blank, "Press Q to stop", (60, 430), ZONE_LABEL_FONT, 0.7, (255, 255, 0), 2)
    return blank


class ChunkRepository:
    def __init__(self, config: BackendConfig):
        self.config = config
        self.raw_chunk_dir = config.raw_chunk_dir
        self.annotated_dir = config.annotated_dir
        self.detection_json_dir = config.detection_json_dir
        self.failed_chunk_dir = config.failed_chunk_dir
        self.recording_state_path = self.raw_chunk_dir / RECORDING_STATE_FILENAME
        self.ensure_directories()

    def ensure_directories(self) -> None:
        self.raw_chunk_dir.mkdir(parents=True, exist_ok=True)
        self.annotated_dir.mkdir(parents=True, exist_ok=True)
        self.detection_json_dir.mkdir(parents=True, exist_ok=True)
        self.failed_chunk_dir.mkdir(parents=True, exist_ok=True)

    def chunk_name(self, index: int) -> str:
        return f"chunk_{index:04d}.mp4"

    def raw_chunk_path(self, index: int) -> Path:
        return self.raw_chunk_dir / self.chunk_name(index)

    def ready_marker_path(self, index: int) -> Path:
        return self.raw_chunk_dir / f"chunk_{index:04d}.ready.json"

    def annotated_chunk_path(self, index: int) -> Path:
        return self.annotated_dir / f"chunk_{index:04d}_annotated.mp4"

    def detection_json_path(self, index: int) -> Path:
        return self.detection_json_dir / f"chunk_{index:04d}.json"

    def build_chunk(self, index: int, ready_metadata: Optional[Dict[str, object]] = None) -> ChunkPaths:
        return ChunkPaths(
            index=index,
            chunk_path=self.raw_chunk_path(index),
            ready_marker_path=self.ready_marker_path(index),
            annotated_path=self.annotated_chunk_path(index),
            detection_json_path=self.detection_json_path(index),
            ready_metadata=ready_metadata or {},
        )

    def parse_index(self, path_or_name) -> Optional[int]:
        match = CHUNK_NAME_RE.search(str(path_or_name))
        if not match:
            return None
        return int(match.group(1))

    def next_recording_index(self) -> int:
        known_indices = [chunk.index for chunk in self.list_ready_chunks()]
        recording_state = self.read_recording_state()
        for key in ("current_chunk_index", "latest_ready_index"):
            value = recording_state.get(key)
            if value is None:
                continue
            try:
                known_indices.append(int(value))
            except (TypeError, ValueError):
                continue

        if not known_indices:
            return 1
        return max(known_indices) + 1

    def mark_chunk_ready(self, index: int, metadata: Optional[Dict[str, object]] = None) -> Optional[ChunkPaths]:
        chunk = self.build_chunk(index)
        if not chunk.chunk_path.exists() or chunk.chunk_path.stat().st_size <= 0:
            return None
        if chunk.ready_marker_path.exists():
            return chunk

        ready_payload = {
            "chunk_index": index,
            "chunk_name": chunk.chunk_name,
            "chunk_path": str(chunk.chunk_path),
            "ready_at": utc_now_iso(),
            "chunk_duration_seconds": self.config.chunk_duration_seconds,
        }
        if metadata:
            ready_payload.update(metadata)
        write_json_atomic(chunk.ready_marker_path, ready_payload)
        return self.build_chunk(index, ready_payload)

    def list_ready_chunks(self) -> List[ChunkPaths]:
        chunks: List[ChunkPaths] = []
        for ready_path in sorted(self.raw_chunk_dir.glob("chunk_*.ready.json")):
            index = self.parse_index(ready_path.name)
            if index is None:
                continue
            ready_metadata: Dict[str, object] = {}
            try:
                ready_metadata = json.loads(ready_path.read_text(encoding="utf-8"))
            except Exception:
                ready_metadata = {}
            chunks.append(self.build_chunk(index, ready_metadata))
        chunks.sort(key=lambda chunk: chunk.index)
        return chunks

    def list_pending_chunks(self) -> List[ChunkPaths]:
        return [
            chunk
            for chunk in self.list_ready_chunks()
            if (
                not chunk.detection_json_path.exists()
                or chunk.detection_json_path.stat().st_mtime < chunk.ready_marker_path.stat().st_mtime
            )
        ]

    def read_recording_state(self) -> Dict[str, object]:
        if not self.recording_state_path.exists():
            return {}
        try:
            return json.loads(self.recording_state_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def write_recording_state(self, payload: Dict[str, object]) -> None:
        write_json_atomic(self.recording_state_path, payload)

    @staticmethod
    def _delete_file_if_exists(path: Path) -> bool:
        try:
            path.unlink()
            return True
        except FileNotFoundError:
            return False

    def _move_file_if_exists(self, path: Path, destination_dir: Path) -> Optional[Path]:
        if not path.exists():
            return None
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination_path = destination_dir / path.name
        self._delete_file_if_exists(destination_path)
        path.replace(destination_path)
        return destination_path

    def cleanup_startup_runtime_files(self) -> int:
        # Startup cleanup removes stale generated chunk artifacts from prior runs
        # before ffmpeg starts writing fresh chunks.
        removed_count = 0
        cleanup_specs = (
            (self.raw_chunk_dir, ("chunk_*.mp4", "chunk_*.ready.json", "chunk_*.tmp")),
            (
                self.annotated_dir,
                ("chunk_*_annotated.mp4", "chunk_*.tmp"),
            ),
            (self.detection_json_dir, ("chunk_*.json", "chunk_*.tmp")),
        )
        for directory, patterns in cleanup_specs:
            for pattern in patterns:
                for path in directory.glob(pattern):
                    if path.is_file() and self._delete_file_if_exists(path):
                        removed_count += 1

        if self._delete_file_if_exists(self.recording_state_path):
            removed_count += 1
        if self._delete_file_if_exists(Path(f"{self.recording_state_path}.tmp")):
            removed_count += 1
        return removed_count

    def cleanup_processed_chunk(self, chunk: ChunkPaths) -> int:
        if not self.config.delete_processed_chunks:
            return 0

        # Once a chunk has been fully processed and reports are persisted, remove
        # the per-chunk working files so the server does not accumulate backlog.
        removed_count = 0
        for path in (
            chunk.chunk_path,
            chunk.ready_marker_path,
            chunk.annotated_path,
            chunk.detection_json_path,
            Path(f"{chunk.chunk_path}.tmp"),
            Path(f"{chunk.ready_marker_path}.tmp"),
            Path(f"{chunk.annotated_path}.tmp"),
            Path(f"{chunk.detection_json_path}.tmp"),
        ):
            if self._delete_file_if_exists(path):
                removed_count += 1
        return removed_count

    def mark_chunk_failed(self, chunk: ChunkPaths, reason: str) -> None:
        failure_payload = {
            "chunk_index": chunk.index,
            "chunk_name": chunk.chunk_name,
            "failed_at": utc_now_iso(),
            "reason": str(reason),
            "kept_for_debug": self.config.keep_failed_chunks,
        }

        if self.config.keep_failed_chunks:
            moved_paths = {}
            for label, path in (
                ("chunk_path", chunk.chunk_path),
                ("ready_marker_path", chunk.ready_marker_path),
                ("annotated_path", chunk.annotated_path),
                ("detection_json_path", chunk.detection_json_path),
                ("chunk_tmp_path", Path(f"{chunk.chunk_path}.tmp")),
                ("ready_marker_tmp_path", Path(f"{chunk.ready_marker_path}.tmp")),
                ("annotated_tmp_path", Path(f"{chunk.annotated_path}.tmp")),
                ("detection_json_tmp_path", Path(f"{chunk.detection_json_path}.tmp")),
            ):
                moved_path = self._move_file_if_exists(path, self.failed_chunk_dir)
                if moved_path is not None:
                    moved_paths[label] = str(moved_path)
            failure_payload["artifacts"] = moved_paths
            failure_marker_path = self.failed_chunk_dir / f"chunk_{chunk.index:04d}.failed.json"
        else:
            self._delete_file_if_exists(chunk.ready_marker_path)
            self._delete_file_if_exists(Path(f"{chunk.ready_marker_path}.tmp"))
            failure_payload["artifacts"] = {
                "chunk_path": str(chunk.chunk_path),
                "annotated_path": str(chunk.annotated_path),
                "detection_json_path": str(chunk.detection_json_path),
            }
            failure_marker_path = self.raw_chunk_dir / f"chunk_{chunk.index:04d}.failed.json"

        write_json_atomic(failure_marker_path, failure_payload)


class StreamRecorder:
    def __init__(self, config: BackendConfig, repository: ChunkRepository, stop_event: threading.Event):
        self.config = config
        self.repository = repository
        self.stop_event = stop_event
        self.stream_quality = "unknown"
        self._process: Optional[subprocess.Popen] = None
        self._process_lock = threading.Lock()

    def _resolve_input_source(self) -> Tuple[str, str]:
        stream_url = self.config.stream_url
        if not stream_url:
            raise RuntimeError("JUTC_STREAM_URL is empty.")

        candidate_path = Path(stream_url)
        if candidate_path.exists():
            return str(candidate_path.resolve()), "file"

        lower_url = stream_url.lower()
        if ".m3u8" in lower_url or lower_url.endswith(".mp4") or lower_url.endswith(".ts"):
            return stream_url, "direct"

        session = streamlink.Streamlink()
        session.set_option("hls-live-edge", self.config.stream_live_edge)
        session.set_option("hls-segment-threads", self.config.stream_segment_threads)
        streams = session.streams(stream_url)
        stream_quality, stream = select_stream(streams, self.config.stream_quality_preference)
        return stream.url, stream_quality

    def _build_ffmpeg_command(self, input_source: str, start_index: int) -> List[str]:
        output_pattern = str(self.repository.raw_chunk_dir / "chunk_%04d.mp4")
        return [
            self.config.ffmpeg_binary,
            "-hide_banner",
            "-loglevel",
            "info",
            "-nostdin",
            "-y",
            "-i",
            input_source,
            "-map",
            "0:v:0",
            "-c",
            "copy",
            "-f",
            "segment",
            "-segment_time",
            str(self.config.chunk_duration_seconds),
            "-reset_timestamps",
            "1",
            "-segment_start_number",
            str(start_index),
            "-segment_format",
            "mp4",
            output_pattern,
        ]

    def _update_recording_state(
        self,
        *,
        status: str,
        current_chunk_index: Optional[int],
        latest_ready_index: Optional[int],
    ) -> None:
        payload = {
            "status": status,
            "stream_quality": self.stream_quality,
            "chunk_duration_seconds": self.config.chunk_duration_seconds,
            "updated_at": utc_now_iso(),
            "current_chunk_index": current_chunk_index,
            "current_chunk_started_epoch": time.time() if current_chunk_index else None,
            "latest_ready_index": latest_ready_index,
        }
        self.repository.write_recording_state(payload)

    def _parse_opened_chunk_index(self, line: str) -> Optional[int]:
        match = FFMPEG_OPEN_RE.search(line)
        if not match:
            return None
        return int(match.group(2))

    def stop(self) -> None:
        with self._process_lock:
            process = self._process
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()

    def run_forever(self) -> None:
        self.repository.ensure_directories()
        while not self.stop_event.is_set():
            start_index = self.repository.next_recording_index()
            latest_ready_index = start_index - 1 if start_index > 1 else None

            try:
                input_source, self.stream_quality = self._resolve_input_source()
            except Exception as exc:
                print(f"[RECORDER] Failed to resolve input stream: {exc}")
                if self.stop_event.wait(5.0):
                    break
                continue

            command = self._build_ffmpeg_command(input_source, start_index)
            print(
                f"[RECORDER] Starting ffmpeg | quality={self.stream_quality} "
                f"| next chunk={start_index:04d}"
            )
            print(f"[RECORDER] ffmpeg command: {subprocess.list2cmdline(command)}")

            try:
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                )
            except FileNotFoundError as exc:
                raise RuntimeError(
                    f"ffmpeg was not found. Install ffmpeg or set JUTC_FFMPEG_BINARY. {exc}"
                ) from exc

            with self._process_lock:
                self._process = process

            active_chunk_index: Optional[int] = None

            while not self.stop_event.is_set():
                line = process.stderr.readline()
                if line == "":
                    if process.poll() is not None:
                        break
                    continue

                line = line.strip()
                if not line:
                    continue

                opened_chunk_index = self._parse_opened_chunk_index(line)
                if opened_chunk_index is not None:
                    if active_chunk_index is not None and opened_chunk_index != active_chunk_index:
                        ready_chunk = self.repository.mark_chunk_ready(
                            active_chunk_index,
                            {"stream_quality": self.stream_quality},
                        )
                        if ready_chunk is not None:
                            latest_ready_index = ready_chunk.index
                            print(f"[RECORDER] Chunk ready: {ready_chunk.chunk_name}")

                    active_chunk_index = opened_chunk_index
                    print(f"[RECORDER] Chunk created: chunk_{opened_chunk_index:04d}.mp4")
                    self._update_recording_state(
                        status="recording",
                        current_chunk_index=active_chunk_index,
                        latest_ready_index=latest_ready_index,
                    )
                    continue

                lower_line = line.lower()
                if "error" in lower_line or "failed" in lower_line:
                    print(f"[RECORDER][ffmpeg] {line}")

            if self.stop_event.is_set():
                self.stop()

            return_code = process.wait()

            with self._process_lock:
                self._process = None

            if (return_code == 0 or self.stop_event.is_set()) and active_chunk_index is not None:
                ready_chunk = self.repository.mark_chunk_ready(
                    active_chunk_index,
                    {"stream_quality": self.stream_quality, "finalized_reason": "ffmpeg-exit"},
                )
                if ready_chunk is not None:
                    latest_ready_index = ready_chunk.index
                    print(f"[RECORDER] Chunk ready: {ready_chunk.chunk_name}")

            self.repository.write_recording_state(
                {
                    "status": "stopped" if self.stop_event.is_set() else "restarting",
                    "stream_quality": self.stream_quality,
                    "chunk_duration_seconds": self.config.chunk_duration_seconds,
                    "updated_at": utc_now_iso(),
                    "current_chunk_index": None,
                    "current_chunk_started_epoch": None,
                    "latest_ready_index": latest_ready_index,
                }
            )

            if self.stop_event.is_set():
                break

            print(f"[RECORDER] ffmpeg exited with code {return_code}. Restarting in 5 seconds.")
            if self.stop_event.wait(5.0):
                break


class AIChunkProcessor:
    def __init__(self, config: BackendConfig, repository: ChunkRepository, report_store):
        self.config = config
        self.repository = repository
        self.report_store = report_store
        self.model = YOLO(str(config.model_path))
        self.route_matcher = RouteScheduleMatcher(config.routes_path)
        self.zones = self._load_zones(config.zone_config_path)
        self.track_states: Dict[int, TrackState] = {}
        self.zone_route_cooldowns: Dict[Tuple[str, str], float] = {}
        self.global_frame_index = 0
        self.total_reports_emitted = 0
        self.current_detect_every_n_frames = config.detect_every_n_frames
        self.status_text = "Idle"

    def _load_zones(self, zone_config_path: Path) -> List[ZoneDefinition]:
        with Path(zone_config_path).open(encoding="utf-8") as config_file:
            raw_config = json.load(config_file)

        zones: List[ZoneDefinition] = []
        for zone_name, raw_zone in raw_config.items():
            route_map = raw_zone.get("routes") or {}
            candidate_routes = [
                self._build_candidate_route(route_number, destination)
                for route_number, destination in route_map.items()
            ]
            zones.append(
                ZoneDefinition(
                    name=zone_name,
                    polygon=[tuple(point) for point in raw_zone["polygon"]],
                    color=tuple(raw_zone["color"]),
                    route_targets=[
                        self._format_candidate_route(candidate)
                        for candidate in candidate_routes
                    ],
                    candidate_routes=candidate_routes,
                    schedule_stop_keywords=list(
                        raw_zone.get("schedule_stop_keywords") or ["WASHINGTON BOULEVARD"]
                    ),
                )
            )
        return zones

    def _infer_candidate_direction(self, route_number: str, destination: str) -> Optional[str]:
        normalized_route = normalize_route_number(route_number)
        normalized_destination = normalize_stop_name(destination)
        matched_directions = {
            record["direction_normalized"]
            for record in self.route_matcher.records
            if record["route_number"] == normalized_route
            and normalized_text_matches(record["destination_normalized"], normalized_destination)
        }
        if len(matched_directions) == 1:
            return next(iter(matched_directions))
        return None

    def _build_candidate_route(self, route_number: str, destination: str) -> Dict[str, str]:
        candidate = {
            "route": str(route_number).upper(),
            "destination": str(destination).upper(),
        }
        direction = self._infer_candidate_direction(candidate["route"], candidate["destination"])
        if direction:
            candidate["direction"] = direction
        return candidate

    @staticmethod
    def _format_candidate_route(candidate: Dict[str, str]) -> str:
        description = candidate["route"]
        if candidate.get("destination"):
            description = f"{description} {candidate['destination']}"
        if candidate.get("direction"):
            description = f"{description} [{candidate['direction']}]"
        return description

    def _build_track_prediction(self, zone: ZoneDefinition) -> Dict[str, object]:
        prediction_result = self.route_matcher.predict(
            candidate_routes=zone.candidate_routes,
            stop_keywords=zone.schedule_stop_keywords,
            service_day=self.config.service_day_override,
            top_n=self.config.top_route_predictions,
            early_tolerance_minutes=self.config.schedule_early_tolerance_minutes,
            late_tolerance_minutes=self.config.schedule_late_tolerance_minutes,
        )
        prediction_state = {
            "zone_name": zone.name,
            "service_day": prediction_result["service_day"],
            "predictions": prediction_result["predictions"],
            "expanded_candidates": prediction_result["expanded_candidates"],
            "stop_keywords": prediction_result["stop_keywords"],
            "route_targets": zone.route_targets,
        }
        return prediction_state

    def _cooldown_allows(self, zone_name: str, route_number: str) -> bool:
        cooldown_key = (zone_name, route_number)
        last_emit = self.zone_route_cooldowns.get(cooldown_key)
        if last_emit is None:
            return True
        return (time.time() - last_emit) >= self.config.report_cooldown_seconds

    def _mark_cooldown(self, zone_name: str, route_number: str) -> None:
        self.zone_route_cooldowns[(zone_name, route_number)] = time.time()

    def _save_snapshot(self, frame, bbox: Tuple[int, int, int, int], report_id: str) -> Optional[str]:
        x1, y1, x2, y2 = bbox
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        snapshot = frame[y1:y2, x1:x2]
        if snapshot.size == 0:
            return None
        self.config.snapshots_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = self.config.snapshots_dir / f"{report_id}.jpg"
        cv2.imwrite(str(snapshot_path), snapshot)
        return str(snapshot_path)

    def _build_report(
        self,
        *,
        track_id: int,
        zone: ZoneDefinition,
        bus_confidence: float,
        prediction_state: Dict[str, object],
        frame,
        bbox: Tuple[int, int, int, int],
        chunk: ChunkPaths,
        frame_index: int,
        frame_time_seconds: float,
    ) -> Optional[AIReport]:
        predictions = prediction_state.get("predictions") or []
        if not predictions:
            return None

        top_prediction = predictions[0]
        route_number = str(top_prediction.get("route_number") or "")
        if not self._cooldown_allows(zone.name, route_number):
            return None

        report_id = new_report_id()
        snapshot_path = self._save_snapshot(frame, bbox, report_id)
        prediction_summaries = [
            RoutePredictionSummary.from_prediction(prediction)
            for prediction in predictions
        ]

        report = AIReport(
            id=report_id,
            source="ai",
            camera_id=self.config.camera_id,
            stream_name=self.config.stream_name,
            detected_at=utc_now_iso(),
            track_id=track_id,
            zone_name=zone.name,
            bus_confidence=round(bus_confidence, 4),
            likely_route=route_number or None,
            route_confidence=int(top_prediction.get("probability") or 0),
            direction=top_prediction.get("direction"),
            destination=top_prediction.get("destination"),
            matched_stop=top_prediction.get("matched_stop"),
            schedule_instance_id=top_prediction.get("schedule_instance_id"),
            trip_start_time=top_prediction.get("trip_start_time"),
            trip_end_time=top_prediction.get("trip_end_time"),
            model_version=self.config.model_path.name,
            snapshot_path=snapshot_path,
            predictions=prediction_summaries,
            metadata={
                "bbox": list(bbox),
                "frame_index": frame_index,
                "chunk_name": chunk.chunk_name,
                "chunk_index": chunk.index,
                "chunk_timestamp_seconds": round(frame_time_seconds, 3),
                "service_day": prediction_state.get("service_day"),
                "expanded_candidates": prediction_state.get("expanded_candidates"),
                "stop_keywords": prediction_state.get("stop_keywords"),
                "stream_quality": chunk.ready_metadata.get("stream_quality"),
            },
        )
        self._mark_cooldown(zone.name, route_number)
        return report

    def _cleanup_stale_tracks(self) -> None:
        stale_track_ids = [
            track_id
            for track_id, state in self.track_states.items()
            if (self.global_frame_index - state.last_seen_frame) > 150
        ]
        for track_id in stale_track_ids:
            self.track_states.pop(track_id, None)

    def _estimate_delay_seconds(self, chunk_index: int, frame_time_seconds: float) -> float:
        recording_state = self.repository.read_recording_state()
        current_chunk_index = int(recording_state.get("current_chunk_index") or 0)
        current_chunk_started_epoch = recording_state.get("current_chunk_started_epoch")
        if current_chunk_index <= 0 or current_chunk_started_epoch is None:
            pending_chunks = len(self.repository.list_pending_chunks())
            return max(
                0.0,
                (pending_chunks * self.config.chunk_duration_seconds) - frame_time_seconds,
            )

        current_chunk_progress = max(0.0, time.time() - float(current_chunk_started_epoch))
        current_chunk_progress = min(current_chunk_progress, self.config.chunk_duration_seconds)
        delay_seconds = (
            (max(0, current_chunk_index - chunk_index) * self.config.chunk_duration_seconds)
            + current_chunk_progress
            - frame_time_seconds
        )
        return max(0.0, delay_seconds)

    def _build_prediction_label(self, report: Optional[AIReport]) -> Optional[str]:
        if report is None or not report.likely_route:
            return None
        if report.route_confidence is None:
            return f"Route {report.likely_route}"
        return f"Route {report.likely_route} {report.route_confidence}%"

    def _resolve_detect_every_n_frames(self, backlog_chunks: int) -> int:
        min_interval = self.config.detect_every_n_frames_min
        max_interval = self.config.detect_every_n_frames_max
        if not self.config.dynamic_detect_every_n_frames or min_interval == max_interval:
            return min_interval
        if backlog_chunks <= 1:
            return min_interval
        if backlog_chunks >= self.config.backlog_warning_chunks:
            return max_interval

        backlog_span = max(1, self.config.backlog_warning_chunks - 1)
        normalized_backlog = (backlog_chunks - 1) / backlog_span
        interpolated = min_interval + ((max_interval - min_interval) * normalized_backlog)
        return max(min_interval, min(max_interval, int(round(interpolated))))

    def _should_run_inference(self, frame_index: int, detect_every_n_frames: int) -> bool:
        interval = max(1, detect_every_n_frames)
        zero_based_frame_index = max(0, frame_index - 1)
        return (zero_based_frame_index % interval) == 0

    def _process_frame(
        self,
        *,
        chunk: ChunkPaths,
        frame,
        results,
        frame_index: int,
        frame_time_seconds: float,
    ) -> Tuple[np.ndarray, List[Dict[str, object]], List[Dict[str, object]], int]:
        source_height, source_width = frame.shape[:2]
        scale_x = FRAME_SIZE[0] / max(1, source_width)
        scale_y = FRAME_SIZE[1] / max(1, source_height)

        annotated = cv2.resize(frame.copy(), FRAME_SIZE)
        for zone in self.zones:
            draw_polygon_zone(annotated, zone)

        frame_detections: List[Dict[str, object]] = []
        frame_events: List[Dict[str, object]] = []
        bus_count = 0

        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            bus_count = len(boxes)

            for box in boxes:
                class_id = int(box.cls[0].item()) if box.cls is not None else 0
                class_name = lookup_class_name(self.model.names, class_id)
                confidence = float(box.conf[0]) if box.conf is not None else 0.0
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                display_x1, display_y1, display_x2, display_y2 = scale_box(
                    x1,
                    y1,
                    x2,
                    y2,
                    scale_x,
                    scale_y,
                )
                display_center = scale_point(box_center(x1, y1, x2, y2), scale_x, scale_y)
                track_id = int(box.id.item()) if box.id is not None else -1

                hit_zone_names: List[str] = []
                prediction_label = None
                if track_id >= 0:
                    track_state = self.track_states.setdefault(track_id, TrackState())
                    track_state.last_seen_frame = self.global_frame_index
                    track_state.max_bus_confidence = max(track_state.max_bus_confidence, confidence)
                    active_zones = set()

                    for zone in self.zones:
                        if not point_in_polygon(display_center, zone.polygon):
                            continue

                        active_zones.add(zone.name)
                        hit_zone_names.append(zone.name)
                        track_state.zone_hits[zone.name] = track_state.zone_hits.get(zone.name, 0) + 1

                        if track_state.zone_hits[zone.name] < self.config.stable_hits_required:
                            continue
                        if zone.name in track_state.emitted_zones:
                            continue

                        prediction_state = self._build_track_prediction(zone)
                        report = self._build_report(
                            track_id=track_id,
                            zone=zone,
                            bus_confidence=track_state.max_bus_confidence,
                            prediction_state=prediction_state,
                            frame=frame,
                            bbox=(x1, y1, x2, y2),
                            chunk=chunk,
                            frame_index=frame_index,
                            frame_time_seconds=frame_time_seconds,
                        )
                        if report is None:
                            continue

                        self.report_store.write_report(report.to_dict())
                        track_state.emitted_zones.add(zone.name)
                        self.total_reports_emitted += 1
                        prediction_label = self._build_prediction_label(report)

                        frame_events.append(
                            {
                                "type": "bus_entered_zone",
                                "chunk_index": chunk.index,
                                "chunk_name": chunk.chunk_name,
                                "frame_index": frame_index,
                                "chunk_timestamp_seconds": round(frame_time_seconds, 3),
                                "track_id": track_id,
                                "zone_name": zone.name,
                                "bus_confidence": round(track_state.max_bus_confidence, 4),
                                "likely_route": report.likely_route,
                                "route_confidence": report.route_confidence,
                                "report_id": report.id,
                            }
                        )
                        print(
                            f"[PROCESSOR] AI report {report.id} | chunk={chunk.chunk_name} "
                            f"| zone={report.zone_name} | route={report.likely_route}"
                        )

                    for zone_name in list(track_state.zone_hits):
                        if zone_name not in active_zones:
                            track_state.zone_hits[zone_name] = 0

                display_label_parts = [class_name, f"{confidence:.0%}"]
                if track_id >= 0:
                    display_label_parts.append(f"#{track_id}")
                if hit_zone_names:
                    display_label_parts.append("/".join(hit_zone_names))
                if prediction_label:
                    display_label_parts.append(prediction_label)
                display_label = " | ".join(display_label_parts)

                box_color = (0, 255, 0)
                for zone in self.zones:
                    if zone.name in hit_zone_names:
                        box_color = zone.color
                        break
                draw_detection_box(
                    annotated,
                    display_x1,
                    display_y1,
                    display_x2,
                    display_y2,
                    display_label,
                    box_color,
                )

                frame_detections.append(
                    {
                        "track_id": track_id if track_id >= 0 else None,
                        "label": class_name,
                        "confidence": round(confidence, 4),
                        "bbox": [x1, y1, x2, y2],
                        "display_bbox": [display_x1, display_y1, display_x2, display_y2],
                        "center": [display_center[0], display_center[1]],
                        "zones": hit_zone_names,
                        "prediction_label": prediction_label,
                    }
                )

        backlog_chunks = len(self.repository.list_pending_chunks())
        delay_seconds = self._estimate_delay_seconds(chunk.index, frame_time_seconds)
        timestamp_label = (
            f"{chunk.chunk_name}  t={format_seconds(frame_time_seconds)}  "
            f"delay={delay_seconds:.1f}s"
        )
        cv2.putText(
            annotated,
            timestamp_label,
            (20, FRAME_SIZE[1] - 20),
            ZONE_LABEL_FONT,
            0.7,
            (255, 255, 255),
            2,
        )
        draw_debug_hud(
            annotated,
            chunk_name=chunk.chunk_name,
            frame_index=frame_index,
            frame_time_seconds=frame_time_seconds,
            bus_count=bus_count,
            backlog_chunks=backlog_chunks,
            delay_seconds=delay_seconds,
            reports_emitted=self.total_reports_emitted,
            status=self.status_text,
        )
        return annotated, frame_detections, frame_events, bus_count

    def process_chunk(
        self,
        chunk: ChunkPaths,
        stop_event: threading.Event,
        detect_every_n_frames: int,
    ) -> None:
        self.status_text = f"Processing {chunk.chunk_name}"
        backlog_chunks = len(self.repository.list_pending_chunks())
        self.current_detect_every_n_frames = detect_every_n_frames
        print(
            f"[PROCESSOR] Starting {chunk.chunk_name} | backlog={backlog_chunks} ready chunks "
            f"| detect_every_n_frames={detect_every_n_frames} "
            f"| detect_range={self.config.detect_every_n_frames_min}-{self.config.detect_every_n_frames_max} "
            f"| dynamic_detect={self.config.dynamic_detect_every_n_frames} "
            f"| save_annotated_video={self.config.save_annotated_video}"
        )

        capture = cv2.VideoCapture(str(chunk.chunk_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open chunk file: {chunk.chunk_path}")

        timing_info = probe_video_timing(
            chunk.chunk_path,
            capture,
            self.config.ffprobe_binary,
        )
        source_fps = timing_info.source_fps
        output_fps = timing_info.output_fps
        print(
            f"[PROCESSOR] Timing for {chunk.chunk_name} | "
            f"source_fps={source_fps:.3f} ({timing_info.fps_source}) | "
            f"output_fps={output_fps:.3f} | "
            f"reported_frames={timing_info.total_frames_reported} | "
            f"source_duration={timing_info.source_duration_seconds:.3f}s"
        )
        if abs(output_fps - source_fps) > 0.01:
            print(
                f"[PROCESSOR][WARN] Output FPS {output_fps:.3f} does not match "
                f"source FPS {source_fps:.3f} for {chunk.chunk_name}"
            )
        if self.config.show_debug:
            print(
                "[PROCESSOR] Debug viewer is enabled. Viewer speed does not change the saved "
                "annotated file timing; it only affects how fast you watch it live."
            )

        writer = None
        if self.config.save_annotated_video:
            writer = cv2.VideoWriter(
                str(chunk.annotated_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                output_fps,
                FRAME_SIZE,
            )
            if not writer.isOpened():
                capture.release()
                raise RuntimeError(f"Could not open annotated writer: {chunk.annotated_path}")
        else:
            print(
                f"[PROCESSOR] Annotated video saving disabled for {chunk.chunk_name}; "
                "skipping video encoding."
            )

        frames_payload: List[Dict[str, object]] = []
        chunk_events: List[Dict[str, object]] = []
        started_at = utc_now_iso()
        processing_started_at = time.perf_counter()
        frame_index = 0
        decoded_frame_count = 0
        processed_frame_count = 0
        inference_frame_count = 0
        written_frame_count = 0
        last_frame_time_seconds = 0.0
        last_inference_results = None

        while not stop_event.is_set():
            ok, frame = capture.read()
            if not ok:
                break

            frame_index += 1
            decoded_frame_count += 1
            self.global_frame_index += 1

            pos_msec = float(capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
            if pos_msec > 0:
                frame_time_seconds = pos_msec / 1000.0
            else:
                frame_time_seconds = (frame_index - 1) / source_fps
            last_frame_time_seconds = frame_time_seconds

            ran_inference = self._should_run_inference(frame_index, detect_every_n_frames)
            if ran_inference:
                inference_frame_count += 1
                last_inference_results = self.model.track(
                    frame,
                    verbose=False,
                    conf=self.config.confidence,
                    persist=True,
                    imgsz=self.config.yolo_imgsz,
                )
            active_results = last_inference_results

            annotated, frame_detections, frame_events, _ = self._process_frame(
                chunk=chunk,
                frame=frame,
                results=active_results,
                frame_index=frame_index,
                frame_time_seconds=frame_time_seconds,
            )
            processed_frame_count += 1
            if writer is not None:
                writer.write(annotated)
                written_frame_count += 1

            if self.global_frame_index % 120 == 0:
                self._cleanup_stale_tracks()

            frames_payload.append(
                {
                    "frame_index": frame_index,
                    "chunk_timestamp_seconds": round(frame_time_seconds, 3),
                    "approx_live_delay_seconds": round(
                        self._estimate_delay_seconds(chunk.index, frame_time_seconds),
                        3,
                    ),
                    "inference_ran": ran_inference,
                    "detection_source": (
                        "inference"
                        if ran_inference
                        else ("cached" if active_results is not None else "none")
                    ),
                    "detected_objects": frame_detections,
                    "events": frame_events,
                }
            )
            chunk_events.extend(frame_events)

            if self.config.show_debug:
                cv2.imshow(WINDOW_NAME, annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stop_event.set()
                    break

        capture.release()
        if writer is not None:
            writer.release()
        processing_elapsed_seconds = time.perf_counter() - processing_started_at

        annotated_duration_seconds = None
        if writer is not None and output_fps > 0:
            annotated_duration_seconds = written_frame_count / output_fps
        if timing_info.source_duration_seconds <= 0 and decoded_frame_count > 0:
            timing_info.source_duration_seconds = max(
                last_frame_time_seconds,
                decoded_frame_count / source_fps if source_fps > 0 else 0.0,
            )

        output_duration_label = (
            f"{annotated_duration_seconds:.3f}s"
            if annotated_duration_seconds is not None
            else "disabled"
        )

        print(
            f"[PROCESSOR] Summary for {chunk.chunk_name} | "
            f"decoded={decoded_frame_count} | processed={processed_frame_count} | "
            f"inference={inference_frame_count} | written={written_frame_count} | "
            f"detect_every_n_frames={detect_every_n_frames} | "
            f"processing_time={processing_elapsed_seconds:.3f}s | "
            f"output_fps={output_fps:.3f} | output_duration={output_duration_label}"
        )
        if writer is not None and decoded_frame_count != written_frame_count:
            print(
                f"[PROCESSOR][WARN] Decoded frames ({decoded_frame_count}) do not match "
                f"written frames ({written_frame_count}) for {chunk.chunk_name}"
            )
        if decoded_frame_count != processed_frame_count:
            print(
                f"[PROCESSOR][WARN] Decoded frames ({decoded_frame_count}) do not match "
                f"processed frames ({processed_frame_count}) for {chunk.chunk_name}"
            )
        if (
            detect_every_n_frames > 1
            and processed_frame_count > 1
            and inference_frame_count >= processed_frame_count
        ):
            print(
                f"[PROCESSOR][WARN] Inference frames ({inference_frame_count}) should be lower than "
                f"processed frames ({processed_frame_count}) for {chunk.chunk_name}"
            )
        if timing_info.total_frames_reported > 0 and timing_info.total_frames_reported != decoded_frame_count:
            print(
                f"[PROCESSOR][WARN] Reported input frame count ({timing_info.total_frames_reported}) "
                f"differs from decoded frame count ({decoded_frame_count}) for {chunk.chunk_name}"
            )
        if writer is not None and timing_info.source_duration_seconds > 0:
            duration_delta = abs(timing_info.source_duration_seconds - annotated_duration_seconds)
            if duration_delta > 0.25:
                print(
                    f"[PROCESSOR][WARN] Annotated duration ({annotated_duration_seconds:.3f}s) "
                    f"differs from source duration ({timing_info.source_duration_seconds:.3f}s) "
                    f"by {duration_delta:.3f}s for {chunk.chunk_name}"
                )

        payload = {
            "chunk_index": chunk.index,
            "chunk_name": chunk.chunk_name,
            "chunk_path": str(chunk.chunk_path),
            "ready_at": chunk.ready_metadata.get("ready_at"),
            "processed_started_at": started_at,
            "processed_finished_at": utc_now_iso(),
            "save_annotated_video": self.config.save_annotated_video,
            "annotated_video_path": str(chunk.annotated_path) if writer is not None else None,
            "source_fps": source_fps,
            "source_fps_origin": timing_info.fps_source,
            "output_fps": output_fps,
            "detect_every_n_frames": detect_every_n_frames,
            "detect_every_n_frames_min": self.config.detect_every_n_frames_min,
            "detect_every_n_frames_max": self.config.detect_every_n_frames_max,
            "dynamic_detect_every_n_frames": self.config.dynamic_detect_every_n_frames,
            "reported_input_frames": timing_info.total_frames_reported,
            "decoded_frames": decoded_frame_count,
            "processed_frames": processed_frame_count,
            "inference_frames": inference_frame_count,
            "written_frames": written_frame_count,
            "processing_time_seconds": round(processing_elapsed_seconds, 3),
            "source_duration_seconds": round(timing_info.source_duration_seconds, 3),
            "annotated_output_duration_seconds": (
                round(annotated_duration_seconds, 3)
                if annotated_duration_seconds is not None
                else None
            ),
            "frame_size": {
                "width": FRAME_SIZE[0],
                "height": FRAME_SIZE[1],
            },
            "status": "interrupted" if stop_event.is_set() else "processed",
            "events": chunk_events,
            "frames": frames_payload,
        }
        write_json_atomic(chunk.detection_json_path, payload)
        print(
            f"[PROCESSOR] Finished {chunk.chunk_name} | frames={frame_index} "
            f"| events={len(chunk_events)}"
        )
        removed_count = self.repository.cleanup_processed_chunk(chunk)
        if removed_count > 0:
            print(
                f"[PROCESSOR] Cleaned up {removed_count} generated files for {chunk.chunk_name}"
            )

    def run_forever(self, stop_event: threading.Event) -> None:
        if self.config.show_debug:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW_NAME, *FRAME_SIZE)

        try:
            while not stop_event.is_set():
                pending_chunks = self.repository.list_pending_chunks()
                backlog_chunks = len(pending_chunks)

                if backlog_chunks == 0:
                    self.current_detect_every_n_frames = self.config.detect_every_n_frames_min
                    self.status_text = "Waiting for next completed chunk"
                    if self.config.show_debug:
                        frame = make_status_frame(
                            "Waiting for next chunk...",
                            f"Chunk duration={self.config.chunk_duration_seconds}s",
                        )
                        cv2.imshow(WINDOW_NAME, frame)
                        if cv2.waitKey(200) & 0xFF == ord("q"):
                            stop_event.set()
                            break
                    else:
                        stop_event.wait(self.config.processor_poll_interval_seconds)
                    continue

                if backlog_chunks >= self.config.backlog_warning_chunks:
                    print(
                        f"[PROCESSOR] Warning: backlog has grown to {backlog_chunks} chunks."
                    )

                next_chunk = pending_chunks[0]
                effective_detect_every_n_frames = self._resolve_detect_every_n_frames(
                    backlog_chunks
                )
                if effective_detect_every_n_frames != self.current_detect_every_n_frames:
                    print(
                        f"[PROCESSOR] Adjusting detect_every_n_frames to "
                        f"{effective_detect_every_n_frames} | backlog={backlog_chunks}"
                    )
                try:
                    self.process_chunk(
                        next_chunk,
                        stop_event,
                        effective_detect_every_n_frames,
                    )
                except Exception as exc:
                    self.status_text = "Processing error"
                    print(f"[PROCESSOR] Error while processing {next_chunk.chunk_name}: {exc}")
                    self.repository.mark_chunk_failed(next_chunk, str(exc))
                    if stop_event.wait(self.config.processor_poll_interval_seconds):
                        break
        finally:
            if self.config.show_debug:
                cv2.destroyAllWindows()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Delayed chunk-based JUTC bus detector."
    )
    parser.add_argument(
        "--mode",
        choices=["all", "recorder", "processor"],
        default="all",
        help="Run the recorder, the processor, or both together.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_backend_config()
    repository = ChunkRepository(config)
    if config.clean_startup_chunks:
        removed_count = repository.cleanup_startup_runtime_files()
        if removed_count > 0:
            print(f"[STARTUP] Removed {removed_count} stale chunk artifacts from previous runs.")
    stop_event = threading.Event()

    recorder = None
    recorder_thread = None
    if args.mode in {"all", "recorder"}:
        recorder = StreamRecorder(config, repository, stop_event)

    processor = None
    if args.mode in {"all", "processor"}:
        report_store = build_report_store(config)
        processor = AIChunkProcessor(config, repository, report_store)

    try:
        if args.mode == "recorder":
            recorder.run_forever()
            return

        if args.mode == "processor":
            processor.run_forever(stop_event)
            return

        recorder_thread = threading.Thread(target=recorder.run_forever, daemon=True)
        recorder_thread.start()
        processor.run_forever(stop_event)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        if recorder is not None:
            recorder.stop()
        if recorder_thread is not None:
            recorder_thread.join(timeout=5)


if __name__ == "__main__":
    main()
