import json
import re
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from zoneinfo import ZoneInfo


DAY_ALIASES = {
    "weekday": "Weekday",
    "weekdays": "Weekday",
    "saturday": "Saturday",
    "turday": "Saturday",
    "sunday": "Sunday",
    "public holiday": "Public Holiday",
}

ROUTE_TOKEN_RE = re.compile(r"[^A-Z0-9]+")
STOP_TOKEN_RE = re.compile(r"[^A-Z0-9]+")
TIME_RE = re.compile(r"^\s*(\d{1,2}):(\d{2})\s*([AP]M)?\s*$", re.IGNORECASE)
JAMAICA_TIMEZONE = ZoneInfo("America/Jamaica")
WASHINGTON_BLVD_CANONICAL_TOKENS = (
    frozenset(("MOLYNES", "ROAD", "WASHINGTON", "BOULEVARD")),
)
WASHINGTON_BLVD_INFERENCE_MAP = {
    "15A|Outbound": {
        "group": "PORTMORE",
        "prev": "DUHANEY PARK",
        "next": "HALF WAY TREE",
    },
    "16A|Outbound": {
        "group": "PORTMORE",
        "prev": "DUHANEY PARK",
        "next": "HALF WAY TREE",
    },
    "16B|Outbound": {
        "group": "PORTMORE",
        "prev": "DUHANEY PARK",
        "next": "HALF WAY TREE",
    },
    "16BX|Outbound": {
        "group": "PORTMORE",
        "prev": "DUHANEY PARK",
        "next": "DUNROBIN AVENUE / CONSTANT SPRING ROAD",
    },
    "21AX|Outbound": {
        "group": "SPANISH TOWN",
        "prev": "DUHANEY PARK",
        "next": "HALF WAY TREE",
    },
    "24EX|Outbound": {
        "group": "SPANISH TOWN",
        "prev": "DUHANEY PARK",
        "next": "CONSTANT SPRING ROAD",
    },
    "30|Outbound": {
        "group": "CHANCERY STREET",
        "prev": "DUHANEY PARK",
        "next": "HALF WAY TREE",
    },
    "75|Outbound": {
        "group": "PAPINE",
        "prev": "FERRY",
        "next": "HALF WAY TREE",
    },
    "75A|Outbound": {
        "group": "PAPINE",
        "prev": "FERRY",
        "next": "HALF WAY TREE",
    },
    "75AX|Outbound": {
        "group": "PAPINE",
        "prev": "FERRY",
        "next": "HOPE ROAD / WATERLOO ROAD",
    },
}


def jamaica_now():
    return datetime.now(JAMAICA_TIMEZONE)


def normalize_service_day(day_name):
    if not day_name:
        return None
    return DAY_ALIASES.get(str(day_name).strip().lower(), str(day_name).strip())


def infer_service_day(now=None):
    now = now or jamaica_now()
    weekday = now.weekday()
    if weekday < 5:
        return "Weekday"
    if weekday == 5:
        return "Saturday"
    return "Sunday"


def normalize_route_number(route_number):
    if route_number is None:
        return ""
    return ROUTE_TOKEN_RE.sub("", str(route_number).upper())


def normalize_stop_name(stop_name):
    if stop_name is None:
        return ""
    return STOP_TOKEN_RE.sub(" ", str(stop_name).upper()).strip()


def stop_name_tokens(stop_name):
    normalized = normalize_stop_name(stop_name)
    if not normalized:
        return frozenset()
    return frozenset(token for token in normalized.split() if token)


def normalize_checkpoint_name(stop_name):
    tokens = stop_name_tokens(stop_name)
    if tokens in WASHINGTON_BLVD_CANONICAL_TOKENS:
        return "MOLYNES ROAD WASHINGTON BOULEVARD"
    return normalize_stop_name(stop_name)


def normalize_direction(direction_name):
    if not direction_name:
        return ""

    normalized = str(direction_name).strip().lower()
    if "inbound" in normalized:
        return "Inbound"
    if "outbound" in normalized:
        return "Outbound"
    return str(direction_name).strip().title()


def normalized_text_matches(actual_value, expected_value):
    if not expected_value:
        return True
    if not actual_value:
        return False

    if actual_value == expected_value:
        return True
    if expected_value in actual_value or actual_value in expected_value:
        return True

    return SequenceMatcher(None, actual_value, expected_value).ratio() >= 0.8


def extract_candidate_route_number(candidate):
    if isinstance(candidate, dict):
        return candidate.get("route_number") or candidate.get("route")
    return candidate


def candidate_allows_prefix(candidate):
    if not isinstance(candidate, dict):
        return True
    return bool(candidate.get("allow_prefix"))


def parse_time_to_minutes(time_value):
    if time_value is None:
        return None

    match = TIME_RE.match(str(time_value))
    if not match:
        return None

    hour = int(match.group(1))
    minute = int(match.group(2))
    meridiem = match.group(3)

    if meridiem:
        meridiem = meridiem.upper()
        if hour == 0 and minute == 0:
            return None
        hour %= 12
        if meridiem == "PM":
            hour += 12
    elif hour == 0 and minute == 0:
        return None

    return hour * 60 + minute


def format_minutes_as_time(minutes_since_midnight):
    total_minutes = int(round(minutes_since_midnight)) % (24 * 60)
    hour = total_minutes // 60
    minute = total_minutes % 60
    suffix = "AM" if hour < 12 else "PM"
    display_hour = hour % 12 or 12
    return f"{display_hour}:{minute:02d} {suffix}"


def circular_delta_minutes(now_minutes, scheduled_minutes):
    delta = now_minutes - scheduled_minutes
    if delta > 12 * 60:
        delta -= 24 * 60
    elif delta < -12 * 60:
        delta += 24 * 60
    return delta


def format_delta_label(delta_minutes):
    rounded = int(round(delta_minutes))
    if rounded == 0:
        return "on time"
    if rounded > 0:
        return f"{rounded} min late"
    return f"{abs(rounded)} min early"


def is_within_prediction_window(delta_minutes, early_tolerance_minutes, late_tolerance_minutes):
    return (-float(early_tolerance_minutes)) <= float(delta_minutes) <= float(
        late_tolerance_minutes
    )


def time_based_plausibility_score(
    delta_minutes,
    early_tolerance_minutes,
    late_tolerance_minutes,
):
    delta_minutes = float(delta_minutes)
    if delta_minutes == 0:
        return 1.0

    if delta_minutes < 0:
        tolerance = max(float(early_tolerance_minutes), 0.0)
        distance = abs(delta_minutes)
    else:
        tolerance = max(float(late_tolerance_minutes), 0.0)
        distance = delta_minutes

    if tolerance <= 0 or distance > tolerance:
        return 0.0

    # Keep edge-of-window candidates weak but still comparable until they age out.
    return max(0.05, 1.0 - (distance / tolerance))


def departure_sort_key(range_name):
    digits = "".join(ch for ch in str(range_name) if ch.isdigit())
    if digits:
        return int(digits)
    return 10**9


class RouteScheduleMatcher:
    def __init__(self, routes_path):
        self.routes_path = Path(routes_path)
        with self.routes_path.open(encoding="utf-8") as route_file:
            raw_records = json.load(route_file)

        self.records = [self._prepare_record(record) for record in raw_records]
        self.known_routes = sorted(
            {
                record["route_number"]
                for record in self.records
                if record["route_number"]
            }
        )

    def _prepare_record(self, record):
        via_stops = list(record.get("via") or [])
        departures = {
            range_name: [parse_time_to_minutes(value) for value in times]
            for range_name, times in (record.get("departures") or {}).items()
        }

        return {
            "route_number": normalize_route_number(record.get("route_number")),
            "route_number_display": str(record.get("route_number", "")).upper(),
            "day": normalize_service_day(record.get("day")),
            "direction": str(record.get("direction") or "Unknown"),
            "direction_normalized": normalize_direction(record.get("direction")),
            "origin": str(record.get("origin") or ""),
            "origin_normalized": normalize_stop_name(record.get("origin")),
            "destination": str(record.get("destination") or ""),
            "destination_normalized": normalize_stop_name(record.get("destination")),
            "via": via_stops,
            "via_normalized": [normalize_stop_name(stop) for stop in via_stops],
            "departures": departures,
        }

    def expand_candidate_routes(self, candidate_routes):
        expanded, _ = self._prepare_candidate_filters(candidate_routes)
        return expanded

    def _prepare_candidate_filters(self, candidate_routes):
        expanded = set()
        filters_by_route = {}

        for candidate in candidate_routes or []:
            normalized = normalize_route_number(extract_candidate_route_number(candidate))
            if not normalized:
                continue

            route_matches = {
                route for route in self.known_routes if route == normalized
            }

            if normalized.isdigit() and candidate_allows_prefix(candidate):
                route_matches.update(
                    route for route in self.known_routes if route.startswith(normalized)
                )

            if route_matches:
                expanded.update(route_matches)

            if not isinstance(candidate, dict):
                continue

            candidate_filter = {
                "direction": normalize_direction(candidate.get("direction")),
                "origin": normalize_stop_name(candidate.get("origin")),
                "destination": normalize_stop_name(candidate.get("destination")),
            }

            if not any(candidate_filter.values()):
                continue

            for route in route_matches:
                filters_by_route.setdefault(route, []).append(candidate_filter)

        return expanded, filters_by_route

    def _record_matches_candidate_filters(self, record, filters_by_route):
        route_filters = filters_by_route.get(record["route_number"])
        if not route_filters:
            return True

        for candidate_filter in route_filters:
            if (
                candidate_filter["direction"]
                and record["direction_normalized"] != candidate_filter["direction"]
            ):
                continue

            if (
                candidate_filter["origin"]
                and not normalized_text_matches(
                    record["origin_normalized"],
                    candidate_filter["origin"],
                )
            ):
                continue

            if (
                candidate_filter["destination"]
                and not normalized_text_matches(
                    record["destination_normalized"],
                    candidate_filter["destination"],
                )
            ):
                continue

            return True

        return False

    def _matching_stop_indices(self, record, stop_keywords):
        if not stop_keywords:
            return list(range(len(record["via"])))

        normalized_keywords = [
            normalize_stop_name(keyword) for keyword in stop_keywords if keyword
        ]
        if not normalized_keywords:
            return list(range(len(record["via"])))

        # Preserve caller order so camera configs can express checkpoint priority.
        # Example: ["DAWKINS DRIVE", "PORTMORE MALL"] will use Dawkins Drive
        # whenever a route contains both stops, and fall back to Portmore Mall
        # only for routes that do not include Dawkins Drive.
        for keyword in normalized_keywords:
            matched_indices = [
                index
                for index, stop_name in enumerate(record["via_normalized"])
                if keyword in stop_name
            ]
            if matched_indices:
                return matched_indices

        return []

    def _record_supports_camera_inference(
        self,
        record,
        stop_keywords,
        inference_map=WASHINGTON_BLVD_INFERENCE_MAP,
    ):
        if not stop_keywords:
            return False

        normalized_keywords = {
            normalize_checkpoint_name(keyword) for keyword in stop_keywords if keyword
        }
        if "MOLYNES ROAD WASHINGTON BOULEVARD" not in normalized_keywords:
            return False

        route_key = f"{record['route_number_display']}|{record['direction_normalized']}"
        return route_key in inference_map

    def _find_stop_index(self, record, stop_name):
        normalized_target = normalize_checkpoint_name(stop_name)
        target_tokens = stop_name_tokens(stop_name)

        for index, actual_stop_name in enumerate(record["via"]):
            normalized_actual = normalize_checkpoint_name(actual_stop_name)
            if normalized_actual == normalized_target:
                return index

            actual_tokens = stop_name_tokens(actual_stop_name)
            if target_tokens and actual_tokens and (
                target_tokens <= actual_tokens or actual_tokens <= target_tokens
            ):
                return index

            if normalized_text_matches(normalized_actual, normalized_target):
                return index

        return None

    def _get_exact_checkpoint_time(self, record, range_key, camera_name):
        stop_index = self._find_stop_index(record, camera_name)
        if stop_index is None:
            return None

        stop_times = record["departures"].get(range_key) or []
        if stop_index >= len(stop_times):
            return None

        scheduled_minutes = stop_times[stop_index]
        if scheduled_minutes is None:
            return None

        return {
            "stop_index": stop_index,
            "stop_name": record["via"][stop_index],
            "scheduled_minutes": scheduled_minutes,
        }

    def _infer_midpoint_checkpoint_time(self, record, range_key, camera_name, inference_map):
        route_key = f"{record['route_number_display']}|{record['direction_normalized']}"
        inference_config = inference_map.get(route_key)
        if not inference_config:
            return None

        prev_index = self._find_stop_index(record, inference_config.get("prev"))
        next_index = self._find_stop_index(record, inference_config.get("next"))
        if prev_index is None or next_index is None or prev_index == next_index:
            return None

        stop_times = record["departures"].get(range_key) or []
        if prev_index >= len(stop_times) or next_index >= len(stop_times):
            return None

        prev_minutes = stop_times[prev_index]
        next_minutes = stop_times[next_index]
        if prev_minutes is None or next_minutes is None:
            return None
        if prev_minutes >= next_minutes:
            return None

        return {
            "stop_index": -1,
            "stop_name": camera_name,
            "scheduled_minutes": (prev_minutes + next_minutes) / 2.0,
            "inferred_from": {
                "group": inference_config.get("group"),
                "prev": record["via"][prev_index],
                "next": record["via"][next_index],
            },
        }

    def get_camera_checkpoint_time(
        self,
        record,
        range_key,
        camera_name,
        inference_map=WASHINGTON_BLVD_INFERENCE_MAP,
    ):
        exact_match = self._get_exact_checkpoint_time(record, range_key, camera_name)
        if exact_match is not None:
            return exact_match
        return self._infer_midpoint_checkpoint_time(
            record,
            range_key,
            camera_name,
            inference_map,
        )

    def _best_trip_for_record(self, record, stop_indices, now_minutes):
        best_match = None

        for range_name, stop_times in sorted(
            record["departures"].items(),
            key=lambda item: departure_sort_key(item[0]),
        ):
            range_candidates = []
            for stop_index in stop_indices:
                if stop_index >= len(stop_times):
                    continue

                scheduled_minutes = stop_times[stop_index]
                if scheduled_minutes is None:
                    continue

                range_candidates.append(
                    {
                        "stop_index": stop_index,
                        "stop_name": record["via"][stop_index],
                        "scheduled_minutes": scheduled_minutes,
                    }
                )

            if not range_candidates:
                inferred_candidate = self.get_camera_checkpoint_time(
                    record,
                    range_name,
                    "WASHINGTON BOULEVARD / MOLYNES ROAD",
                )
                if inferred_candidate is not None:
                    range_candidates.append(inferred_candidate)

            for checkpoint_candidate in range_candidates:
                scheduled_minutes = checkpoint_candidate["scheduled_minutes"]
                delta_minutes = circular_delta_minutes(now_minutes, scheduled_minutes)
                candidate = {
                    "range_name": range_name,
                    "stop_index": checkpoint_candidate["stop_index"],
                    "stop_name": checkpoint_candidate["stop_name"],
                    "scheduled_minutes": scheduled_minutes,
                    "scheduled_time": format_minutes_as_time(scheduled_minutes),
                    "delta_minutes": delta_minutes,
                    "delta_label": format_delta_label(delta_minutes),
                }
                if "inferred_from" in checkpoint_candidate:
                    candidate["inferred_from"] = checkpoint_candidate["inferred_from"]

                if best_match is None or abs(candidate["delta_minutes"]) < abs(
                    best_match["delta_minutes"]
                ):
                    best_match = candidate

        return best_match

    def _build_schedule_instance_metadata(self, record, best_trip):
        stop_times = record["departures"].get(best_trip["range_name"]) or []
        valid_times = [value for value in stop_times if value is not None]
        if not valid_times:
            return {
                "schedule_instance_id": None,
                "trip_start_minutes": None,
                "trip_end_minutes": None,
                "trip_start_time": "",
                "trip_end_time": "",
            }

        trip_start_minutes = min(valid_times)
        trip_end_minutes = max(valid_times)
        schedule_instance_id = "|".join(
            [
                record["route_number_display"],
                record["day"] or "",
                record["direction_normalized"] or record["direction"] or "",
                best_trip["range_name"],
                format_minutes_as_time(trip_start_minutes),
            ]
        )
        return {
            "schedule_instance_id": schedule_instance_id,
            "trip_start_minutes": trip_start_minutes,
            "trip_end_minutes": trip_end_minutes,
            "trip_start_time": format_minutes_as_time(trip_start_minutes),
            "trip_end_time": format_minutes_as_time(trip_end_minutes),
        }

    def _build_prediction_window_metadata(
        self,
        best_trip,
        early_tolerance_minutes,
        late_tolerance_minutes,
    ):
        # Scheduled passages stay plausible across a rolling early/late window instead
        # of being treated as a one-time slot assignment.
        window_start_minutes = best_trip["scheduled_minutes"] - float(
            early_tolerance_minutes
        )
        window_end_minutes = best_trip["scheduled_minutes"] + float(
            late_tolerance_minutes
        )
        return {
            "prediction_window_start_minutes": window_start_minutes,
            "prediction_window_end_minutes": window_end_minutes,
            "prediction_window_start": format_minutes_as_time(window_start_minutes),
            "prediction_window_end": format_minutes_as_time(window_end_minutes),
        }

    def predict(
        self,
        candidate_routes,
        stop_keywords=None,
        now=None,
        service_day=None,
        top_n=3,
        early_tolerance_minutes=4.0,
        late_tolerance_minutes=8.0,
    ):
        now = now or jamaica_now()
        service_day = normalize_service_day(service_day) or infer_service_day(now)
        now_minutes = now.hour * 60 + now.minute + (now.second / 60.0)
        expanded_candidates, filters_by_route = self._prepare_candidate_filters(candidate_routes)
        predictions_by_route = {}

        for record in self.records:
            if record["day"] != service_day:
                continue

            if record["route_number"] not in expanded_candidates:
                continue

            if not self._record_matches_candidate_filters(record, filters_by_route):
                continue

            stop_indices = self._matching_stop_indices(record, stop_keywords)
            if (
                stop_keywords
                and not stop_indices
                and not self._record_supports_camera_inference(record, stop_keywords)
            ):
                continue

            best_trip = self._best_trip_for_record(record, stop_indices, now_minutes)
            if not best_trip:
                continue

            if not is_within_prediction_window(
                best_trip["delta_minutes"],
                early_tolerance_minutes,
                late_tolerance_minutes,
            ):
                continue

            window_metadata = self._build_prediction_window_metadata(
                best_trip,
                early_tolerance_minutes,
                late_tolerance_minutes,
            )

            raw_score = time_based_plausibility_score(
                best_trip["delta_minutes"],
                early_tolerance_minutes,
                late_tolerance_minutes,
            )

            if not stop_keywords:
                raw_score *= 0.75

            route_number = record["route_number_display"]
            candidate_prediction = {
                "route_number": route_number,
                "raw_score": raw_score,
                "direction": record["direction"],
                "origin": record["origin"],
                "destination": record["destination"],
                "matched_stop": best_trip["stop_name"],
                "matched_stop_index": best_trip["stop_index"],
                "scheduled_minutes": best_trip["scheduled_minutes"],
                "scheduled_time": best_trip["scheduled_time"],
                "delta_minutes": best_trip["delta_minutes"],
                "delta_label": best_trip["delta_label"],
                "range_name": best_trip["range_name"],
            }
            candidate_prediction.update(window_metadata)
            candidate_prediction.update(
                self._build_schedule_instance_metadata(record, best_trip)
            )

            existing_prediction = predictions_by_route.get(route_number)
            if existing_prediction is None or raw_score > existing_prediction["raw_score"]:
                predictions_by_route[route_number] = candidate_prediction

        ranked_predictions = sorted(
            predictions_by_route.values(),
            key=lambda item: (-item["raw_score"], abs(item["delta_minutes"]), item["route_number"]),
        )

        top_predictions = ranked_predictions[:top_n]
        total_score = sum(item["raw_score"] for item in top_predictions)

        for item in top_predictions:
            if total_score > 0:
                probability = round((item["raw_score"] / total_score) * 100)
            else:
                probability = 0
            item["probability"] = int(probability)
            item.pop("raw_score", None)

        return {
            "service_day": service_day,
            "stop_keywords": list(stop_keywords or []),
            "expanded_candidates": sorted(expanded_candidates),
            "predictions": top_predictions,
        }
