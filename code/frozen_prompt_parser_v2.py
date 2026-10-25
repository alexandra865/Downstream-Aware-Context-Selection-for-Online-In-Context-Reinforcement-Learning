#!/usr/bin/env python3
"""Frozen observable prompt parser shared by state selection and H20 features."""

from __future__ import annotations

import math
import re
from typing import Any


SCHEMA_VERSION = "conflict_time_parser_v2"

CONFLICT_TIME_PATTERNS = (
    r"expected\s+to\s+overlap\s+in\s+"
    r"(?:approximately|about|around)?\s*(\d+(?:\.\d+)?)\s*(?:s|seconds?)\b",
    r"predicted\s+overlap[^\n]*?(\d+(?:\.\d+)?)\s*(?:s|seconds?)\b",
    r"reach\s+the\s+colli(?:de|sion)\s+point[^\n]*?"
    r"(\d+(?:\.\d+)?)\s*(?:s|seconds?)\b",
)
NEARBY_VEHICLE_PATTERNS = (
    r"(?m)^\s*-\s*Vehicle\s+'[^']+'",
    r"(?m)^\s*Vehicle\s+'[^']+'",
    r"(?m)^\s*-\s*The\s+vehicle\s+'[^']+'",
)


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def parse_lane(prompt: str) -> tuple[float, float]:
    count_patterns = (
        r"road with\s+(\d+)\s+lanes? in your direction",
        r"There are\s+(\d+)\s+lanes?",
    )
    index_patterns = (
        r"number\s+(\d+)\s+lane from the left",
        r"lane number\s+(\d+)",
    )
    count = index = math.nan
    for pattern in count_patterns:
        match = re.search(pattern, prompt, flags=re.I)
        if match:
            count = float(match.group(1))
            break
    for pattern in index_patterns:
        match = re.search(pattern, prompt, flags=re.I)
        if match:
            index = float(match.group(1))
            break
    return count, index


def extract_conflict_times(prompt: Any) -> list[float]:
    text = str(prompt or "")
    return [
        float(value)
        for pattern in CONFLICT_TIME_PATTERNS
        for value in re.findall(pattern, text, flags=re.I)
    ]


def count_nearby_vehicles(prompt: Any) -> tuple[int, bool]:
    text = str(prompt or "")
    matches = {match.group(0).strip() for pattern in NEARBY_VEHICLE_PATTERNS for match in re.finditer(pattern, text, flags=re.I)}
    section_observed = bool(re.search(r"Nearby vehicles description", text, flags=re.I))
    explicit_none = bool(re.search(r"There are no other vehicles driving near you", text, flags=re.I))
    return len(matches), section_observed or explicit_none


def prompt_snapshot(prompt: Any, navigation: Any = "") -> dict[str, Any]:
    text = str(prompt or "")
    nav = str(navigation or "")
    lane_count, lane_index = parse_lane(text)
    speeds = re.findall(
        r"(?:speed is|speed of your vehicle is)\s*(-?\d+(?:\.\d+)?)\s*m/s",
        text,
        flags=re.I,
    )
    nearby_count, nearby_observed = count_nearby_vehicles(text)
    distances = [
        abs(float(value))
        for value in re.findall(
            r"distance between you and vehicle[^.]*?is\s+"
            r"(-?\d+(?:\.\d+)?)\s*m",
            text,
            flags=re.I,
        )
    ]
    conflict_times = extract_conflict_times(text)
    combined = f"{nav}\n{text}".lower()
    return {
        "lane_count": lane_count,
        "lane_count_observed": math.isfinite(lane_count),
        "lane_index": lane_index,
        "lane_index_observed": math.isfinite(lane_index),
        "speed_from_prompt": safe_float(speeds[0]) if speeds else math.nan,
        "speed_observed": bool(speeds),
        "nearby_count": nearby_count if nearby_observed else math.nan,
        "nearby_count_observed": nearby_observed,
        "minimum_distance": min(distances, default=math.nan),
        "minimum_distance_observed": bool(distances),
        "minimum_conflict_time": min(conflict_times, default=math.nan),
        "minimum_conflict_time_observed": bool(conflict_times),
        "junction": bool(re.search(r"junction|intersection|merge|roundabout", combined)),
        "nav_left": bool(re.search(r"left|action[_ ]?id:\s*3", nav, flags=re.I)),
        "nav_right": bool(re.search(r"right|action[_ ]?id:\s*4", nav, flags=re.I)),
        "nav_straight": bool(re.search(r"straight|continue", nav, flags=re.I)),
    }
