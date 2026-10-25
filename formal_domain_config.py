"""Shared driving scene, route, weather, and traffic configuration."""

from __future__ import annotations

import itertools
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCENE_CONFIG_ID = "driving_scene_config"
ACTION_IDS = [1, 2, 3, 4, 8]

WEATHER_PRESETS: dict[str, dict[str, int]] = {
    "clear": {
        "rain": 0,
        "fog": 0,
        "wind": 0,
        "wet_road": 0,
    },
    "slightly_inclement": {
        "rain": 25,
        "fog": 20,
        "wind": 10,
        "wet_road": 30,
    },
    "moderately_inclement": {
        "rain": 50,
        "fog": 45,
        "wind": 25,
        "wet_road": 60,
    },
    "severely_inclement": {
        "rain": 80,
        "fog": 70,
        "wind": 50,
        "wet_road": 90,
    },
}

WEATHER_LEVEL_TO_ID = {
    "clear": "clear",
    "slightly": "slightly_inclement",
    "moderate": "moderately_inclement",
    "severe": "severely_inclement",
}
WEATHER_ID_TO_LEVEL = {value: key for key, value in WEATHER_LEVEL_TO_ID.items()}

DENSITY_PRESETS = {
    "1x": 1.0,
    "2x": 2.0,
    "3x": 3.0,
}

DOMAIN_WEATHER_LEVELS = ["clear", "slightly", "moderate", "severe"]
DOMAIN_DENSITY_IDS = ["1x", "2x", "3x"]

ID_TRAIN_VISIBLE_DOMAINS = [
    "clear_1x",
    "clear_2x",
    "slightly_1x",
    "slightly_2x",
    "moderate_1x",
    "moderate_2x",
]
OOD_WEATHER_DOMAINS = [
    "severe_1x",
    "severe_2x",
]
OOD_DENSITY_DOMAINS = [
    "clear_3x",
    "slightly_3x",
    "moderate_3x",
]
OOD_JOINT_DOMAINS = [
    "severe_3x",
]
ALL_FORMAL_DOMAINS = (
    ID_TRAIN_VISIBLE_DOMAINS
    + OOD_WEATHER_DOMAINS
    + OOD_DENSITY_DOMAINS
    + OOD_JOINT_DOMAINS
)
FORBIDDEN_TRAINING_DOMAINS = sorted(
    set(OOD_WEATHER_DOMAINS + OOD_DENSITY_DOMAINS + OOD_JOINT_DOMAINS)
)


@dataclass(frozen=True)
class RouteEntry:
    map_id: str
    route_id: str
    route_variant_id: str
    ego_id: str
    sumo_net_file: str
    sumo_cfg_file: str
    base_sumo_rou_file: str
    base_vtypes_file: str


ROUTE_CATALOG: list[RouteEntry] = [
    RouteEntry(
        map_id="Town05",
        route_id="town05_default",
        route_variant_id="default",
        ego_id="50",
        sumo_net_file="networkFiles/CarlaTown05/Town05.net.xml",
        sumo_cfg_file="networkFiles/CarlaTown05/Town05.sumocfg",
        base_sumo_rou_file="networkFiles/CarlaTown05/Town05.rou.xml",
        base_vtypes_file="networkFiles/CarlaTown05/carlavtypes.rou.xml",
    )
]


def repo_root() -> Path:
    return Path(__file__).resolve().parent


def domain_id_from_parts(weather_level: str, traffic_density_id: str) -> str:
    if weather_level not in WEATHER_LEVEL_TO_ID:
        raise ValueError(f"Unknown weather level {weather_level!r}")
    if traffic_density_id not in DENSITY_PRESETS:
        raise ValueError(f"Unknown traffic density id {traffic_density_id!r}")
    return f"{weather_level}_{traffic_density_id}"


def domain_metadata(domain_id: str) -> dict[str, Any]:
    if domain_id not in ALL_FORMAL_DOMAINS:
        raise ValueError(
            f"Unknown formal domain {domain_id!r}. Supported domains: {','.join(ALL_FORMAL_DOMAINS)}"
        )
    weather_level, traffic_density_id = domain_id.rsplit("_", 1)
    weather_id = WEATHER_LEVEL_TO_ID[weather_level]
    return {
        "domain_id": domain_id,
        "weather_id": weather_id,
        "weather_level": weather_level,
        "weather_params": dict(WEATHER_PRESETS[weather_id]),
        "weather_prompt_effective": True,
        "weather_sim_effective": False,
        "weather_effective_type": "prompt_observed_numeric_0_100",
        "traffic_density_id": traffic_density_id,
        "traffic_density_scale": DENSITY_PRESETS[traffic_density_id],
        "traffic_density_effective": traffic_density_id != "1x",
        "traffic_density_method": "randomTrips_period_scaling_route_demand",
    }


def parse_domains(raw: str) -> list[str]:
    value = (raw or "").strip()
    if value == "all":
        return list(ALL_FORMAL_DOMAINS)
    if value == "id":
        return list(ID_TRAIN_VISIBLE_DOMAINS)
    if value in {"validation", "staged_validation", "validation_id"}:
        return ["clear_1x", "clear_2x"]
    if value == "ood_weather":
        return list(OOD_WEATHER_DOMAINS)
    if value == "ood_density":
        return list(OOD_DENSITY_DOMAINS)
    if value == "ood_joint":
        return list(OOD_JOINT_DOMAINS)
    domains = [part.strip() for part in value.split(",") if part.strip()]
    unknown = [domain for domain in domains if domain not in ALL_FORMAL_DOMAINS]
    if unknown:
        raise ValueError(f"Unknown formal domains: {','.join(unknown)}")
    return domains


def route_entries_for_maps(map_ids: list[str] | None = None) -> list[RouteEntry]:
    wanted = set(map_ids or [])
    return [
        entry
        for entry in ROUTE_CATALOG
        if not wanted or entry.map_id in wanted
    ]


def find_route_entry(map_id: str, route_id: str | None = None, ego_id: str | None = None) -> RouteEntry:
    candidates = [
        entry
        for entry in ROUTE_CATALOG
        if entry.map_id == map_id
        and (route_id is None or entry.route_id == route_id)
        and (ego_id is None or entry.ego_id == ego_id)
    ]
    if not candidates:
        raise ValueError(
            f"No route catalog entry for map_id={map_id!r}, route_id={route_id!r}, ego_id={ego_id!r}"
        )
    return candidates[0]


def route_entry_to_paths(entry: RouteEntry, root: Path | None = None) -> dict[str, Path]:
    base = root or repo_root()
    return {
        "sumo_net_file": (base / entry.sumo_net_file).resolve(),
        "sumo_cfg_file": (base / entry.sumo_cfg_file).resolve(),
        "base_sumo_rou_file": (base / entry.base_sumo_rou_file).resolve(),
        "base_vtypes_file": (base / entry.base_vtypes_file).resolve(),
    }


def validate_route_entry(entry: RouteEntry, root: Path | None = None) -> dict[str, Any]:
    paths = route_entry_to_paths(entry, root)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        return {
            "available": False,
            "missing_files": missing,
            "entry": entry.__dict__,
        }
    ego_present = vehicle_exists(paths["base_sumo_rou_file"], entry.ego_id)
    return {
        "available": bool(ego_present),
        "missing_files": [],
        "ego_present": bool(ego_present),
        "entry": entry.__dict__,
    }


def require_route_entry_available(entry: RouteEntry, root: Path | None = None) -> dict[str, Path]:
    status = validate_route_entry(entry, root)
    if not status["available"]:
        reason = json.dumps(status, indent=2)
        raise FileNotFoundError(f"Route entry is not available:\n{reason}")
    return route_entry_to_paths(entry, root)


def route_depart(vehicle: ET.Element) -> float:
    return float(vehicle.get("depart", "0"))


def vehicle_exists(rou_file: Path, ego_id: str) -> bool:
    if not rou_file.exists():
        return False
    root = ET.parse(rou_file).getroot()
    return any(child.tag == "vehicle" and child.get("id") == ego_id for child in root)


def vehicle_departures(rou_file: Path) -> list[tuple[str, float]]:
    root = ET.parse(rou_file).getroot()
    return [
        (child.get("id", ""), route_depart(child))
        for child in root
        if child.tag == "vehicle"
    ]


def sorted_departures(rou_file: Path) -> bool:
    deps = [depart for _, depart in vehicle_departures(rou_file)]
    return deps == sorted(deps)


def positive_depart_gaps(rou_file: Path) -> list[float]:
    departures = sorted(depart for _, depart in vehicle_departures(rou_file))
    return [
        round(b - a, 6)
        for a, b in zip(departures, departures[1:])
        if b > a
    ]


def infer_regular_period_seconds(rou_file: Path) -> float:
    """Infer the regular randomTrips period from the base route file."""
    gaps = positive_depart_gaps(rou_file)
    if not gaps:
        return 2.0
    gaps = sorted(gaps)
    return float(gaps[len(gaps) // 2])


def clone_vehicle_with_new_id_and_depart(vehicle: ET.Element, new_id: str, new_depart: float) -> ET.Element:
    clone = ET.fromstring(ET.tostring(vehicle, encoding="unicode"))
    clone.set("id", new_id)
    clone.set("depart", f"{new_depart:.4f}")
    return clone


def make_density_route_file(
    base_rou_file: Path,
    output_rou_file: Path,
    ego_id: str,
    density_scale: float,
    validate_against_scale: float | None = None,
) -> dict[str, Any]:
    """Generate a density-scaled SUMO route file by scaling the randomTrips period.

    The Town05/Town06 base route files are randomTrips-style demand streams with
    a regular period near 2.0 seconds. Higher density is represented by a lower
    effective period over the full route-file horizon:
      - 1x: period ~= 2.0 seconds
      - 2x: period ~= 1.0 seconds
      - 3x: period ~= 0.6667 seconds

    The ego vehicle is preserved exactly once with its original route, original
    depart time, and original attributes. Only background demand is repeated.
    """
    if density_scale <= 0:
        raise ValueError(f"density_scale must be positive, got {density_scale}")
    if abs(density_scale - round(density_scale)) > 1e-9:
        raise ValueError(
            "randomTrips-style deterministic period scaling expects integer "
            f"density scales such as 1, 2, or 3; got {density_scale}"
        )
    scale_int = int(round(density_scale))

    if not base_rou_file.exists():
        raise FileNotFoundError(f"Base route file does not exist: {base_rou_file}")

    ET.register_namespace("xsi", "http://www.w3.org/2001/XMLSchema-instance")
    tree = ET.parse(base_rou_file)
    root = tree.getroot()
    vehicles = [child for child in list(root) if child.tag == "vehicle"]
    if not vehicles:
        raise ValueError(f"No <vehicle> entries found in {base_rou_file}")

    ego_vehicle = next((vehicle for vehicle in vehicles if vehicle.get("id") == ego_id), None)
    if ego_vehicle is None:
        raise ValueError(f"Ego vehicle id={ego_id!r} not found in {base_rou_file}")

    base_period = infer_regular_period_seconds(base_rou_file)
    effective_period = base_period / density_scale

    base_departs = [route_depart(vehicle) for vehicle in vehicles]
    min_depart = min(base_departs)
    max_depart = max(base_departs)
    horizon = (max_depart - min_depart) + base_period

    generated: list[ET.Element] = []
    for vehicle in vehicles:
        vid = vehicle.get("id")
        if vid == ego_id:
            continue
        depart = route_depart(vehicle)
        relative_depart = depart - min_depart
        for copy_idx in range(scale_int):
            new_depart = min_depart + (relative_depart + copy_idx * horizon) / density_scale
            new_id = vid if scale_int == 1 and copy_idx == 0 else f"{vid}_d{scale_int}x_{copy_idx}"
            generated.append(clone_vehicle_with_new_id_and_depart(vehicle, new_id, new_depart))

    generated.append(ET.fromstring(ET.tostring(ego_vehicle, encoding="unicode")))

    for vehicle in vehicles:
        root.remove(vehicle)

    generated.sort(key=lambda vehicle: (route_depart(vehicle), vehicle.get("id", "")))
    for vehicle in generated:
        root.append(vehicle)

    output_rou_file.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_rou_file, encoding="utf-8", xml_declaration=True)

    if not vehicle_exists(output_rou_file, ego_id):
        raise ValueError(f"Generated route file lost ego vehicle id={ego_id!r}: {output_rou_file}")
    if not sorted_departures(output_rou_file):
        raise ValueError(f"Generated route file depart times are not sorted: {output_rou_file}")

    generated_departs = vehicle_departures(output_rou_file)
    generated_gaps = positive_depart_gaps(output_rou_file)
    generated_period_median = (
        sorted(generated_gaps)[len(generated_gaps) // 2]
        if generated_gaps
        else ""
    )

    compare_scale = validate_against_scale
    compare_expected_period = ""
    if compare_scale is not None:
        compare_expected_period = base_period / float(compare_scale)
        if effective_period >= compare_expected_period:
            raise ValueError(
                f"Density validation failed: scale {density_scale:g} effective period "
                f"{effective_period:g} must be smaller than compare scale {compare_scale:g} "
                f"effective period {compare_expected_period:g}."
            )

    return {
        "density_generated_sumo_rou_file": str(output_rou_file.resolve()),
        "generated_sumo_rou_file": str(output_rou_file.resolve()),
        "ego_preserved": True,
        "ego_depart_preserved": True,
        "depart_times_sorted": True,
        "traffic_density_method": "randomTrips_period_scaling_route_demand",
        "density_regular_period_seconds": base_period,
        "density_effective_period_seconds": effective_period,
        "density_scale_integer": scale_int,
        "density_base_vehicle_count": len(vehicles),
        "density_generated_vehicle_count": len(generated_departs),
        "density_generated_median_depart_gap_seconds": generated_period_median,
        "density_compare_scale": compare_scale if compare_scale is not None else "",
        "density_compare_expected_period_seconds": compare_expected_period,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_randomtrips_background_path(
    background_dir: Path,
    map_id: str,
    traffic_density_id: str,
    traffic_route_seed: int,
) -> Path:
    return background_dir / (
        f"{map_id}_background_{traffic_density_id}_seed{traffic_route_seed}.rou.xml"
    )


def make_frozen_randomtrips_density_route_file(
    *,
    base_rou_file: Path,
    background_rou_file: Path,
    output_rou_file: Path,
    map_id: str,
    net_file: Path,
    vtypes_file: Path,
    ego_id: str,
    traffic_density_id: str,
    density_scale: float,
    traffic_route_seed: int,
    background_manifest_file: Path | None = None,
    begin_seconds: float = 0.0,
    end_seconds: float = 3600.0,
) -> dict[str, Any]:
    """Inject the original ego into a frozen randomTrips background.

    The background must have been generated independently at period
    ``2.0 / density_scale`` and must not contain the ego ID. The ego XML
    element is copied byte-for-byte at the element level from the original
    route file, preserving route, type, departure time, and departure attrs.
    """
    if density_scale <= 0:
        raise ValueError(f"density_scale must be positive, got {density_scale}")
    if traffic_density_id not in DENSITY_PRESETS:
        raise ValueError(f"Unknown traffic density id {traffic_density_id!r}")
    expected_scale = float(DENSITY_PRESETS[traffic_density_id])
    if not math.isclose(
        float(density_scale),
        expected_scale,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "Density ID/scale mismatch: "
            f"id={traffic_density_id}, scale={density_scale}, "
            f"expected_scale={expected_scale}"
        )
    if end_seconds <= begin_seconds:
        raise ValueError("randomTrips end_seconds must exceed begin_seconds")
    required_files = {
        "base route": base_rou_file,
        "frozen background": background_rou_file,
        "network": net_file,
        "vehicle types": vtypes_file,
    }
    for label, path in required_files.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {label} file: {path}")

    manifest_file = (
        background_manifest_file
        if background_manifest_file is not None
        else background_rou_file.parent / "density_background_manifest.json"
    )
    if not manifest_file.is_file():
        raise FileNotFoundError(
            f"Frozen background manifest does not exist: {manifest_file}"
        )
    manifest = json.loads(manifest_file.read_text())
    if manifest.get("schema_version") != "frozen_randomtrips_density_v1":
        raise ValueError("Unsupported frozen background manifest schema")
    if str(manifest.get("map_id")) != str(map_id):
        raise ValueError(
            f"Frozen manifest map mismatch: {manifest.get('map_id')} != {map_id}"
        )
    if int(manifest.get("traffic_route_seed", -1)) != int(traffic_route_seed):
        raise ValueError("Frozen manifest traffic route seed mismatch")
    if not math.isclose(
        float(manifest.get("begin_seconds", float("nan"))),
        float(begin_seconds),
        rel_tol=0.0,
        abs_tol=1e-9,
    ) or not math.isclose(
        float(manifest.get("end_seconds", float("nan"))),
        float(end_seconds),
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("Frozen manifest begin/end horizon mismatch")
    if manifest.get("net_file_sha256") != sha256_file(net_file):
        raise ValueError("Frozen manifest network hash mismatch")
    if manifest.get("vtypes_file_sha256") != sha256_file(vtypes_file):
        raise ValueError("Frozen manifest vehicle-types hash mismatch")
    script_path = Path(str(manifest.get("randomTrips_script", "")))
    if (
        not script_path.is_file()
        or manifest.get("randomTrips_script_sha256") != sha256_file(script_path)
    ):
        raise ValueError("Frozen manifest randomTrips script provenance mismatch")

    background_records = [
        row
        for row in manifest.get("backgrounds", [])
        if str(row.get("density_id")) == str(traffic_density_id)
    ]
    if len(background_records) != 1:
        raise ValueError(
            f"Expected one manifest entry for density {traffic_density_id}; "
            f"found {len(background_records)}"
        )
    background_record = background_records[0]
    manifest_route_name = Path(
        str(background_record.get("route_file", ""))
    ).name
    if manifest_route_name != background_rou_file.name:
        raise ValueError("Frozen manifest route filename mismatch")
    background_sha256 = sha256_file(background_rou_file)
    if background_record.get("route_sha256") != background_sha256:
        raise ValueError("Frozen background route hash mismatch")

    base_root = ET.parse(base_rou_file).getroot()
    ego_matches = [
        child
        for child in base_root
        if child.tag == "vehicle" and child.get("id") == str(ego_id)
    ]
    if len(ego_matches) != 1:
        raise ValueError(
            f"Expected exactly one ego id={ego_id!r} in {base_rou_file}; "
            f"found {len(ego_matches)}"
        )
    ego_vehicle = ET.fromstring(ET.tostring(ego_matches[0], encoding="unicode"))
    ego_depart = route_depart(ego_vehicle)
    ego_route = ego_vehicle.find("route")
    if ego_route is None or not str(ego_route.get("edges", "")).strip():
        raise ValueError(f"Ego id={ego_id!r} has no inline route in {base_rou_file}")

    ET.register_namespace("xsi", "http://www.w3.org/2001/XMLSchema-instance")
    background_tree = ET.parse(background_rou_file)
    background_root = background_tree.getroot()
    background_vehicles = [
        child for child in list(background_root) if child.tag == "vehicle"
    ]
    if not background_vehicles:
        raise ValueError(f"No background vehicles in {background_rou_file}")

    background_ids = [str(vehicle.get("id", "")) for vehicle in background_vehicles]
    if any(not vehicle_id for vehicle_id in background_ids):
        raise ValueError(f"Background contains a vehicle without an ID: {background_rou_file}")
    if len(background_ids) != len(set(background_ids)):
        raise ValueError(f"Background vehicle IDs are not unique: {background_rou_file}")
    if str(ego_id) in set(background_ids):
        raise ValueError(
            f"Frozen background must not contain ego id={ego_id!r}: {background_rou_file}"
        )
    missing_types = [
        vehicle_id
        for vehicle_id, vehicle in zip(background_ids, background_vehicles)
        if not str(vehicle.get("type", "")).strip()
    ]
    if missing_types:
        raise ValueError(
            "Frozen background vehicles require deterministic type assignments; "
            f"missing for {missing_types[:5]}"
        )

    expected_period = 2.0 / expected_scale
    if not math.isclose(
        float(background_record.get("period_seconds", float("nan"))),
        expected_period,
        rel_tol=0.0,
        abs_tol=max(1e-9, expected_period * 1e-9),
    ):
        raise ValueError("Frozen manifest period mismatch")
    background_departs = sorted(route_depart(vehicle) for vehicle in background_vehicles)
    gaps = [
        round(b - a, 6)
        for a, b in zip(background_departs, background_departs[1:])
        if b > a
    ]
    if not gaps:
        raise ValueError(f"Cannot infer departure period from {background_rou_file}")
    median_gap = float(sorted(gaps)[len(gaps) // 2])
    schedule_errors = [
        abs(depart - (begin_seconds + index * expected_period))
        for index, depart in enumerate(background_departs)
    ]
    max_schedule_error = max(schedule_errors)
    schedule_tolerance = 0.0051
    if max_schedule_error > schedule_tolerance:
        raise ValueError(
            "Frozen background departure schedule mismatch: "
            f"density={traffic_density_id}, max_error={max_schedule_error}, "
            f"tolerance={schedule_tolerance}"
        )
    realized_mean_gap = (
        background_departs[-1] - background_departs[0]
    ) / (len(background_departs) - 1)

    expected_count = int(math.ceil((end_seconds - begin_seconds) / expected_period))
    if abs(len(background_vehicles) - expected_count) > 1:
        raise ValueError(
            "Frozen background count does not match the configured period/horizon: "
            f"actual={len(background_vehicles)}, expected~={expected_count}"
        )
    if int(background_record.get("background_vehicle_count", -1)) != len(
        background_vehicles
    ):
        raise ValueError("Frozen manifest background vehicle count mismatch")

    background_id_prefix = f"bg_{traffic_route_seed}_"
    invalid_background_ids = [
        vehicle_id
        for vehicle_id in background_ids
        if not vehicle_id.startswith(background_id_prefix)
    ]
    if invalid_background_ids:
        raise ValueError(
            "Frozen background ID/seed mismatch; invalid IDs include "
            f"{invalid_background_ids[:5]}"
        )

    try:
        reserved_background_id = (
            f"bg_{traffic_route_seed}_{int(str(ego_id)):05d}"
        )
    except ValueError as exc:
        raise ValueError(
            f"Numeric ego ID required for frozen trip reservation: {ego_id!r}"
        ) from exc
    reserved_matches = [
        vehicle
        for vehicle in background_vehicles
        if vehicle.get("id") == reserved_background_id
    ]
    if len(reserved_matches) != 1:
        raise ValueError(
            "Expected exactly one reserved randomTrips counterpart for ego "
            f"{ego_id!r}: id={reserved_background_id!r}, "
            f"found={len(reserved_matches)}"
        )
    background_count_before_reservation = len(background_vehicles)
    background_vehicles.remove(reserved_matches[0])
    background_count_after_reservation = len(background_vehicles)

    non_vehicle_children = [
        child for child in list(background_root) if child.tag != "vehicle"
    ]
    for child in list(background_root):
        background_root.remove(child)
    for child in non_vehicle_children:
        background_root.append(child)

    combined_vehicles = list(background_vehicles) + [ego_vehicle]
    combined_vehicles.sort(
        key=lambda vehicle: (route_depart(vehicle), str(vehicle.get("id", "")))
    )
    for vehicle in combined_vehicles:
        background_root.append(vehicle)

    output_rou_file.parent.mkdir(parents=True, exist_ok=True)
    background_tree.write(output_rou_file, encoding="utf-8", xml_declaration=True)

    combined_root = ET.parse(output_rou_file).getroot()
    written_ego = [
        child
        for child in combined_root
        if child.tag == "vehicle" and child.get("id") == str(ego_id)
    ]
    if len(written_ego) != 1:
        raise ValueError(
            f"Combined route must contain exactly one ego id={ego_id!r}; "
            f"found {len(written_ego)}"
        )
    written_route = written_ego[0].find("route")
    if (
        route_depart(written_ego[0]) != ego_depart
        or written_ego[0].get("type") != ego_vehicle.get("type")
        or written_route is None
        or written_route.get("edges") != ego_route.get("edges")
    ):
        raise ValueError("Injected ego route/depart/type changed during combination")
    written_ids = [
        str(child.get("id", ""))
        for child in combined_root
        if child.tag == "vehicle"
    ]
    if reserved_background_id in written_ids:
        raise ValueError("Reserved ego counterpart remains in combined route")
    if len(combined_vehicles) != background_count_before_reservation:
        raise ValueError(
            "Combined count must equal frozen background count after replacing "
            "one reserved trip with the original ego"
        )
    if not sorted_departures(output_rou_file):
        raise ValueError(f"Combined route departures are not sorted: {output_rou_file}")

    return {
        "density_generated_sumo_rou_file": str(output_rou_file.resolve()),
        "generated_sumo_rou_file": str(output_rou_file.resolve()),
        "traffic_density_method": "frozen_randomTrips_period_scaling_ego_injected",
        "traffic_route_seed": int(traffic_route_seed),
        "randomTrips_period_seconds": expected_period,
        "randomTrips_begin_seconds": float(begin_seconds),
        "randomTrips_end_seconds": float(end_seconds),
        "frozen_background_manifest_file": str(manifest_file.resolve()),
        "frozen_background_manifest_sha256": sha256_file(manifest_file),
        "frozen_background_route_file": str(background_rou_file.resolve()),
        "frozen_background_route_sha256": background_sha256,
        "density_generated_route_sha256": sha256_file(output_rou_file),
        "reserved_background_vehicle_id": reserved_background_id,
        "background_vehicle_removed_for_ego": True,
        "background_vehicle_count_before_reservation": (
            background_count_before_reservation
        ),
        "background_vehicle_count_after_reservation": (
            background_count_after_reservation
        ),
        "background_vehicle_count": background_count_after_reservation,
        "density_generated_vehicle_count": len(combined_vehicles),
        "density_generated_median_depart_gap_seconds": median_gap,
        "density_generated_effective_period_seconds": realized_mean_gap,
        "density_generated_max_depart_schedule_error_seconds": (
            max_schedule_error
        ),
        "ego_injected_separately": True,
        "ego_depart": ego_depart,
        "ego_depart_preserved_across_density": True,
        "ego_route_edges": ego_route.get("edges", ""),
        "ego_type": ego_vehicle.get("type", ""),
        "ego_preserved": True,
        "ego_depart_preserved": True,
        "depart_times_sorted": True,
    }


def subset_masks(context_size: int = 5) -> list[tuple[int, ...]]:
    return [
        combo
        for size in range(context_size + 1)
        for combo in itertools.combinations(range(context_size), size)
    ]


def subset_mask_string(indices: tuple[int, ...], context_size: int = 5) -> str:
    kept = set(indices)
    return "".join("1" if index in kept else "0" for index in range(context_size))


def split_group_for_domain(domain_id: str) -> str:
    if domain_id in ID_TRAIN_VISIBLE_DOMAINS:
        return "id"
    if domain_id in OOD_WEATHER_DOMAINS:
        return "ood_weather"
    if domain_id in OOD_DENSITY_DOMAINS:
        return "ood_density"
    if domain_id in OOD_JOINT_DOMAINS:
        return "ood_joint"
    raise ValueError(f"Unknown formal domain: {domain_id}")


def validate_domain_metadata(record: dict[str, Any]) -> None:
    domain_id = str(record.get("domain_id", ""))
    expected = domain_metadata(domain_id)
    for key in ["weather_id", "weather_level", "traffic_density_id"]:
        if str(record.get(key, "")) != str(expected[key]):
            raise ValueError(
                f"Domain metadata mismatch for state_id={record.get('state_id')}: "
                f"{key}={record.get(key)!r}, expected {expected[key]!r}"
            )
    scale = float(record.get("traffic_density_scale", math.nan))
    if not math.isfinite(scale) or abs(scale - float(expected["traffic_density_scale"])) > 1e-9:
        raise ValueError(
            f"Density scale mismatch for state_id={record.get('state_id')}: "
            f"{scale}, expected {expected['traffic_density_scale']}"
        )
