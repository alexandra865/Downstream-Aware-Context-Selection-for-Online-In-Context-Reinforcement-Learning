from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from trafficManager.common.vehicle import Behaviour
from utils.roadgraph import JunctionLane, NormalLane, RoadGraph
from utils.trajectory import RecCollide, Rectangle, State, Trajectory


VALID_ACTION_IDS = {
    int(Behaviour.AC),
    int(Behaviour.DC),
    int(Behaviour.LCL),
    int(Behaviour.LCR),
    int(Behaviour.IDLE),
}


@dataclass(frozen=True)
class ActionSafety:
    action_id: int
    trajectory_generated: bool
    collision_free: bool
    first_collision_t: Optional[float]
    min_clearance: float
    conflict_vehicle_id: Optional[str]
    generation_error: str = ""
    prediction_truncated: bool = False
    prediction_coverage_by_vehicle: dict[str, dict[str, Any]] = field(
        default_factory=dict
    )
    traffic_control_feasible: bool = True
    traffic_control_violation: str = ""
    red_light_active: bool = False
    red_light_stop_line_distance: Optional[float] = None
    red_light_crossing_t: Optional[float] = None
    traffic_control_commitment_horizon: Optional[float] = None


class SafetyShieldEvaluationError(RuntimeError):
    """Raised when no candidate has a complete geometric evaluation."""

    def __init__(self, assessments: dict[int, ActionSafety]) -> None:
        self.assessments = assessments
        errors = {
            str(action_id): {
                "generation_error": assessment.generation_error,
                "prediction_truncated": assessment.prediction_truncated,
            }
            for action_id, assessment in assessments.items()
        }
        super().__init__(
            "Safety shield has no completely evaluated candidate trajectory: "
            + repr(errors)
        )


class SafetyShieldNoAdmissibleActionError(SafetyShieldEvaluationError):
    """Raised when evaluated actions cannot satisfy all hard constraints."""

    def __init__(self, assessments: dict[int, ActionSafety]) -> None:
        self.assessments = assessments
        details = {
            str(action_id): {
                "collision_free": assessment.collision_free,
                "traffic_control_feasible": (
                    assessment.traffic_control_feasible
                ),
                "traffic_control_violation": (
                    assessment.traffic_control_violation
                ),
                "red_light_crossing_t": assessment.red_light_crossing_t,
            }
            for action_id, assessment in assessments.items()
        }
        RuntimeError.__init__(
            self,
            "Safety shield found no action satisfying both collision and "
            "traffic-control constraints: " + repr(details),
        )


@dataclass
class ShieldResult:
    base_actions: list[int]
    safe_actions: list[int]
    least_risk_action: int
    no_safe_action: bool
    assessments: dict[int, ActionSafety] = field(default_factory=dict)

    @property
    def shield_triggered(self) -> bool:
        return self.safe_actions != self.base_actions

    @property
    def forced_single_action(self) -> bool:
        return len(self.safe_actions) == 1

    @property
    def any_prediction_truncated(self) -> bool:
        return any(
            assessment.prediction_truncated
            for assessment in self.assessments.values()
        )

    def policy_actions(self) -> list[int]:
        if self.safe_actions:
            return list(self.safe_actions)
        return [self.least_risk_action]

    def resolve_raw_action(self, raw_action_id: int) -> tuple[int, bool]:
        raw_action_id = int(raw_action_id)
        if raw_action_id in self.policy_actions():
            return raw_action_id, False
        return int(self.least_risk_action), True

    def audit_fields(self) -> dict[str, Any]:
        def finite_or_none(value: float) -> Optional[float]:
            return float(value) if math.isfinite(value) else None

        return {
            "base_action_ids": list(self.base_actions),
            "safe_action_ids": list(self.safe_actions),
            "policy_action_ids": self.policy_actions(),
            "shield_triggered": self.shield_triggered,
            "forced_single_action": self.forced_single_action,
            "no_safe_action": self.no_safe_action,
            "least_risk_action": self.least_risk_action,
            "trajectory_generated_by_action": {
                str(action_id): assessment.trajectory_generated
                for action_id, assessment in self.assessments.items()
            },
            "collision_free_by_action": {
                str(action_id): assessment.collision_free
                for action_id, assessment in self.assessments.items()
            },
            "first_collision_t_by_action": {
                str(action_id): assessment.first_collision_t
                for action_id, assessment in self.assessments.items()
            },
            "min_clearance_by_action": {
                str(action_id): finite_or_none(assessment.min_clearance)
                for action_id, assessment in self.assessments.items()
            },
            "conflict_vehicle_by_action": {
                str(action_id): assessment.conflict_vehicle_id
                for action_id, assessment in self.assessments.items()
            },
            "trajectory_generation_error_by_action": {
                str(action_id): assessment.generation_error
                for action_id, assessment in self.assessments.items()
            },
            "prediction_truncated_by_action": {
                str(action_id): assessment.prediction_truncated
                for action_id, assessment in self.assessments.items()
            },
            "prediction_coverage_by_action": {
                str(action_id): assessment.prediction_coverage_by_vehicle
                for action_id, assessment in self.assessments.items()
            },
            "traffic_control_feasible_by_action": {
                str(action_id): assessment.traffic_control_feasible
                for action_id, assessment in self.assessments.items()
            },
            "traffic_control_violation_by_action": {
                str(action_id): assessment.traffic_control_violation
                for action_id, assessment in self.assessments.items()
            },
            "red_light_active_by_action": {
                str(action_id): assessment.red_light_active
                for action_id, assessment in self.assessments.items()
            },
            "red_light_stop_line_distance_by_action": {
                str(action_id): assessment.red_light_stop_line_distance
                for action_id, assessment in self.assessments.items()
            },
            "red_light_crossing_t_by_action": {
                str(action_id): assessment.red_light_crossing_t
                for action_id, assessment in self.assessments.items()
            },
            "traffic_control_commitment_horizon_by_action": {
                str(action_id): (
                    assessment.traffic_control_commitment_horizon
                )
                for action_id, assessment in self.assessments.items()
            },
        }


def _trajectory_states(trajectory: Trajectory | Iterable[State]) -> list[State]:
    if isinstance(trajectory, Trajectory):
        return list(trajectory.states)
    return list(trajectory)


def preview_ego_trajectory(
    planner,
    sim_time: float,
    roadgraph: RoadGraph,
    vehicles_info: dict,
    action_id: int,
) -> Trajectory:
    """Preview one action without mutating the live TrafficManager."""
    action_id = int(action_id)
    if action_id not in VALID_ACTION_IDS:
        raise ValueError(f"Unsupported action id for safety preview: {action_id}")

    live_time_step = planner.time_step
    live_lastseen = planner.lastseen_vehicles
    live_last_decision_time = planner.last_decision_time
    live_mul_decisions = planner.mul_decisions

    shadow = copy.copy(planner)
    shadow.lastseen_vehicles = copy.deepcopy(planner.lastseen_vehicles)
    shadow.time_step = planner.time_step
    shadow.last_decision_time = planner.last_decision_time
    shadow.mul_decisions = copy.deepcopy(planner.mul_decisions)

    scene_copy = copy.deepcopy(vehicles_info)
    trajectories = shadow.plan(
        sim_time,
        roadgraph,
        scene_copy,
        Behaviour(action_id),
        other_plan=False,
    )

    if planner.time_step != live_time_step:
        raise RuntimeError("Safety preview mutated live planner.time_step")
    if planner.lastseen_vehicles is not live_lastseen:
        raise RuntimeError("Safety preview replaced live planner.lastseen_vehicles")
    if planner.last_decision_time != live_last_decision_time:
        raise RuntimeError("Safety preview mutated live planner.last_decision_time")
    if planner.mul_decisions is not live_mul_decisions:
        raise RuntimeError("Safety preview replaced live planner.mul_decisions")

    ego_id = vehicles_info["egoCar"]["id"]
    if ego_id not in trajectories:
        raise RuntimeError(
            f"Safety preview did not return ego trajectory for ego_id={ego_id}"
        )
    trajectory = trajectories[ego_id]
    if not _trajectory_states(trajectory):
        raise RuntimeError(
            f"Safety preview returned an empty ego trajectory for action={action_id}"
        )
    return trajectory


def _lane_length(lane) -> float:
    return float(lane.course_spline.s[-1])


def _active_red_light_constraint(
    roadgraph: RoadGraph,
    ego: dict,
) -> Optional[tuple[NormalLane, JunctionLane, float]]:
    lane_history = ego.get("laneIDQ", [])
    position_history = ego.get("lanePosQ", [])
    if not lane_history or not position_history:
        return None

    current_lane = roadgraph.get_lane_by_id(str(lane_history[-1]))
    if not isinstance(current_lane, NormalLane):
        return None

    available_lanes = {
        str(lane_id)
        for lane_id in ego.get("availableLanes", [])
    }
    next_lane = roadgraph.get_available_next_lane(
        current_lane.id,
        available_lanes,
    )
    if not isinstance(next_lane, JunctionLane):
        matching_junction_lanes = []
        for lane_id in sorted(available_lanes):
            lane = roadgraph.get_lane_by_id(lane_id)
            if (
                isinstance(lane, JunctionLane)
                and lane.last_lane_id == current_lane.id
            ):
                matching_junction_lanes.append(lane)
        if len(matching_junction_lanes) == 1:
            next_lane = matching_junction_lanes[0]

    if (
        not isinstance(next_lane, JunctionLane)
        or str(next_lane.currTlState or "") not in {"r", "R"}
    ):
        return None

    stop_line_distance = max(
        0.0,
        _lane_length(current_lane) - float(position_history[-1]),
    )
    return current_lane, next_lane, stop_line_distance


def _red_light_trajectory_crossing_time(
    trajectory: Trajectory | Iterable[State],
    roadgraph: RoadGraph,
    current_lane: NormalLane,
    red_junction_lane: JunctionLane,
    ego_length: float,
) -> Optional[float]:
    current_edge_id = (
        current_lane.affiliated_edge.id
        if current_lane.affiliated_edge is not None
        else current_lane.id.rsplit("_", 1)[0]
    )
    for state in _trajectory_states(trajectory):
        if not state.laneID:
            raise ValueError(
                "Cannot evaluate red-light compliance: trajectory state "
                "has no lane id"
            )
        state_lane = roadgraph.get_lane_by_id(state.laneID)
        if state_lane is None:
            raise ValueError(
                "Cannot evaluate red-light compliance: unknown trajectory "
                f"lane {state.laneID}"
            )
        if isinstance(state_lane, JunctionLane):
            if (
                state_lane.id == red_junction_lane.id
                or state_lane.last_lane_id == current_lane.id
            ):
                return float(state.t)
            continue
        if not isinstance(state_lane, NormalLane):
            continue

        state_edge_id = (
            state_lane.affiliated_edge.id
            if state_lane.affiliated_edge is not None
            else state_lane.id.rsplit("_", 1)[0]
        )
        if state_edge_id != current_edge_id:
            return float(state.t)

        front_bumper_s = float(state.s) + 0.5 * float(ego_length)
        if front_bumper_s >= _lane_length(state_lane):
            return float(state.t)
    return None


def _next_route_lane(
    roadgraph: RoadGraph,
    lane,
    available_lanes: set[str],
):
    if isinstance(lane, JunctionLane):
        # The exported available-lane set belongs to the approach edge and
        # commonly omits the normal lane after a junction. The junction lane's
        # next_lane_id is the authoritative route continuation.
        return roadgraph.get_lane_by_id(lane.next_lane_id)

    candidate = roadgraph.get_available_next_lane(lane.id, available_lanes)
    if candidate is not None:
        return candidate

    if isinstance(lane, NormalLane):
        candidate = roadgraph.get_next_lane(lane.id)
    else:
        candidate = None

    if candidate is None:
        return None
    if available_lanes and candidate.id not in available_lanes:
        return None
    return candidate


def _build_lane_chain(
    roadgraph: RoadGraph,
    current_lane,
    available_lanes: set[str],
    required_distance: float,
    start_s: float,
) -> list:
    lanes = [current_lane]
    covered = max(0.0, _lane_length(current_lane) - start_s)
    visited = {current_lane.id}
    while covered < required_distance and len(lanes) < 64:
        next_lane = _next_route_lane(
            roadgraph,
            lanes[-1],
            available_lanes,
        )
        if next_lane is None or next_lane.id in visited:
            break
        lanes.append(next_lane)
        visited.add(next_lane.id)
        covered += max(0.0, _lane_length(next_lane) - 0.1)
    return lanes


def _position_along_lane_chain(
    lanes: list,
    start_s: float,
    distance: float,
    lateral_offset: float,
) -> tuple[float, float, float, str, float]:
    lane_index = 0
    local_s = start_s + max(0.0, distance)
    while lane_index < len(lanes) - 1:
        transition_s = max(0.1, _lane_length(lanes[lane_index]) - 0.1)
        if local_s <= transition_s:
            break
        local_s -= transition_s
        lane_index += 1

    lane = lanes[lane_index]
    lane_length = _lane_length(lane)
    local_s = min(max(local_s, 0.0), max(0.0, lane_length - 1e-6))
    rx, ry = lane.course_spline.calc_position(local_s)
    yaw = float(lane.course_spline.calc_yaw(local_s))
    x = float(rx) - math.sin(yaw) * lateral_offset
    y = float(ry) + math.cos(yaw) * lateral_offset
    return x, y, yaw, lane.id, local_s


def _available_lane_chain_distance(lanes: list, start_s: float) -> float:
    if not lanes:
        return 0.0
    available = max(0.0, _lane_length(lanes[0]) - start_s)
    for lane in lanes[1:]:
        available += max(0.0, _lane_length(lane) - 0.1)
    return available


def predict_other_vehicle_trajectory(
    vehicle_info: dict,
    roadgraph: RoadGraph,
    horizon: float = 5.0,
    dt: float = 0.1,
    route_margin: float = 5.0,
) -> Trajectory:
    """Constant-velocity route-compatible prediction across a lane chain."""
    if horizon <= 0 or dt <= 0:
        raise ValueError("Prediction horizon and dt must be positive")

    lane_id = str(vehicle_info["laneIDQ"][-1])
    current_lane = roadgraph.get_lane_by_id(lane_id)
    if current_lane is None:
        raise ValueError(
            f"Vehicle {vehicle_info.get('id')} has unknown lane {lane_id}"
        )

    x0 = float(vehicle_info["xQ"][-1])
    y0 = float(vehicle_info["yQ"][-1])
    speed = max(0.0, float(vehicle_info["speedQ"][-1]))
    start_s, lateral_offset = current_lane.course_spline.cartesian_to_frenet1D(
        x0,
        y0,
    )
    start_s = float(start_s)
    lateral_offset = float(lateral_offset)
    available_lanes = {
        str(lane)
        for lane in vehicle_info.get("availableLanes", [])
    }
    required_prediction_distance = speed * horizon
    lanes = _build_lane_chain(
        roadgraph,
        current_lane,
        available_lanes,
        required_distance=required_prediction_distance + route_margin,
        start_s=start_s,
    )
    available_lane_chain_distance = _available_lane_chain_distance(
        lanes,
        start_s,
    )
    prediction_truncated = (
        available_lane_chain_distance + 1e-6
        < required_prediction_distance
    )

    trajectory = Trajectory()
    step_count = int(round(horizon / dt))
    for step in range(step_count + 1):
        t = round(step * dt, 10)
        distance = speed * t
        x, y, yaw, predicted_lane_id, local_s = _position_along_lane_chain(
            lanes,
            start_s,
            distance,
            lateral_offset,
        )
        trajectory.states.append(
            State(
                t=t,
                x=x,
                y=y,
                yaw=yaw,
                s=local_s,
                s_d=speed,
                vel=speed,
                acc=0.0,
                laneID=predicted_lane_id,
            )
        )
    trajectory.required_prediction_distance = required_prediction_distance
    trajectory.available_lane_chain_distance = available_lane_chain_distance
    trajectory.prediction_truncated = prediction_truncated
    return trajectory


def extend_trajectory_to_horizon(
    trajectory: Trajectory,
    vehicle_info: dict,
    roadgraph: RoadGraph,
    *,
    horizon: float,
    dt: float,
) -> Trajectory:
    """Extend a short planner trajectory with constant-velocity lane motion."""
    states = _trajectory_states(trajectory)
    if not states:
        raise ValueError("Cannot extend an empty trajectory")
    last = states[-1]
    last_t = float(last.t)
    if last_t >= horizon - dt * 0.5:
        trajectory.required_prediction_distance = 0.0
        trajectory.available_lane_chain_distance = None
        trajectory.prediction_truncated = False
        trajectory.prediction_coverage_source = "planner"
        return trajectory
    if not last.laneID:
        raise ValueError("Planner trajectory endpoint has no lane id")

    remaining = horizon - last_t
    continuation_info = {
        "id": vehicle_info["id"],
        "laneIDQ": [last.laneID],
        "xQ": [float(last.x)],
        "yQ": [float(last.y)],
        "yawQ": [float(last.yaw or 0.0)],
        "speedQ": [max(0.0, float(last.vel))],
        "availableLanes": vehicle_info.get("availableLanes", []),
        "length": vehicle_info["length"],
        "width": vehicle_info["width"],
    }
    continuation = predict_other_vehicle_trajectory(
        continuation_info,
        roadgraph,
        horizon=remaining,
        dt=dt,
    )
    for state in continuation.states[1:]:
        state.t = last_t + float(state.t)
        trajectory.states.append(state)
    trajectory.required_prediction_distance = getattr(
        continuation,
        "required_prediction_distance",
        None,
    )
    trajectory.available_lane_chain_distance = getattr(
        continuation,
        "available_lane_chain_distance",
        None,
    )
    trajectory.prediction_truncated = bool(
        getattr(continuation, "prediction_truncated", False)
    )
    trajectory.prediction_coverage_source = "constant_velocity_extension"
    return trajectory


def prediction_coverage(trajectory: Trajectory) -> dict[str, Any]:
    return {
        "required_prediction_distance": getattr(
            trajectory,
            "required_prediction_distance",
            None,
        ),
        "available_lane_chain_distance": getattr(
            trajectory,
            "available_lane_chain_distance",
            None,
        ),
        "prediction_truncated": bool(
            getattr(trajectory, "prediction_truncated", False)
        ),
        "source": getattr(
            trajectory,
            "prediction_coverage_source",
            "constant_velocity_prediction",
        ),
    }


def inflated_box(
    state: State,
    length: float,
    width: float,
    longitudinal_margin: float,
    lateral_margin: float,
) -> Rectangle:
    yaw = float(state.yaw or 0.0)
    return Rectangle(
        center=[float(state.x), float(state.y)],
        length=float(length) + 2.0 * longitudinal_margin,
        width=float(width) + 2.0 * lateral_margin,
        yaw=yaw,
    )


def trajectories_collide(
    ego_trajectory: Trajectory | Iterable[State],
    ego_dimensions: tuple[float, float],
    sv_trajectory: Trajectory | Iterable[State],
    sv_dimensions: tuple[float, float],
    *,
    longitudinal_margin: float = 0.5,
    lateral_margin: float = 0.25,
) -> tuple[Optional[float], float]:
    ego_states = _trajectory_states(ego_trajectory)
    sv_states = _trajectory_states(sv_trajectory)
    sv_by_time = {
        round(float(state.t), 3): state
        for state in sv_states
    }

    first_collision_t = None
    min_clearance = float("inf")
    matched_times = 0
    ego_length, ego_width = map(float, ego_dimensions)
    sv_length, sv_width = map(float, sv_dimensions)
    if min(ego_length, ego_width, sv_length, sv_width) <= 0:
        raise ValueError("Vehicle dimensions must be positive")

    inflated_ego_radius = 0.5 * math.hypot(
        ego_length + 2.0 * longitudinal_margin,
        ego_width + 2.0 * lateral_margin,
    )
    inflated_sv_radius = 0.5 * math.hypot(
        sv_length + 2.0 * longitudinal_margin,
        sv_width + 2.0 * lateral_margin,
    )

    for ego_state in ego_states:
        key = round(float(ego_state.t), 3)
        sv_state = sv_by_time.get(key)
        if sv_state is None:
            continue
        matched_times += 1

        ego_box = inflated_box(
            ego_state,
            ego_length,
            ego_width,
            longitudinal_margin,
            lateral_margin,
        )
        sv_box = inflated_box(
            sv_state,
            sv_length,
            sv_width,
            longitudinal_margin,
            lateral_margin,
        )
        center_distance = math.hypot(
            float(ego_state.x) - float(sv_state.x),
            float(ego_state.y) - float(sv_state.y),
        )
        clearance = center_distance - inflated_ego_radius - inflated_sv_radius
        min_clearance = min(min_clearance, clearance)

        if (
            first_collision_t is None
            and RecCollide(ego_box, sv_box).isCollide()
        ):
            first_collision_t = float(ego_state.t)

    if matched_times == 0:
        raise ValueError("Ego and surrounding-vehicle trajectories share no timestamps")
    return first_collision_t, min_clearance


def _risk_key(assessment: ActionSafety) -> tuple:
    collision_time = (
        float("inf")
        if assessment.first_collision_t is None
        else float(assessment.first_collision_t)
    )
    return (
        bool(assessment.trajectory_generated),
        bool(not assessment.prediction_truncated),
        bool(assessment.traffic_control_feasible),
        bool(assessment.collision_free),
        collision_time,
        float(assessment.min_clearance),
    )


class SafetyShield:
    def __init__(
        self,
        *,
        horizon: float = 5.0,
        dt: float = 0.1,
        longitudinal_margin: float = 0.5,
        lateral_margin: float = 0.25,
        traffic_control_commitment_horizon: Optional[float] = None,
    ) -> None:
        if horizon <= 0 or dt <= 0:
            raise ValueError("Shield horizon and dt must be positive")
        if longitudinal_margin < 0 or lateral_margin < 0:
            raise ValueError("Shield geometry margins cannot be negative")
        if traffic_control_commitment_horizon is None:
            traffic_control_commitment_horizon = min(1.0, float(horizon))
        if not 0 < traffic_control_commitment_horizon <= horizon:
            raise ValueError(
                "Traffic-control commitment horizon must be in (0, horizon]"
            )
        self.horizon = float(horizon)
        self.dt = float(dt)
        self.longitudinal_margin = float(longitudinal_margin)
        self.lateral_margin = float(lateral_margin)
        self.traffic_control_commitment_horizon = float(
            traffic_control_commitment_horizon
        )

    def evaluate(
        self,
        *,
        planner,
        sim_time: float,
        roadgraph: RoadGraph,
        vehicles_info: dict,
        base_action_ids: Iterable[int],
    ) -> ShieldResult:
        base_actions = []
        for action in base_action_ids:
            action_id = int(action)
            if action_id not in VALID_ACTION_IDS:
                raise ValueError(f"Invalid base action id: {action_id}")
            if action_id not in base_actions:
                base_actions.append(action_id)
        if not base_actions:
            raise ValueError("Safety shield requires at least one base action")

        ego = vehicles_info["egoCar"]
        ego_dimensions = (float(ego["length"]), float(ego["width"]))
        surrounding = list(vehicles_info.get("carInAoI", []))
        assessments: dict[int, ActionSafety] = {}
        red_light_constraint = _active_red_light_constraint(
            roadgraph,
            ego,
        )

        for action_id in base_actions:
            first_collision_t: Optional[float] = None
            min_clearance = float("inf")
            conflict_vehicle_id: Optional[str] = None
            generation_error = ""
            prediction_coverage_by_vehicle: dict[str, dict[str, Any]] = {}
            traffic_control_feasible = True
            traffic_control_violation = ""
            red_light_crossing_t: Optional[float] = None
            try:
                ego_trajectory = preview_ego_trajectory(
                    planner,
                    sim_time,
                    roadgraph,
                    vehicles_info,
                    action_id,
                )
                ego_trajectory = extend_trajectory_to_horizon(
                    ego_trajectory,
                    ego,
                    roadgraph,
                    horizon=self.horizon,
                    dt=self.dt,
                )
                prediction_coverage_by_vehicle["ego"] = prediction_coverage(
                    ego_trajectory
                )
                if red_light_constraint is not None:
                    (
                        current_lane,
                        red_junction_lane,
                        _stop_line_distance,
                    ) = red_light_constraint
                    red_light_crossing_t = (
                        _red_light_trajectory_crossing_time(
                            ego_trajectory,
                            roadgraph,
                            current_lane,
                            red_junction_lane,
                            ego_dimensions[0],
                        )
                    )
                    if (
                        red_light_crossing_t is not None
                        and red_light_crossing_t
                        <= self.traffic_control_commitment_horizon + 1e-9
                    ):
                        traffic_control_feasible = False
                        traffic_control_violation = (
                            "red_light_stop_line_crossing"
                        )
                for vehicle in surrounding:
                    sv_trajectory = predict_other_vehicle_trajectory(
                        vehicle,
                        roadgraph,
                        horizon=self.horizon,
                        dt=self.dt,
                    )
                    vehicle_id = str(vehicle["id"])
                    prediction_coverage_by_vehicle[vehicle_id] = (
                        prediction_coverage(sv_trajectory)
                    )
                    collision_t, clearance = trajectories_collide(
                        ego_trajectory,
                        ego_dimensions,
                        sv_trajectory,
                        (
                            float(vehicle["length"]),
                            float(vehicle["width"]),
                        ),
                        longitudinal_margin=self.longitudinal_margin,
                        lateral_margin=self.lateral_margin,
                    )
                    min_clearance = min(min_clearance, clearance)
                    if collision_t is not None and (
                        first_collision_t is None
                        or collision_t < first_collision_t
                    ):
                        first_collision_t = collision_t
                        conflict_vehicle_id = vehicle_id
            except Exception as exc:
                generation_error = f"{type(exc).__name__}: {exc}"

            trajectory_generated = not generation_error
            prediction_truncated = any(
                bool(coverage.get("prediction_truncated", False))
                for coverage in prediction_coverage_by_vehicle.values()
            )
            collision_free = (
                trajectory_generated
                and not prediction_truncated
                and first_collision_t is None
            )
            assessments[action_id] = ActionSafety(
                action_id=action_id,
                trajectory_generated=trajectory_generated,
                collision_free=collision_free,
                first_collision_t=first_collision_t,
                min_clearance=min_clearance,
                conflict_vehicle_id=conflict_vehicle_id,
                generation_error=generation_error,
                prediction_truncated=prediction_truncated,
                prediction_coverage_by_vehicle=(
                    prediction_coverage_by_vehicle
                ),
                traffic_control_feasible=traffic_control_feasible,
                traffic_control_violation=traffic_control_violation,
                red_light_active=(red_light_constraint is not None),
                red_light_stop_line_distance=(
                    red_light_constraint[2]
                    if red_light_constraint is not None
                    else None
                ),
                red_light_crossing_t=red_light_crossing_t,
                traffic_control_commitment_horizon=(
                    self.traffic_control_commitment_horizon
                    if red_light_constraint is not None
                    else None
                ),
            )

        complete_assessments = [
            assessment
            for assessment in assessments.values()
            if (
                assessment.trajectory_generated
                and not assessment.prediction_truncated
            )
        ]
        if not complete_assessments:
            raise SafetyShieldEvaluationError(assessments)

        safe_actions = [
            action_id
            for action_id in base_actions
            if (
                assessments[action_id].collision_free
                and assessments[action_id].traffic_control_feasible
            )
        ]
        if red_light_constraint is not None and not safe_actions:
            raise SafetyShieldNoAdmissibleActionError(assessments)
        least_risk = max(complete_assessments, key=_risk_key)
        return ShieldResult(
            base_actions=base_actions,
            safe_actions=safe_actions,
            least_risk_action=least_risk.action_id,
            no_safe_action=not safe_actions,
            assessments=assessments,
        )
