import os
import argparse
import csv
import hashlib
import json
import math
import random
import re
import sqlite3
import statistics
import sys
import textwrap
import time
import traceback
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from rich import print
from typing import Any, List, Tuple
from datetime import datetime
from langchain.chat_models import AzureChatOpenAI, ChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from DriverAgent.Memory import DrivingMemory
from simInfo.ICRLEnvDescriptor import ICRLEnvDescription

from simInfo.CustomExceptions import (
    CollisionChecker, CollisionException, 
    record_result, LaneChangeException, 
    BrainDeadlockException, TimeOutException
)
from simModel.Model import Model
from simModel.MPGUI import GUI
from trafficManager.traffic_manager import TrafficManager
from trafficManager.safety_shield import (
    SafetyShield,
    SafetyShieldEvaluationError,
    ShieldResult,
)
from langchain.callbacks.openai_info import OpenAICallbackHandler

from simModel.DataQueue import QuestionAndAnswer
from formal_domain_config import (
    ALL_FORMAL_DOMAINS,
    SCENE_CONFIG_ID,
    WEATHER_ID_TO_LEVEL,
    domain_metadata,
    find_route_entry,
    frozen_randomtrips_background_path,
    make_frozen_randomtrips_density_route_file,
    make_density_route_file as make_formal_density_route_file,
    require_route_entry_available,
)

import logger, logging
from trafficManager.common.vehicle import Behaviour

CODE_DIR = Path(__file__).resolve().parent / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
from h20_ours_runtime import FinalAdaptiveH20Selector

decision_logger = logger.setup_app_level_logger(
    logger_name="LLMAgent", file_name="llm_decision.log")
LLM_logger = logging.getLogger("LLMAgent").getChild(__name__)

DRIVING_POLICY_MODEL = "gpt-4o"
DRIVING_POLICY_RETURNED_MODEL = "gpt-4o-2024-08-06"
DRIVING_POLICY_PROTOCOL_ID = "h20_driving_policy"
HISTORY_STORAGE_ORDER = "newest_to_oldest"
HISTORY_RENDER_ORDER = "oldest_to_newest"
SELECTED_MASK_ORDER = "oldest_to_newest"
DRIVING_POLICY_PROMPT_TEMPLATE = """        # Here is the current scenario:
        ## Driving scenario description:
        {scenario_description}

        ## Navigation instruction:
        {driving_intensions}

        ## Available actions:
        {available_actions}

        ##
        You MUST choose exactly one Action_id from this CURRENT Available actions
        list: {displayed_action_ids}. Actions mentioned in examples or past
        experiences are unavailable unless their Action_id is also in this list.
        Never select, recommend, or output an Action_id outside this list. If your
        preferred action is unavailable, choose the safest legal alternative from
        this list.

        Remember to follow the format instructions and think more steps.
        You can stop reasoning once you have a valid action to take. After you come to a decision, score your decision for safety, efficiency, and comfort. Your goal should be to maximize for all three categories. Print out the individual scores, and come up with a single scalar score as your "reward". Write this reward as "FINAL REWARD: [only the scalar as a decimal]". DO NOT SHOW CALCULATIONS. Present your final answer exactly as in the example.
"""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode())


def sha256_path(path: str | Path) -> str:
    candidate = Path(path).expanduser().resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"Cannot hash missing provenance file: {candidate}")
    return sha256_bytes(candidate.read_bytes())


def serialized_messages(messages: list[Any]) -> list[dict[str, str]]:
    role_by_type = {
        SystemMessage: "system",
        HumanMessage: "user",
        AIMessage: "assistant",
    }
    return [
        {
            "role": role_by_type.get(type(message), type(message).__name__),
            "content": str(getattr(message, "content", "")),
        }
        for message in messages
    ]


def response_model_id(response: Any) -> str:
    metadata = getattr(response, "response_metadata", {}) or {}
    generation = getattr(response, "generation_info", {}) or {}
    additional = getattr(response, "additional_kwargs", {}) or {}
    for source in (metadata, generation, additional):
        for key in ("model_name", "model", "model_id"):
            value = source.get(key) if isinstance(source, dict) else None
            if value:
                return str(value)
    return ""


def write_context_pool_log(log_path: str, record: dict) -> None:
    directory = os.path.dirname(log_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()
        os.fsync(f.fileno())


def write_episode_status(status_path: Path, status_record: dict) -> None:
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status_record, indent=2) + "\n")


SELECTOR_LOG_FIELDS = [
    "pair_id", "episode_id", "route_id", "domain_id", "sumo_seed",
    "context_policy", "adaptive_selector_artifact_sha256",
    "decision_index", "H_available", "selected_mask", "selected_mask_order",
    "selected_K", "ensemble_mean_logkl", "ensemble_std_logkl", "selected_upper_bound",
    "threshold_log1p_tau", "fallback_to_full", "actual_input_tokens", "completion_tokens",
    "total_tokens", "action_selected", "selector_inference_seconds", "driving_policy_api_seconds",
    "policy_api_call_count", "policy_api_request_attempts", "policy_api_retry_attempts",
    "action_contract_repair_count", "initial_invalid_action",
    "action_parse_method", "action_parse_fallback_used", "action_parse_error",
    "reward_parse_method", "reward_parse_error",
    "base_action_ids", "safe_action_ids", "policy_action_ids",
    "shield_triggered", "forced_single_action",
    "trajectory_generated_by_action", "collision_free_by_action",
    "prediction_truncated_by_action", "prediction_coverage_by_action",
    "first_collision_t_by_action", "min_clearance_by_action",
    "conflict_vehicle_by_action", "trajectory_generation_error_by_action",
    "raw_llm_action", "executed_action", "shield_override",
    "no_safe_action", "least_risk_action",
    "history_write_skipped_reason",
]


def append_selector_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    if exists:
        with path.open(newline="") as handle:
            if next(csv.reader(handle), None) != SELECTOR_LOG_FIELDS:
                raise ValueError("Selector CSV header changed; use a new output prefix")
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SELECTOR_LOG_FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def shield_log_fields(
    result: ShieldResult,
    *,
    raw_llm_action: Any = "",
    executed_action: Any = "",
    shield_override: Any = "",
    history_write_skipped_reason: str = "",
) -> dict[str, Any]:
    fields = result.audit_fields()
    for key in [
        "base_action_ids",
        "safe_action_ids",
        "policy_action_ids",
        "trajectory_generated_by_action",
        "collision_free_by_action",
        "prediction_truncated_by_action",
        "prediction_coverage_by_action",
        "first_collision_t_by_action",
        "min_clearance_by_action",
        "conflict_vehicle_by_action",
        "trajectory_generation_error_by_action",
    ]:
        fields[key] = json.dumps(fields[key], sort_keys=True)
    fields.update(
        {
            "raw_llm_action": raw_llm_action,
            "executed_action": executed_action,
            "shield_override": shield_override,
            "history_write_skipped_reason": history_write_skipped_reason,
        }
    )
    return fields


def unshielded_result(base_action_ids: list[int]) -> ShieldResult:
    actions = [int(action_id) for action_id in base_action_ids]
    if not actions:
        raise ValueError("Unshielded policy requires at least one base action")
    return ShieldResult(
        base_actions=actions,
        safe_actions=list(actions),
        least_risk_action=actions[0],
        no_safe_action=False,
        assessments={},
    )


def valid_history_indices(
    previous_prompts: list[Any],
    previous_decisions: list[Any],
    rewards: list[Any],
) -> list[int]:
    if not (
        len(previous_prompts)
        == len(previous_decisions)
        == len(rewards)
    ):
        raise RuntimeError(
            "History buffers are misaligned: "
            f"prompts={len(previous_prompts)}, "
            f"decisions={len(previous_decisions)}, "
            f"rewards={len(rewards)}"
        )

    valid = []
    for idx, (prompt, decision, reward) in enumerate(
        zip(previous_prompts, previous_decisions, rewards)
    ):
        try:
            action_ok = int(decision) in {1, 2, 3, 4, 8}
        except (TypeError, ValueError):
            action_ok = False
        try:
            reward_ok = bool(str(reward).strip()) and math.isfinite(float(reward))
        except (TypeError, ValueError):
            reward_ok = False
        prompt_ok = bool(str(prompt or "").strip())
        if not (prompt_ok and action_ok and reward_ok):
            raise RuntimeError(
                "Invalid history chunk: "
                f"index={idx}, prompt_ok={prompt_ok}, "
                f"action_ok={action_ok}, reward_ok={reward_ok}"
            )
        valid.append(idx)
    return valid


def subset_buffers(
    indices: list[int],
    previous_prompts: list[Any],
    previous_decisions: list[Any],
    rewards: list[Any],
) -> tuple[list[Any], list[Any], list[Any]]:
    ordered = sorted(indices)
    return (
        [previous_prompts[idx] for idx in ordered],
        [previous_decisions[idx] for idx in ordered],
        [rewards[idx] for idx in ordered],
    )


def compose_selected_env_prompt(
    short_prompt: str,
    selected_raw_indices: list[int],
    selected_prompts: list[Any],
    selected_decisions: list[Any],
    selected_rewards: list[Any],
) -> str:
    """Rebuild the descriptor history suffix without a second scene query."""
    if not (
        len(selected_raw_indices) == len(selected_prompts)
        == len(selected_decisions) == len(selected_rewards)
    ):
        raise RuntimeError("Selected history buffers are misaligned")
    prompt = str(short_prompt)
    prompt += "Here are your previous steps, the decisions you took, and their corresponding rewards:\n\n"
    history = zip(selected_raw_indices, selected_prompts, selected_decisions, selected_rewards)
    # Live buffers are newest-to-oldest. Preserve acquisition-time canonical
    # oldest-to-newest Step labels instead of renumbering the selected subset.
    for raw_index, step, decision, reward in reversed(list(history)):
        prompt += f"Step {20 - int(raw_index)}: {step}\n"
        prompt += f"Decision: {decision}\n"
        prompt += f"Reward: {reward}\n\n"
    return prompt


def update_full_history(
    previous_prompts: list[Any],
    previous_decisions: list[Any],
    rewards: list[Any],
    short_prompt: str,
    action: int,
    reward: Any,
    limit: int | None,
) -> None:
    for buffer, value in [
        (previous_prompts, short_prompt),
        (previous_decisions, action),
        (rewards, reward),
    ]:
        if limit is not None and len(buffer) >= limit:
            buffer.pop()
        buffer.insert(0, value)


def evaluate_database_metrics(database: str) -> dict[str, Any]:
    """Replay the DB with the unchanged original Decision_Evaluation implementation."""
    result = {
        "evaluation_status": "failed", "evaluation_error": "",
        "evaluation_valid": False, "recorded_episode_success": None,
        "num_eval_frames": 0,
    }
    try:
        from DriverAgent.Evaluation import Decision_Evaluation
        from simModel.Replay import ReplayModel

        replay = ReplayModel(database)
        evaluator = Decision_Evaluation(database, replay.timeStep)
        while not replay.tpEnd:
            replay.runStep()
            evaluator.Evaluate(replay)
        conn = sqlite3.connect(database)
        result_row = conn.execute(
            "SELECT result, complete_percentage, drive_score, total_score, fail_reason FROM resultINFO LIMIT 1"
        ).fetchone()
        eval_row = conn.execute(
            "SELECT count(*), avg(collision_score), avg(comfort_score), avg(efficiency_score), "
            "avg(speed_limit_score), avg(traffic_light_score), avg(decision_score), "
            "count(collision_score), count(comfort_score), count(efficiency_score) FROM evaluationINFO"
        ).fetchone()
        conn.close()
        if result_row:
            raw_success, completion, drive_score, final_score, fail_reason = result_row
            result.update(
                recorded_episode_success=bool(raw_success),
                route_completion=completion, drive_score=drive_score,
                final_score=final_score, evaluation_fail_reason=fail_reason,
            )
        if eval_row:
            count, safety, comfort, efficiency, speed_limit, traffic_light, decision_score = eval_row[:7]
            result.update(
                num_eval_frames=int(count or 0), safety=safety, comfort=comfort,
                efficiency=efficiency, speed_limit_score=speed_limit,
                traffic_light_score=traffic_light, decision_score=decision_score,
            )
        if result_row is None:
            raise ValueError("Evaluation produced no episode result")
        if eval_row is None or int(eval_row[0] or 0) <= 0:
            raise ValueError("Evaluation produced no scored frames")
        if any(count != eval_row[0] for count in eval_row[7:]):
            raise ValueError("Evaluation has frames with missing Safety, Comfort, or Efficiency")
        if any(value is None or not math.isfinite(float(value)) for value in eval_row[1:4]):
            raise ValueError("Evaluation requires finite Safety, Comfort, and Efficiency")
        result.update(evaluation_status="success", evaluation_valid=True)
        return result
    except Exception as exc:
        result["evaluation_valid"] = False
        result["evaluation_error"] = f"{type(exc).__name__}: {exc}"
        result["evaluation_traceback"] = traceback.format_exc()
        return result


class MaxDecisionsReached(Exception):
    pass


class ActionParseException(Exception):
    """Raised when one policy response has no unambiguous valid action."""

    def __init__(self, message: str, metadata: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.metadata = dict(metadata or {})


class RewardParseException(ActionParseException):
    """Raised when a policy response has no single finite scalar reward."""


VALID_ACTION_IDS = {1, 2, 3, 4, 8}


def _strip_final_reward_lines(text: str) -> str:
    return re.sub(
        r"(?im)^\s*(?:[-*+]\s*)?(?:#+\s*)?FINAL\s+REWARD\s*:.*$",
        "",
        str(text or ""),
    )


def _last_labeled_final_section(response_text: str) -> str:
    matches = list(
        re.finditer(
            r"(?im)^\s*(?:final\s+answer|final\s+decision|decision|"
            r"final\s+action|action|action_id)\s*[:=-]",
            response_text or "",
        )
    )
    return response_text[matches[-1].end() :] if matches else ""


def _exact_labeled_action_line(response_text: str) -> int | None:
    matches = list(
        re.finditer(
            r"(?im)^\s*(?:final\s+answer|final\s+decision|decision|"
            r"final\s+action|action|action_id)\s*[:=-]\s*(.+?)\s*$",
            response_text or "",
        )
    )
    for match in reversed(matches):
        value = match.group(1).strip().rstrip(".;").strip().lower()
        numeric = re.fullmatch(r"(?:action(?:_id)?\s*[:=-]?\s*)?([12348])", value)
        if numeric:
            return int(numeric.group(1))
        for pattern, action_id in [
            (r"(?:idle|remain\s+idle)", 8),
            (r"(?:acceleration|accelerate)", 1),
            (r"(?:deceleration|decelerate)", 2),
            (r"(?:turn-left|change\s+lane\s+to\s+the\s+left)", 3),
            (r"(?:turn-right|change\s+lane\s+to\s+the\s+right)", 4),
        ]:
            if re.fullmatch(pattern, value):
                return action_id
    return None


def parse_policy_action(response_text: str, delimiter: str) -> tuple[int, str, bool]:
    """Recover exactly one action locally without another model call."""
    text = str(response_text or "")
    if delimiter and delimiter in text:
        response_line_matches = list(
            re.finditer(
                rf"(?im)^\s*response\s+to\s+user\s*:\s*{re.escape(delimiter)}"
                rf"\s*([12348])\s*[.;]?\s*$",
                text,
            )
        )
        if response_line_matches:
            return int(response_line_matches[-1].group(1)), "explicit_response_to_user_action_line", True

        final_delimited = text.rsplit(delimiter, 1)[1].strip()
        exact = re.fullmatch(r"([12348])", final_delimited)
        if exact:
            return int(exact.group(1)), "exact_integer_after_final_delimiter", False

        # The deployed response may append a reward line after the requested
        # action. Accept only a standalone action on the first delimited line;
        # values such as "8.5" remain ineligible.
        first_action_line = re.match(r"^\s*([12348])\s*(?:\r?\n|$)", final_delimited)
        if first_action_line:
            return int(first_action_line.group(1)), "action_line_after_final_delimiter", True

    labeled_line_action = _exact_labeled_action_line(text)
    if labeled_line_action is not None:
        return labeled_line_action, "exact_labeled_action_line", True

    labeled_section = _last_labeled_final_section(text)
    if labeled_section:
        clean_labeled_section = _strip_final_reward_lines(labeled_section)
        label_text = clean_labeled_section.strip().lower()
        named_actions = []
        for pattern, action_id in [
            (r"\b(?:acceleration|accelerate)\b", 1),
            (r"\b(?:deceleration|decelerate)\b", 2),
            (r"\b(?:turn-left|change\s+lane\s+to\s+the\s+left)\b", 3),
            (r"\b(?:turn-right|change\s+lane\s+to\s+the\s+right)\b", 4),
            (r"\bidle\b", 8),
        ]:
            if re.search(pattern, label_text):
                named_actions.append(action_id)
        if len(set(named_actions)) == 1:
            return named_actions[0], "explicit_final_decision_label", True

    raise ActionParseException(
        "No unambiguous action id in {1,2,3,4,8} could be parsed from the single policy response",
        {
            "policy_api_call_count": 1,
            "action_parse_method": "failed",
            "action_parse_fallback_used": True,
            "action_parse_error": "unambiguous_valid_action_not_found",
            "response": text,
        },
    )


def parse_policy_reward(response_text: str) -> str:
    """Require one finite FINAL REWARD line, optionally Markdown-bulleted."""
    matches = list(
        re.finditer(
            r"(?im)^\s*(?:[-*+]\s*)?(?:#+\s*)?FINAL\s+REWARD\s*:\s*"
            r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*$",
            str(response_text or ""),
        )
    )
    if len(matches) != 1:
        reason = "missing_final_reward" if not matches else "multiple_final_rewards"
        raise RewardParseException(
            "Policy response must contain exactly one finite decimal FINAL REWARD",
            {
                "action_parse_error": "invalid_final_reward",
                "reward_parse_method": "failed",
                "reward_parse_error": reason,
                "response": str(response_text or ""),
            },
        )
    value = matches[0].group(1)
    if not math.isfinite(float(value)):
        raise RewardParseException(
            "FINAL REWARD must be finite",
            {
                "action_parse_error": "invalid_final_reward",
                "reward_parse_method": "failed",
                "reward_parse_error": "non_finite_final_reward",
                "response": str(response_text or ""),
            },
        )
    return value


FAILURE_MESSAGE_LIMIT = 8000


def exception_message(exc: BaseException) -> str:
    message = str(exc) or repr(exc)
    if len(message) > FAILURE_MESSAGE_LIMIT:
        return message[:FAILURE_MESSAGE_LIMIT] + "...[truncated]"
    return message


def exception_text(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exception_message(exc)}"


def is_quota_or_billing_error(exc: BaseException) -> bool:
    error_text = exception_text(exc).lower()
    quota_patterns = [
        "you exceeded your current quota",
        "please check your plan and billing details",
        "insufficient_quota",
        "billing",
    ]
    return any(pattern in error_text for pattern in quota_patterns)


def classify_failure_error(exc: BaseException) -> str:
    if isinstance(exc, KeyboardInterrupt):
        return "keyboard_interrupt"
    if is_quota_or_billing_error(exc):
        return "insufficient_quota_or_billing"
    return type(exc).__name__


def normalize_terminal_outcome(
    episode_status: str,
    failure_error_type: str,
    failure_stage: str,
) -> str:
    if episode_status == "success":
        return "success"
    if episode_status == "max_decisions_reached" or failure_error_type == "MaxDecisionsReached":
        return "max_decisions_reached"
    mapped = {
        "CollisionException": "collision",
        "TimeOutException": "timeout",
        "LaneChangeException": "lane_change_failure",
        "BrainDeadlockException": "deadlock",
    }
    if failure_error_type in mapped:
        return mapped[failure_error_type]
    if failure_stage in {
        "llm_invoke",
        "shield_evaluation_failure",
    } or failure_error_type in {
        "insufficient_quota_or_billing",
        "APIConnectionError",
        "APITimeoutError",
        "RateLimitError",
        "ServiceUnavailableError",
        "keyboard_interrupt",
    }:
        return "infrastructure_failure"
    return "other_failure"


def recent_action_streak(action_list: List[int]) -> Tuple[int | None, int]:
    if not action_list:
        return None, 0
    last_action = action_list[-1]
    streak_len = 0
    for action in reversed(action_list):
        if action != last_action:
            break
        streak_len += 1
    return last_action, streak_len


def original_deadlock_condition(
    action_list: List[int],
    ego_speed: float,
    window: int = 10,
    speed_threshold: float = 0.1,
) -> bool:
    """Match the released loop: check the last 10 actions only after action 10."""
    if window <= 0:
        raise ValueError("Deadlock window must be positive")
    if len(action_list) <= window:
        return False
    recent_actions = action_list[-window:]
    return (
        min(recent_actions) == max(recent_actions)
        and ego_speed <= speed_threshold
    )


def resolved_path(raw_path: str) -> str:
    return str(Path(raw_path).expanduser().resolve())


def route_depart(vehicle: ET.Element) -> float:
    return float(vehicle.get("depart", "0"))


def positive_depart_gaps_from_vehicles(vehicles: list[ET.Element]) -> list[float]:
    departures = sorted(route_depart(vehicle) for vehicle in vehicles)
    return [
        round(b - a, 6)
        for a, b in zip(departures, departures[1:])
        if b > a
    ]


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
) -> Path:
    """Local fallback density generator using randomTrips-style period scaling."""
    if density_scale <= 0:
        raise ValueError(f"density_scale must be positive, got {density_scale}")
    if abs(density_scale - round(density_scale)) > 1e-9:
        raise ValueError(f"density_scale must be integer 1/2/3-style, got {density_scale}")
    scale_int = int(round(density_scale))

    ET.register_namespace("xsi", "http://www.w3.org/2001/XMLSchema-instance")
    tree = ET.parse(base_rou_file)
    root = tree.getroot()
    vehicles = [child for child in list(root) if child.tag == "vehicle"]
    if not vehicles:
        raise ValueError(f"No <vehicle> entries found in {base_rou_file}")

    ego_vehicle = next((vehicle for vehicle in vehicles if vehicle.get("id") == ego_id), None)
    if ego_vehicle is None:
        raise ValueError(f"Ego vehicle id={ego_id!r} not found in {base_rou_file}")

    gaps = positive_depart_gaps_from_vehicles(vehicles)
    base_period = sorted(gaps)[len(gaps) // 2] if gaps else 2.0

    departs = [route_depart(vehicle) for vehicle in vehicles]
    min_depart = min(departs)
    max_depart = max(departs)
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
    return output_rou_file


def _lane_allows_passenger(lane: ET.Element) -> bool:
    """Return whether SUMO permits a passenger vehicle on this lane."""
    allowed = set((lane.get("allow") or "").split())
    disallowed = set((lane.get("disallow") or "").split())
    if allowed:
        return "all" in allowed or "passenger" in allowed
    return "all" not in disallowed and "passenger" not in disallowed


def _load_net_lane_connections(sumo_net_file: Path) -> tuple[dict[str, list[str]], dict[tuple[str, str], set[str]]]:
    """Return passenger-drivable lanes and route-compatible from-lanes."""
    root = ET.parse(sumo_net_file).getroot()
    edge_lanes: dict[str, list[str]] = {}
    for edge in root.findall("edge"):
        edge_id = edge.get("id")
        if not edge_id or edge_id.startswith(":"):
            continue
        lanes = [
            lane.get("id")
            for lane in edge.findall("lane")
            if lane.get("id") and _lane_allows_passenger(lane)
        ]
        if lanes:
            edge_lanes[edge_id] = lanes

    connections: dict[tuple[str, str], set[str]] = {}
    for connection in root.findall("connection"):
        from_edge = connection.get("from")
        to_edge = connection.get("to")
        from_lane = connection.get("fromLane")
        if (
            not from_edge
            or not to_edge
            or from_lane is None
            or from_edge.startswith(":")
            or to_edge.startswith(":")
        ):
            continue
        source_lane = f"{from_edge}_{from_lane}"
        if source_lane in edge_lanes.get(from_edge, []):
            connections.setdefault((from_edge, to_edge), set()).add(source_lane)
    return edge_lanes, connections


def _load_net_lane_transition_map(
    sumo_net_file: Path,
) -> dict[tuple[str, str], dict[str, set[str]]]:
    """Return passenger-drivable route-edge lane mappings."""
    root = ET.parse(sumo_net_file).getroot()
    drivable_lanes = {
        lane.get("id")
        for edge in root.findall("edge")
        if not (edge.get("id") or "").startswith(":")
        for lane in edge.findall("lane")
        if lane.get("id") and _lane_allows_passenger(lane)
    }
    mappings: dict[tuple[str, str], dict[str, set[str]]] = {}
    for connection in root.findall("connection"):
        from_edge = connection.get("from")
        to_edge = connection.get("to")
        from_lane = connection.get("fromLane")
        to_lane = connection.get("toLane")
        if (
            not from_edge
            or not to_edge
            or from_lane is None
            or to_lane is None
            or from_edge.startswith(":")
            or to_edge.startswith(":")
        ):
            continue
        source = f"{from_edge}_{from_lane}"
        target = f"{to_edge}_{to_lane}"
        if source not in drivable_lanes or target not in drivable_lanes:
            continue
        mappings.setdefault((from_edge, to_edge), {}).setdefault(source, set()).add(target)
    return mappings


def _lane_index(lane_id: str) -> int:
    return int(str(lane_id).rsplit("_", 1)[-1])


def _choose_noncompatible_depart_lane(
    sumo_net_file: Path,
    route_edges: list[str],
) -> dict:
    """Choose a start lane that makes the *original* navigation logic ask for a lane change.

    We only use the first edge transition. This keeps the route definition transparent:
    the ego departs in a lane that exists on the first edge but is not route-compatible
    for the next edge. The prompt/policy/action semantics are unchanged.
    """
    if len(route_edges) < 2:
        raise ValueError("Need at least two route edges to construct a lane-challenge route.")
    edge_lanes, connections = _load_net_lane_connections(sumo_net_file)
    from_edge, to_edge = route_edges[0], route_edges[1]
    lanes = set(edge_lanes.get(from_edge, []))
    compatible = set(connections.get((from_edge, to_edge), set()))
    noncompatible = lanes - compatible
    if len(lanes) < 2 or not compatible or not noncompatible:
        raise ValueError(
            "First route transition is not a clean lane-change challenge: "
            f"{from_edge}->{to_edge}, lanes={sorted(lanes)}, compatible={sorted(compatible)}, "
            f"noncompatible={sorted(noncompatible)}. Use a different ego-route."
        )

    compatible_indices = [_lane_index(lane_id) for lane_id in compatible]
    # Prefer the non-compatible lane farthest from any compatible lane, so the
    # route-compatible direction is unambiguous and less likely to be accidentally
    # corrected before H>=5.
    chosen = max(
        noncompatible,
        key=lambda lane_id: min(abs(_lane_index(lane_id) - idx) for idx in compatible_indices),
    )
    chosen_idx = _lane_index(chosen)
    nearest_compatible_idx = min(compatible_indices, key=lambda idx: abs(chosen_idx - idx))
    required_direction = "left" if nearest_compatible_idx > chosen_idx else "right"
    return {
        "from_edge": from_edge,
        "to_edge": to_edge,
        "edge_lanes": sorted(lanes),
        "route_compatible_lanes": sorted(compatible),
        "route_incompatible_lanes": sorted(noncompatible),
        "forced_depart_lane_id": chosen,
        "forced_depart_lane_index": chosen_idx,
        "expected_navigation_direction": required_direction,
    }


def _choose_first_structural_noncompatible_depart_lane(
    sumo_net_file: Path,
    route_edges: list[str],
    *,
    prefer_nearest: bool = False,
) -> dict:
    """Choose a depart lane that reaches the first later structural constraint.

    Every transition before the target is followed using exact SUMO lane-to-lane
    connections. A start lane is eligible only when all of its reachable lanes at
    the target transition are route-incompatible, so the challenge is deterministic.
    """
    if len(route_edges) < 2:
        raise ValueError("Need at least two route edges to construct a lane-challenge route.")
    edge_lanes, connections = _load_net_lane_connections(sumo_net_file)
    transition_map = _load_net_lane_transition_map(sumo_net_file)

    target_index = None
    target_lanes: set[str] = set()
    target_compatible: set[str] = set()
    target_noncompatible: set[str] = set()
    for index, (from_edge, to_edge) in enumerate(zip(route_edges, route_edges[1:])):
        lanes = set(edge_lanes.get(from_edge, []))
        compatible = set(connections.get((from_edge, to_edge), set()))
        noncompatible = lanes - compatible
        if len(lanes) >= 2 and compatible and noncompatible:
            target_index = index
            target_lanes = lanes
            target_compatible = compatible
            target_noncompatible = noncompatible
            break
    if target_index is None:
        raise ValueError("Route has no clean structural lane-change transition")

    eligible: list[tuple[str, set[str]]] = []
    for depart_lane in edge_lanes.get(route_edges[0], []):
        reachable = {depart_lane}
        for index in range(target_index):
            pair = (route_edges[index], route_edges[index + 1])
            mapping = transition_map.get(pair, {})
            reachable = {
                target
                for source in reachable
                for target in mapping.get(source, set())
            }
            if not reachable:
                break
        if reachable and reachable <= target_noncompatible:
            eligible.append((depart_lane, reachable))
    if not eligible:
        raise ValueError(
            "No depart lane deterministically reaches a route-incompatible lane at "
            f"the first structural transition index={target_index}: "
            f"{route_edges[target_index]}->{route_edges[target_index + 1]}"
        )

    compatible_indices = [_lane_index(lane_id) for lane_id in target_compatible]
    def distance_to_compatible(item: tuple[str, set[str]]) -> int:
        return min(
            abs(_lane_index(target_lane) - compatible_index)
            for target_lane in item[1]
            for compatible_index in compatible_indices
        )

    if prefer_nearest:
        chosen, reachable_at_target = min(
            eligible,
            key=lambda item: (
                distance_to_compatible(item),
                _lane_index(item[0]),
                item[0],
            ),
        )
        chosen_target_lane = min(
            reachable_at_target,
            key=lambda lane_id: (
                min(abs(_lane_index(lane_id) - index) for index in compatible_indices),
                _lane_index(lane_id),
                lane_id,
            ),
        )
    else:
        chosen, reachable_at_target = max(
            eligible,
            key=distance_to_compatible,
        )
        chosen_target_lane = max(
            reachable_at_target,
            key=lambda lane_id: min(
                abs(_lane_index(lane_id) - index) for index in compatible_indices
            ),
        )
    chosen_target_index = _lane_index(chosen_target_lane)
    nearest_compatible_index = min(
        compatible_indices,
        key=lambda index: abs(chosen_target_index - index),
    )
    required_direction = (
        "left" if nearest_compatible_index > chosen_target_index else "right"
    )
    return {
        "from_edge": route_edges[target_index],
        "to_edge": route_edges[target_index + 1],
        "target_transition_index": target_index,
        "edge_lanes": sorted(target_lanes),
        "route_compatible_lanes": sorted(target_compatible),
        "route_incompatible_lanes": sorted(target_noncompatible),
        "forced_depart_lane_id": chosen,
        "forced_depart_lane_index": _lane_index(chosen),
        "reachable_target_lanes": sorted(reachable_at_target),
        "expected_navigation_direction": required_direction,
        "structural_lane_selection_policy": (
            "nearest_noncompatible" if prefer_nearest else "farthest_noncompatible"
        ),
        "required_lane_change_count": abs(
            nearest_compatible_index - chosen_target_index
        ),
    }


def make_ego_lane_challenge_route_file(
    input_rou_file: Path,
    output_rou_file: Path,
    sumo_net_file: Path,
    ego_id: str,
    mode: str,
) -> dict:
    """Create a route file with an integrity-safe ego initial-lane perturbation.

    This changes only the SUMO initial condition (`departLane`) for the selected
    ego vehicle. It does not edit the LLM prompt, the action set, rewards,
    selected-context policy, or downstream policy code.
    """
    if mode == "none":
        return {}
    if mode not in {
        "first_strict_noncompatible",
        "first_structural_noncompatible",
        "first_structural_nearest_noncompatible",
    }:
        raise ValueError(f"Unsupported ego-depart-lane-mode: {mode}")

    ET.register_namespace("xsi", "http://www.w3.org/2001/XMLSchema-instance")
    tree = ET.parse(input_rou_file)
    root = tree.getroot()
    ego_vehicle = next((vehicle for vehicle in root if vehicle.tag == "vehicle" and vehicle.get("id") == ego_id), None)
    if ego_vehicle is None:
        raise ValueError(f"Ego vehicle id={ego_id!r} not found in {input_rou_file}")
    route = ego_vehicle.find("route")
    if route is None or not route.get("edges"):
        raise ValueError(f"Ego vehicle id={ego_id!r} has no inline route edges in {input_rou_file}")
    route_edges = route.get("edges", "").split()
    if mode == "first_strict_noncompatible":
        challenge = _choose_noncompatible_depart_lane(sumo_net_file, route_edges)
    else:
        challenge = _choose_first_structural_noncompatible_depart_lane(
            sumo_net_file,
            route_edges,
            prefer_nearest=(mode == "first_structural_nearest_noncompatible"),
        )
    ego_vehicle.set("departLane", str(challenge["forced_depart_lane_index"]))
    # Keep the ego at the beginning of the route. We do not move the task near a junction;
    # this preserves a natural H>=5 warm-up before the lane-change need appears.
    ego_vehicle.set("departPos", ego_vehicle.get("departPos", "0"))

    output_rou_file.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_rou_file, encoding="utf-8", xml_declaration=True)
    challenge.update(
        {
            "ego_depart_lane_mode": mode,
            "ego_depart_lane_effective": True,
            "ego_depart_lane_route_file": str(output_rou_file.resolve()),
            "ego_depart_lane_integrity_note": (
                "Only the ego vehicle SUMO departLane initial condition was changed. "
                "No LLM prompt, downstream policy, action semantics, reward, or selector objective was changed."
            ),
        }
    )
    return challenge


def build_domain_config(
    args: argparse.Namespace,
    output_prefix: Path,
    context_log_path: str,
    base_rou_file: Path,
    generated_rou_file: Path | None,
    density_generation_info: dict | None,
) -> dict:
    domain_id = args.domain_id
    meta = domain_metadata(domain_id)
    density_scale = float(meta["traffic_density_scale"])
    weather_id = str(meta["weather_id"])
    traffic_density_id = str(meta["traffic_density_id"])

    if args.traffic_density_scale is not None and float(args.traffic_density_scale) != density_scale:
        raise ValueError(
            f"--traffic-density-scale={args.traffic_density_scale} conflicts with "
            f"domain {domain_id} preset density {density_scale}."
        )
    if args.weather_id is not None and args.weather_id != weather_id:
        raise ValueError(
            f"--weather-id={args.weather_id} conflicts with domain {domain_id} preset weather {weather_id}."
        )
    if args.traffic_density_id is not None and args.traffic_density_id != traffic_density_id:
        raise ValueError(
            f"--traffic-density-id={args.traffic_density_id} conflicts with "
            f"domain {domain_id} preset density id {traffic_density_id}."
        )
    weather_params = meta["weather_params"]
    density_effective = density_scale != 1.0

    config = {
        "scene_config_id": SCENE_CONFIG_ID,
        "domain_id": domain_id,
        "scenario_id": args.scenario_id,
        "episode_id": args.episode_id,
        "split_group_candidate": "unassigned_candidate",
        "sumo_seed": args.sumo_seed,
        "sumo_seed_effective": True,
        "traffic_density_id": traffic_density_id,
        "traffic_density_scale": density_scale,
        "traffic_density_effective": density_effective,
        "traffic_density_method": "randomTrips_period_scaling_route_demand",
        "base_sumo_rou_file": str(base_rou_file),
        "generated_sumo_rou_file": str(generated_rou_file) if generated_rou_file else "",
        "weather_id": weather_id,
        "weather_level": WEATHER_ID_TO_LEVEL[weather_id],
        "weather_params": weather_params,
        "weather_prompt_effective": True,
        "weather_sim_effective": False,
        "weather_effective_type": "prompt_observed_numeric_0_100",
        "map_id": args.map_id,
        "route_id": args.route_id,
        "route_variant_id": args.route_variant_id,
        "ego_id": args.ego_id,
        "sumo_cfg_file": "",
        "sumo_net_file": "",
        "sumo_rou_file": "",
        "sumo_launch_method": "net_and_route_files",
        "sumo_cfg_effective": False,
        "output_prefix": str(output_prefix),
        "output_jsonl_path": context_log_path,
    }
    if density_generation_info:
        config.update(density_generation_info)
        # Keep generated_sumo_rou_file as the final effective route file used by SUMO.
        # The density-only intermediate file is recorded separately as
        # density_generated_sumo_rou_file.
        config["generated_sumo_rou_file"] = str(generated_rou_file) if generated_rou_file else ""
    return config


def write_domain_manifest(manifest_path: Path, args: argparse.Namespace, domain_config: dict) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "timestamp": datetime.strftime(datetime.now(), "%Y-%m-%d_%H-%M-%S"),
        "domain_id": domain_config["domain_id"],
        "scene_config_id": domain_config["scene_config_id"],
        "episode_id": domain_config["episode_id"],
        "command_line_args": vars(args),
        "map_id": domain_config["map_id"],
        "route_id": domain_config["route_id"],
        "route_variant_id": domain_config["route_variant_id"],
        "ego_id": domain_config["ego_id"],
        "sumo_cfg_file": domain_config["sumo_cfg_file"],
        "sumo_net_file": domain_config["sumo_net_file"],
        "sumo_rou_file": domain_config["sumo_rou_file"],
        "sumo_launch_method": domain_config["sumo_launch_method"],
        "sumo_cfg_effective": domain_config["sumo_cfg_effective"],
        "sumo_seed": domain_config["sumo_seed"],
        "sumo_seed_effective": domain_config["sumo_seed_effective"],
        "traffic_density_id": domain_config["traffic_density_id"],
        "traffic_density_scale": domain_config["traffic_density_scale"],
        "traffic_density_effective": domain_config["traffic_density_effective"],
        "traffic_density_method": domain_config["traffic_density_method"],
        "base_sumo_rou_file": domain_config["base_sumo_rou_file"],
        "generated_sumo_rou_file": domain_config["generated_sumo_rou_file"],
        "density_generated_sumo_rou_file": domain_config.get("density_generated_sumo_rou_file", ""),
        "density_regular_period_seconds": domain_config.get("density_regular_period_seconds", ""),
        "density_effective_period_seconds": domain_config.get("density_effective_period_seconds", ""),
        "density_scale_integer": domain_config.get("density_scale_integer", ""),
        "density_base_vehicle_count": domain_config.get("density_base_vehicle_count", ""),
        "density_generated_vehicle_count": domain_config.get("density_generated_vehicle_count", ""),
        "density_generated_median_depart_gap_seconds": domain_config.get("density_generated_median_depart_gap_seconds", ""),
        "density_generated_effective_period_seconds": domain_config.get("density_generated_effective_period_seconds", ""),
        "density_generated_max_depart_schedule_error_seconds": domain_config.get("density_generated_max_depart_schedule_error_seconds", ""),
        "traffic_route_seed": domain_config.get("traffic_route_seed", ""),
        "randomTrips_period_seconds": domain_config.get("randomTrips_period_seconds", ""),
        "randomTrips_begin_seconds": domain_config.get("randomTrips_begin_seconds", ""),
        "randomTrips_end_seconds": domain_config.get("randomTrips_end_seconds", ""),
        "frozen_background_route_file": domain_config.get("frozen_background_route_file", ""),
        "frozen_background_route_sha256": domain_config.get("frozen_background_route_sha256", ""),
        "frozen_background_manifest_file": domain_config.get("frozen_background_manifest_file", ""),
        "frozen_background_manifest_sha256": domain_config.get("frozen_background_manifest_sha256", ""),
        "density_generated_route_sha256": domain_config.get("density_generated_route_sha256", ""),
        "reserved_background_vehicle_id": domain_config.get("reserved_background_vehicle_id", ""),
        "background_vehicle_removed_for_ego": domain_config.get("background_vehicle_removed_for_ego", False),
        "background_vehicle_count_before_reservation": domain_config.get("background_vehicle_count_before_reservation", ""),
        "background_vehicle_count_after_reservation": domain_config.get("background_vehicle_count_after_reservation", ""),
        "background_vehicle_count": domain_config.get("background_vehicle_count", ""),
        "ego_injected_separately": domain_config.get("ego_injected_separately", False),
        "ego_depart": domain_config.get("ego_depart", ""),
        "ego_depart_preserved_across_density": domain_config.get("ego_depart_preserved_across_density", False),
        "ego_route_edges": domain_config.get("ego_route_edges", ""),
        "ego_type": domain_config.get("ego_type", ""),
        "weather_id": domain_config["weather_id"],
        "weather_level": domain_config["weather_level"],
        "weather_params": domain_config["weather_params"],
        "weather_prompt_effective": domain_config["weather_prompt_effective"],
        "weather_sim_effective": domain_config["weather_sim_effective"],
        "weather_effective_type": domain_config["weather_effective_type"],
        "output_prefix": domain_config["output_prefix"],
        "output_jsonl_path": domain_config["output_jsonl_path"],
        "policy_provenance": domain_config.get("policy_provenance", {}),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


class LLMAgent:
    def __init__(
        self,
        use_memory: bool = True,
        delimiter: str = "####",
        model_name: str = DRIVING_POLICY_MODEL,
        max_tokens: int = 2000,
        max_retries: int = 3,
        retry_sleep: float = 10.0,
        max_retry_sleep: float = 20.0,
    ) -> None:
        """
        Initialize the LLMAgent_closeloop object.

        Args:
            use_memory (bool, optional): Flag indicating whether to use memory. Defaults to True.
            delimiter (str, optional): Delimiter string. Defaults to "####".
        """

        """ 
        For azure user, if you want to use azure key in this project, you need to check if your azure has deployed embedding model. You need to set your environment as follow and modify the code in file "simInfo/Memory.py" and "simInfo/Reflection.py":
        export OPENAI_API_TYPE="azure"
        export OPENAI_API_KEY='your azure key'
        export OPENAI_API_BASE="your azure node"
        export OPENAI_API_VERSION="your api version"
        export EMBEDDING_MODEL="your embedding model"
        """
        # oai_api_type = os.getenv("OPENAI_API_TYPE")
        # if oai_api_type == "azure":
        #     print("Using Azure Chat API")
        #     self.llm = AzureChatOpenAI(
        #         deployment_name="GPT-16",
        #         temperature=0.5,
        #         max_tokens=2000,
        #         request_timeout=60,
        #     )
        # elif oai_api_type == "openai":
        #     self.llm = ChatOpenAI(
        #         temperature=0,
        #         model_name= 'gpt-4-1106-preview',
        #         max_tokens=2000,
        #         request_timeout=60,
        #     )

        # openai key
        self.llm = ChatOpenAI(
            temperature=0,
            model_name=model_name,
            max_tokens=max_tokens,
            request_timeout=60,
            max_retries=0,
        )
        db_path = os.path.dirname(os.path.abspath(__file__)) + "/db/" + "terrible_decision_mem/" # the path for the memory database
        self.agent_memory = DrivingMemory(db_path=db_path)
        self.few_shot_num = 3

        self.use_memory = use_memory
        self.delimiter = delimiter
        self.model_name = model_name
        self.max_retries = max_retries
        self.retry_sleep = retry_sleep
        self.max_retry_sleep = max_retry_sleep
        self.last_invoke_attempts = 0
        self.last_invoke_retry_attempts = 0

        self.llm_source = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost": 0
        }

    @staticmethod
    def _is_non_retryable_llm_error(exc: Exception) -> bool:
        error_text = exception_text(exc).lower()
        return (
            is_quota_or_billing_error(exc)
            or "request headers are too large" in error_text
            or "request_headers_too_large" in error_text
        )

    @staticmethod
    def _is_retryable_llm_error(exc: Exception) -> bool:
        error_text = exception_text(exc).lower()
        retryable_patterns = [
            "ratelimiterror",
            "rate limit",
            "apierror",
            "api error",
            "timeout",
            "timeouterror",
            "request timed out",
            "read timed out",
            "connection",
            "connectionerror",
            "apiconnectionerror",
            "temporarily unavailable",
            "service unavailable",
            "server error",
            "502",
            "503",
            "504",
        ]
        return any(pattern in error_text for pattern in retryable_patterns)

    def invoke_llm_with_retry(self, messages):
        max_attempts = max(1, self.max_retries)
        self.last_invoke_attempts = 0
        self.last_invoke_retry_attempts = 0
        for attempt in range(1, max_attempts + 1):
            self.last_invoke_attempts = attempt
            try:
                return self.llm.invoke(messages)
            except Exception as exc:
                if self._is_non_retryable_llm_error(exc):
                    LLM_logger.error(
                        "Non-retryable LLM error; failing cleanly without retries: %s",
                        exc,
                    )
                    raise
                if not self._is_retryable_llm_error(exc) or attempt == max_attempts:
                    raise
                self.last_invoke_retry_attempts += 1
                base_sleep = max(0.1, self.retry_sleep)
                max_sleep = max(0.0, self.max_retry_sleep)
                jitter = random.uniform(0.0, min(1.0, base_sleep))
                sleep_seconds = min(max_sleep, base_sleep * (2 ** (attempt - 1)) + jitter)
                print(
                    f"[llm-retry] attempt={attempt}/{max_attempts} "
                    f"sleep={sleep_seconds:.1f}s error={type(exc).__name__}"
                )
                time.sleep(sleep_seconds)

    def getLLMSource(self, cb: OpenAICallbackHandler, init: bool):
        """Calculate the time, money and token spent per call to LLM interface 

        Args:
            cb (OpenAICallbackHandler): callback handler of the LLM interface
            init (bool): whether the call is the first call
        """
        if init:
            self.llm_source["prompt_tokens"] = cb.prompt_tokens
            self.llm_source["completion_tokens"] = cb.completion_tokens
            self.llm_source["total_tokens"] = cb.total_tokens
            self.llm_source["cost"] = cb.total_cost
        else:
            self.llm_source["prompt_tokens"] += cb.prompt_tokens
            self.llm_source["completion_tokens"] += cb.completion_tokens
            self.llm_source["total_tokens"] += cb.total_tokens
            self.llm_source["cost"] += cb.total_cost
        return 

    def makeDecision(self, descriptions: list = None) -> Tuple[int, str, str, str, dict]:
        """Interaction with LLM through rule-based scenario descriptions and pre-written prompt allows LLM to make decisions based on known information. When using memory module, it will extract the memory most similar to the current scene and add it to prompt.
        Components of Prompt:
            ```
            system_message
            example_message
            example_answer
            fewshot_messages, including the message and answer of the relative memories (if use_memory)
            human_message, including the current scenario description, navigation instruction and available actions
            ```

        Args:
            descriptions (list, optional): the description of the current scenario, current naviagtion information and current available action. Defaults to None.

        Raises:
            ActionParseException: When the single policy response has no unambiguous valid action.

        Returns:
            Tuple[int, str, str, str, dict]: the decision action, the response from LLM, the promot for the current scenario, the few shot composed of memories and the LLM source information.
        """

        # step1. get the scenario description
        scenario_description = descriptions[0]
        available_actions = descriptions[1]
        driving_intensions = descriptions[2]
        rewards = descriptions[3]
        previous_prompts = descriptions[4]
        previous_decisions = descriptions[5]
        short_prompt = descriptions[6]
        experience_buffer_limit = descriptions[7]
        displayed_action_ids = [int(action) for action in descriptions[8]]
        action_contract_repair_attempts = int(descriptions[9])
        if (
            not displayed_action_ids
            or len(displayed_action_ids) != len(set(displayed_action_ids))
            or any(action not in VALID_ACTION_IDS for action in displayed_action_ids)
        ):
            raise ValueError(
                f"Invalid displayed action set: {displayed_action_ids}"
            )
        if action_contract_repair_attempts not in {0, 1}:
            raise ValueError(
                "action_contract_repair_attempts must be either 0 or 1"
            )

        # step2. get the few shot examples
        if self.use_memory:
            key = "## Driving scenario description:\n" + scenario_description + "\n## Navigation instruction:\n" + driving_intensions
            key = key.replace("'", "")
            fewshot_results = self.agent_memory.retriveMemory(
                key, self.few_shot_num)
            fewshot_messages = []
            fewshot_answers = []
            fewshot_actions = []
            fewshot_comment = []
            for fewshot_result in fewshot_results:
                fewshot_messages.append(fewshot_result["human_question"])
                fewshot_answers.append(fewshot_result["LLM_response"])
                fewshot_comment.append(fewshot_result["comments"])
                fewshot_actions.append(fewshot_result["action"])

            print("the action in few_shot is: ", fewshot_actions)

            if len(fewshot_actions) == 0:
                LLM_logger.warning("There is no memory!")
        else:
            fewshot_messages = []

        # step3. get system message and make prompt
        with open(os.path.dirname(os.path.abspath(__file__)) + "/simInfo/system_message.txt", "r") as f:
            system_message = f.read()

        with open(os.path.dirname(os.path.abspath(__file__)) + "/simInfo/example_QA.txt", "r") as f:
            example = f.read()
            example_message = example.split("======")[0]
            example_answer = example.split("======")[1]

        human_message = DRIVING_POLICY_PROMPT_TEMPLATE.format(
            scenario_description=scenario_description,
            driving_intensions=driving_intensions,
            available_actions=available_actions,
            displayed_action_ids=displayed_action_ids,
        )

        messages = [
            SystemMessage(content=system_message.format(delimiter = self.delimiter)),
            HumanMessage(content=example_message),
            AIMessage(content=example_answer),
        ]

        # add few shot examples to the prompt
        if len(fewshot_messages) > 0:
            messages.append(HumanMessage(content="----------\nHere are the results of good decisions you have made in the sane situation in the past, please observe them carefully and take them into full consideration when making your current decision!\n"))
            for i in range(len(fewshot_messages)):
                messages.append(
                    HumanMessage(content=fewshot_messages[len(fewshot_messages) - 1 - i])
                )
                messages.append(
                    AIMessage(content=fewshot_answers[len(fewshot_messages)- 1 - i])
                )
                if fewshot_comment[i] != "":
                    messages.append(
                        AIMessage(content=f"Here are some suggestions when driving in this scenario:\n"+fewshot_comment[len(fewshot_messages)-1-i])
                    )
                messages.append(HumanMessage(content="----------\nAbove messages are some examples of how you make a decision successfully in the past and some advise. Those scenarios are similar to the current scenario. You should make your decisions with full reference to past cases of correct decision making! You should discuss prior experience in your current reasoning."))

        messages.append(
            HumanMessage(content=human_message)
        )
        prompt_debug_metadata = {
            "prompt_num_messages": len(messages),
            "prompt_num_chars": sum(len(str(getattr(message, "content", ""))) for message in messages),
            "exact_messages": serialized_messages(messages),
            "exact_current_human_prompt": human_message,
            "rendered_messages_sha256": sha256_text(json.dumps(
                serialized_messages(messages), sort_keys=True, separators=(",", ":")
            )),
            "current_human_prompt_sha256": sha256_text(human_message),
            "model_requested": self.model_name,
            "model_returned": "",
            "previous_prompts_count": sum(1 for prompt in previous_prompts if str(prompt or "").strip()),
            "previous_decisions_count": sum(
                1
                for decision in previous_decisions
                if str(decision).strip() not in {"", "-1"}
            ),
            "experience_buffer_limit": experience_buffer_limit,
            "displayed_action_ids": list(displayed_action_ids),
        }

        # step4. get the response
        with get_openai_callback() as cb:
            try:
                response = self.invoke_llm_with_retry(messages)
            finally:
                self.getLLMSource(cb, True)
        prompt_debug_metadata["model_returned"] = response_model_id(response)

        # Step 5: parse the one policy response locally. No paid output-checker call is allowed.
        call_metadata = {
            "policy_api_call_count": 1,
            "policy_api_request_attempts": int(self.last_invoke_attempts),
            "policy_api_retry_attempts": int(self.last_invoke_retry_attempts),
            "action_contract_repair_count": 0,
            "initial_invalid_action": "",
            "action_parse_method": "",
            "action_parse_fallback_used": False,
            "action_parse_error": "",
            "reward_parse_method": "",
            "reward_parse_error": "",
        }
        try:
            result, parse_method, parse_fallback = parse_policy_action(response.content, self.delimiter)
            call_metadata.update(
                action_parse_method=parse_method,
                action_parse_fallback_used=bool(parse_fallback),
            )
            if result not in displayed_action_ids:
                if action_contract_repair_attempts == 0:
                    raise ActionParseException(
                        f"Parsed action {result} was not displayed to the policy",
                        {
                            "action_parse_method": parse_method,
                            "action_parse_fallback_used": bool(parse_fallback),
                            "action_parse_error": "action_not_in_displayed_set",
                            "parsed_action": result,
                            "displayed_action_ids": displayed_action_ids,
                        },
                    )

                call_metadata.update(
                    action_contract_repair_count=1,
                    initial_invalid_action=result,
                )
                repair_instruction = textwrap.dedent(f"""\
                Your previous response selected Action_id {result}, but that action
                is not available in the unchanged current state. Discard that
                decision. Re-evaluate the same observation and choose exactly one
                legal Action_id from {displayed_action_ids}. Do not choose or
                discuss any unavailable action. Return the complete required
                response, including exactly one finite numeric FINAL REWARD.
                """)
                repair_messages = messages + [
                    AIMessage(content=response.content),
                    HumanMessage(content=repair_instruction),
                ]
                with get_openai_callback() as repair_cb:
                    try:
                        response = self.invoke_llm_with_retry(repair_messages)
                    finally:
                        self.getLLMSource(repair_cb, False)
                prompt_debug_metadata["model_returned"] = response_model_id(response)
                prompt_debug_metadata["repair_messages"] = serialized_messages(
                    repair_messages
                )
                call_metadata["policy_api_call_count"] += 1
                call_metadata["policy_api_request_attempts"] += int(
                    self.last_invoke_attempts
                )
                call_metadata["policy_api_retry_attempts"] += int(
                    self.last_invoke_retry_attempts
                )
                result, parse_method, parse_fallback = parse_policy_action(
                    response.content,
                    self.delimiter,
                )
                call_metadata.update(
                    action_parse_method=parse_method,
                    action_parse_fallback_used=bool(parse_fallback),
                )
                if result not in displayed_action_ids:
                    raise ActionParseException(
                        (
                            f"Parsed action {result} was not displayed to the "
                            "policy after one action-contract repair"
                        ),
                        {
                            "action_parse_method": parse_method,
                            "action_parse_fallback_used": bool(parse_fallback),
                            "action_parse_error": (
                                "action_not_in_displayed_set_after_repair"
                            ),
                            "parsed_action": result,
                            "initial_invalid_action": call_metadata[
                                "initial_invalid_action"
                            ],
                            "displayed_action_ids": displayed_action_ids,
                        },
                    )
            returned_reward = parse_policy_reward(response.content)
            call_metadata["reward_parse_method"] = "exact_final_reward_line"
        except ActionParseException as exc:
            failure_metadata = dict(call_metadata)
            failure_metadata.update(exc.metadata)
            if isinstance(exc, RewardParseException):
                failure_metadata.update(
                    action_parse_method=call_metadata["action_parse_method"],
                    action_parse_fallback_used=call_metadata["action_parse_fallback_used"],
                    action_parse_error="",
                )
            else:
                action_parse_error = str(
                    failure_metadata.get("action_parse_error", "")
                ).strip()
                if action_parse_error:
                    failure_metadata.update(
                        action_parse_method=failure_metadata.get(
                            "action_parse_method"
                        ) or "failed",
                        action_parse_fallback_used=bool(
                            failure_metadata.get(
                                "action_parse_fallback_used",
                                False,
                            )
                        ),
                        action_parse_error=action_parse_error,
                    )
                else:
                    failure_metadata.update(
                        action_parse_method="failed",
                        action_parse_fallback_used=True,
                        action_parse_error=(
                            "unambiguous_valid_action_not_found"
                        ),
                    )
            failure_metadata.update(
                {
                    "response": response.content,
                    "llm_source": dict(self.llm_source),
                    "prompt_debug_metadata": dict(prompt_debug_metadata),
                }
            )
            exc.metadata = failure_metadata
            raise
        prompt_debug_metadata.update(call_metadata)

        if experience_buffer_limit is not None and len(rewards) >= experience_buffer_limit:
            rewards.pop()
        rewards.insert(0, returned_reward)
        if experience_buffer_limit is not None and len(previous_prompts) >= experience_buffer_limit:
            previous_prompts.pop()
        previous_prompts.insert(0, short_prompt)
        if experience_buffer_limit is not None and len(previous_decisions) >= experience_buffer_limit:
            previous_decisions.pop()
        previous_decisions.insert(0, result)

        few_shot_store = ""
        for i in range(len(fewshot_messages)):
            few_shot_store += "Human question: \n" + fewshot_messages[i] + "\nResponse:\n" + fewshot_answers[i] + "\nSuggestions:\n" + fewshot_comment[i] + \
                "\n---------------\n"
        print("Result:", result)
        
        if result not in VALID_ACTION_IDS:
            raise ActionParseException(f"Parsed action is outside the valid action set: {result}")
        LLM_logger.info(f"Output is available, the decision is {result}")
            
        LLM_logger.info(f"Tokens Used: {self.llm_source['total_tokens']}\n"
            f"\tPrompt Tokens: {self.llm_source['prompt_tokens']}\n"
            f"\tCompletion Tokens: {self.llm_source['completion_tokens']}\n"
            f"Total Cost (USD): ${self.llm_source['cost']}")
        
        return result, response.content, human_message, few_shot_store, self.llm_source, rewards, prompt_debug_metadata

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--deadlock-termination-mode",
        choices=["raise", "log_only"],
        default="raise",
        help="Use original hard deadlock termination, or only log potential deadlocks and continue.",
    )
    parser.add_argument(
        "--deadlock-warning-window",
        type=int,
        default=10,
        help="Original loop deadlock window. Formal runs require 10.",
    )
    parser.add_argument(
        "--deadlock-speed-threshold",
        type=float,
        default=0.1,
        help="Speed threshold for potential deadlock detection.",
    )
    parser.add_argument(
        "--experience-buffer-limit",
        type=int,
        default=20,
        help="Number of most recent valid decision-reward history chunks. Must be 20.",
    )
    parser.add_argument(
        "--action-contract-repair-attempts",
        type=int,
        choices=[0, 1],
        default=1,
        help=(
            "Retry the unchanged decision once when the model selects an action "
            "that was not displayed. The invalid response is never executed or "
            "written to history."
        ),
    )
    parser.add_argument(
        "--shield-horizon",
        type=float,
        default=5.0,
        help="Deterministic action-safety preview horizon in seconds.",
    )
    parser.add_argument(
        "--shield-dt",
        type=float,
        default=0.1,
        help="Deterministic action-safety preview time step in seconds.",
    )
    parser.add_argument(
        "--shield-longitudinal-margin",
        type=float,
        default=0.5,
        help="Longitudinal collision-box inflation per vehicle in meters.",
    )
    parser.add_argument(
        "--shield-lateral-margin",
        type=float,
        default=0.25,
        help="Lateral collision-box inflation per vehicle in meters.",
    )
    parser.add_argument(
        "--disable-safety-shield",
        action="store_true",
        help=(
            "Disable trajectory filtering and shield overrides. The policy "
            "sees the environment's complete base action set and its parsed "
            "action is executed directly."
        ),
    )
    parser.add_argument(
        "--llm-max-tokens",
        type=int,
        default=2000,
        help="Maximum completion tokens for the LLM call.",
    )
    parser.add_argument(
        "--llm-model",
        default=DRIVING_POLICY_MODEL,
        help="Exact model snapshot used by the closed-loop driving policy.",
    )
    parser.add_argument(
        "--strict-policy-protocol",
        action="store_true",
        help="Validate the driving policy, model snapshot, and campaign settings.",
    )
    parser.add_argument(
        "--campaign-id",
        default="",
        help="Frozen collection campaign identifier written to every artifact.",
    )
    parser.add_argument(
        "--policy-protocol-id",
        default=DRIVING_POLICY_PROTOCOL_ID,
        help="Frozen driving-policy protocol identifier.",
    )
    parser.add_argument(
        "--llm-max-retries",
        type=int,
        default=3,
        help="Outer retries for transient LLM errors such as TPM/RPM rate limits.",
    )
    parser.add_argument(
        "--llm-retry-sleep",
        type=float,
        default=10.0,
        help="Base sleep in seconds between outer LLM retries.",
    )
    parser.add_argument(
        "--llm-max-retry-sleep",
        type=float,
        default=20.0,
        help="Maximum sleep in seconds for one outer LLM retry wait.",
    )
    parser.add_argument(
        "--llm-call-sleep",
        type=float,
        default=0.0,
        help="Sleep after each successful LLM decision to avoid TPM limits.",
    )
    parser.add_argument(
        "--max-decisions",
        type=int,
        default=None,
        help="Stop cleanly after N completed LLM decisions. Intended for cheap preflight runs.",
    )
    parser.add_argument(
        "--episode-id",
        default=None,
        help="Episode identifier written to context logs. Defaults to the run timestamp.",
    )
    parser.add_argument(
        "--scenario-id",
        default="formal_candidate",
        help="Scenario identifier written to context logs.",
    )
    parser.add_argument(
        "--domain-id",
        default="clear_1x",
        choices=ALL_FORMAL_DOMAINS,
        help="Formal 12-domain identifier.",
    )
    parser.add_argument(
        "--sumo-seed",
        type=int,
        default=0,
        help="SUMO seed passed to traci.start as --seed and written to logs.",
    )
    parser.add_argument(
        "--traffic-density-scale",
        type=float,
        default=None,
        help="Optional override check for the domain preset traffic density scale.",
    )
    parser.add_argument(
        "--traffic-density-id",
        default=None,
        help="Optional override check for the domain preset traffic density id.",
    )
    parser.add_argument(
        "--density-route-protocol",
        choices=["legacy_repeated_routes", "frozen_randomtrips_background"],
        default="legacy_repeated_routes",
        help=(
            "Traffic-demand construction protocol. The default preserves prior "
            "results. frozen_randomtrips_background reads a pre-generated "
            "period-scaled background and injects the original ego once."
        ),
    )
    parser.add_argument(
        "--density-background-dir",
        default=None,
        help=(
            "Directory containing <Map>_background_<density>_seed<seed>.rou.xml. "
            "Required by frozen_randomtrips_background unless the standard "
            "networkFiles/<map>/frozen_randomtrips_density_seed<seed> path exists."
        ),
    )
    parser.add_argument(
        "--traffic-route-seed",
        type=int,
        default=42,
        help="Frozen randomTrips demand seed; distinct from --sumo-seed.",
    )
    parser.add_argument(
        "--randomtrips-begin-seconds",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--randomtrips-end-seconds",
        type=float,
        default=3600.0,
    )
    parser.add_argument(
        "--weather-id",
        default=None,
        help="Optional override check for the domain preset weather id.",
    )
    parser.add_argument("--map-id", default="Town05")
    parser.add_argument("--ego-id", default=None)
    parser.add_argument("--route-id", default=None)
    parser.add_argument("--route-variant-id", default=None)
    parser.add_argument("--sumo-net-file", default=None)
    parser.add_argument("--sumo-cfg-file", default=None)
    parser.add_argument("--base-sumo-rou-file", default=None)
    parser.add_argument("--base-vtypes-file", default=None)
    parser.add_argument(
        "--formal-run",
        action="store_true",
        help="Enable strict formal full-episode-history evaluation checks.",
    )
    parser.add_argument(
        "--sumo-gui",
        action="store_true",
        help="Launch SUMO-GUI for visual monitoring. Defaults to headless SUMO.",
    )
    parser.add_argument(
        "--ego-depart-lane-mode",
        choices=[
            "none",
            "first_strict_noncompatible",
            "first_structural_noncompatible",
            "first_structural_nearest_noncompatible",
        ],
        default="none",
        help=(
            "Integrity-safe route initial-condition mode. "
            "first_strict_noncompatible sets only the ego SUMO departLane to a lane "
            "that is not route-compatible for the first route transition, so the "
            "original navigation logic naturally asks for lane change. It does not "
            "change the LLM prompt, action set, reward, or policy code. "
            "first_structural_noncompatible propagates the forced depart lane through "
            "earlier unconstrained transitions and targets the first later structural "
            "lane constraint. first_structural_nearest_noncompatible uses the same "
            "structural target but starts from the nearest route-incompatible lane, "
            "requiring the minimum non-zero number of lane changes."
        ),
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Optional output prefix for the DB and context JSONL. Defaults to results/<timestamp>.",
    )
    parser.add_argument(
        "--context-policy",
        choices=["full_h20", "adaptive_h20"],
        required=True,
        help="Include up to the 20 most recent valid history chunks.",
    )
    parser.add_argument("--pair-id", required=True, help="Stable route+domain+seed paired-run identifier.")
    parser.add_argument("--selector-device", default="cpu")
    parser.add_argument(
        "--final-adaptive-selector-dir",
        default="results/h20_selector",
        help="Adaptive H20 selector artifact JSON path or directory.",
    )
    args = parser.parse_args()

    adaptive_selector = None
    if args.context_policy == "adaptive_h20":
        adaptive_selector = FinalAdaptiveH20Selector(
            args.final_adaptive_selector_dir,
            args.selector_device,
        )

    if args.experience_buffer_limit != 20:
        raise ValueError(
            "The H=20 diagnostic requires --experience-buffer-limit 20."
        )
    if args.shield_horizon <= 0 or args.shield_dt <= 0:
        raise ValueError("Safety-shield horizon and dt must be positive.")
    if (
        args.shield_longitudinal_margin < 0
        or args.shield_lateral_margin < 0
    ):
        raise ValueError("Safety-shield margins cannot be negative.")
    if args.formal_run:
        if args.deadlock_warning_window != 10:
            raise ValueError(
                "Formal runs require --deadlock-warning-window 10 to match "
                "the released ICRL loop."
            )
        if not math.isclose(
            args.deadlock_speed_threshold,
            0.1,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "Formal runs require --deadlock-speed-threshold 0.1 to match "
                "the released ICRL loop."
            )
    if args.strict_policy_protocol:
        if not args.formal_run:
            raise ValueError("--strict-policy-protocol requires --formal-run")
        if args.llm_model != DRIVING_POLICY_MODEL:
            raise ValueError(
                f"Driving policy protocol requires --llm-model {DRIVING_POLICY_MODEL}"
            )
        if args.policy_protocol_id != DRIVING_POLICY_PROTOCOL_ID:
            raise ValueError(
                "Driving policy protocol mismatch: "
                f"{args.policy_protocol_id!r}"
            )
        if not str(args.campaign_id).strip():
            raise ValueError("Driving policy protocol requires a non-empty --campaign-id")
        if args.action_contract_repair_attempts != 1:
            raise ValueError("Driving policy protocol requires exactly one repair attempt")
        if not args.disable_safety_shield:
            raise ValueError("Driving policy protocol requires --disable-safety-shield")
        if args.deadlock_termination_mode != "log_only":
            raise ValueError(
                "Driving policy protocol requires --deadlock-termination-mode log_only"
            )

    route_entry = find_route_entry(args.map_id, args.route_id, args.ego_id)
    route_paths = require_route_entry_available(route_entry)
    args.route_id = args.route_id or route_entry.route_id
    args.route_variant_id = args.route_variant_id or route_entry.route_variant_id
    args.ego_id = args.ego_id or route_entry.ego_id

    ego_id = args.ego_id
    sumo_gui = args.sumo_gui
    sumo_cfg_file = args.sumo_cfg_file or str(route_paths["sumo_cfg_file"])
    sumo_net_file = args.sumo_net_file or str(route_paths["sumo_net_file"])
    base_vtypes_file = (
        Path(args.base_vtypes_file).expanduser().resolve()
        if args.base_vtypes_file
        else route_paths["base_vtypes_file"]
    )
    base_sumo_rou_file = (
        Path(args.base_sumo_rou_file).expanduser().resolve()
        if args.base_sumo_rou_file
        else route_paths["base_sumo_rou_file"]
    )
    for required_path in [Path(sumo_cfg_file), Path(sumo_net_file), base_vtypes_file, base_sumo_rou_file]:
        if not required_path.exists():
            raise FileNotFoundError(f"Required map/route file is missing: {required_path}")
    sumo_rou_file = f"{base_vtypes_file},{base_sumo_rou_file}"
    carla_host = '127.0.0.1'
    carla_port = 2000
    step_length = 0.1
    tls_manager = 'sumo'
    sync_vehicle_color = True
    sync_vehicle_lights = True

    stringTimestamp = datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M-%S')
    episode_id = args.episode_id or stringTimestamp
    args.episode_id = episode_id
    output_prefix = Path(args.output_prefix) if args.output_prefix else Path("results") / stringTimestamp
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    database = str(output_prefix.with_suffix(".db"))
    context_log_path = str(output_prefix.parent / f"{output_prefix.name}_paired_context_log.jsonl")
    episode_status_path = output_prefix.parent / f"{output_prefix.name}_episode_status.json"
    selector_log_path = output_prefix.parent / f"{output_prefix.name}_per_decision_selector_log.csv"
    episode_metrics_path = output_prefix.parent / f"{output_prefix.name}_per_episode_metrics.csv"
    requested_domain = domain_metadata(args.domain_id)
    preset_density_scale = float(requested_domain["traffic_density_scale"])
    traffic_density_id = str(requested_domain["traffic_density_id"])
    generated_density_rou_file = None
    density_generation_info = {}
    effective_route_file = base_sumo_rou_file
    if args.density_route_protocol == "frozen_randomtrips_background":
        if args.randomtrips_end_seconds <= args.randomtrips_begin_seconds:
            raise ValueError(
                "--randomtrips-end-seconds must exceed --randomtrips-begin-seconds"
            )
        background_dir = (
            Path(args.density_background_dir).expanduser().resolve()
            if args.density_background_dir
            else (
                Path(sumo_net_file).resolve().parent
                / f"frozen_randomtrips_density_seed{args.traffic_route_seed}"
            )
        )
        background_rou_file = frozen_randomtrips_background_path(
            background_dir,
            args.map_id,
            traffic_density_id,
            args.traffic_route_seed,
        )
        generated_density_rou_file = (
            output_prefix.parent
            / "domain_routes"
            / (
                f"{episode_id}_{traffic_density_id}_randomtrips_seed"
                f"{args.traffic_route_seed}_{args.map_id}.rou.xml"
            )
        )
        density_generation_info = make_frozen_randomtrips_density_route_file(
            base_rou_file=base_sumo_rou_file,
            background_rou_file=background_rou_file,
            output_rou_file=generated_density_rou_file,
            map_id=args.map_id,
            net_file=Path(sumo_net_file).resolve(),
            vtypes_file=base_vtypes_file,
            ego_id=ego_id,
            traffic_density_id=traffic_density_id,
            density_scale=preset_density_scale,
            traffic_route_seed=args.traffic_route_seed,
            begin_seconds=args.randomtrips_begin_seconds,
            end_seconds=args.randomtrips_end_seconds,
        )
        effective_route_file = generated_density_rou_file.resolve()
    elif preset_density_scale != 1.0:
        generated_density_rou_file = (
            output_prefix.parent
            / "domain_routes"
            / f"{episode_id}_density{preset_density_scale:g}_{args.map_id}.rou.xml"
        )
        density_generation_info = make_formal_density_route_file(
            base_rou_file=base_sumo_rou_file,
            output_rou_file=generated_density_rou_file,
            ego_id=ego_id,
            density_scale=preset_density_scale,
            validate_against_scale=2.0 if preset_density_scale == 3.0 else None,
        )
        effective_route_file = generated_density_rou_file.resolve()

    lane_challenge_info = {}
    generated_lane_challenge_rou_file = None
    if args.ego_depart_lane_mode != "none":
        generated_lane_challenge_rou_file = (
            output_prefix.parent
            / "domain_routes"
            / f"{episode_id}_lanechallenge_{args.map_id}.rou.xml"
        )
        lane_challenge_info = make_ego_lane_challenge_route_file(
            input_rou_file=Path(effective_route_file),
            output_rou_file=generated_lane_challenge_rou_file,
            sumo_net_file=Path(sumo_net_file),
            ego_id=ego_id,
            mode=args.ego_depart_lane_mode,
        )
        effective_route_file = generated_lane_challenge_rou_file.resolve()

    sumo_rou_file = f"{base_vtypes_file},{Path(effective_route_file).resolve()}"
    route_generation_info = {}
    route_generation_info.update(density_generation_info)
    route_generation_info.update(lane_challenge_info)

    domain_config = build_domain_config(
        args,
        output_prefix,
        context_log_path,
        base_sumo_rou_file,
        Path(effective_route_file).resolve() if Path(effective_route_file).resolve() != base_sumo_rou_file else None,
        route_generation_info,
    )
    domain_config.update(
        {
            "scene_config_id": domain_config["scene_config_id"],
            "history_horizon": 20,
            "safety_shield_enabled": not args.disable_safety_shield,
            "safety_shield_horizon": args.shield_horizon,
            "safety_shield_dt": args.shield_dt,
            "safety_shield_longitudinal_margin": args.shield_longitudinal_margin,
            "safety_shield_lateral_margin": args.shield_lateral_margin,
            "sumo_cfg_file": resolved_path(sumo_cfg_file),
            "sumo_net_file": resolved_path(sumo_net_file),
            "sumo_rou_file": ",".join(resolved_path(path) for path in sumo_rou_file.split(",")),
        }
    )
    policy_provenance = {
        "campaign_id": args.campaign_id,
        "policy_protocol_id": args.policy_protocol_id,
        "strict_policy_protocol": bool(args.strict_policy_protocol),
        "model_requested": args.llm_model,
        "temperature": 0.0,
        "max_completion_tokens": args.llm_max_tokens,
        "system_message_sha256": sha256_path(
            Path(__file__).resolve().parent / "simInfo" / "system_message.txt"
        ),
        "example_qa_sha256": sha256_path(
            Path(__file__).resolve().parent / "simInfo" / "example_QA.txt"
        ),
        "prompt_template_sha256": sha256_text(DRIVING_POLICY_PROMPT_TEMPLATE),
        "runner_sha256": sha256_path(Path(__file__).resolve()),
        "env_descriptor_sha256": sha256_path(
            Path(__file__).resolve().parent / "simInfo" / "ICRLEnvDescriptor.py"
        ),
        "formal_domain_config_sha256": sha256_path(
            Path(__file__).resolve().parent / "formal_domain_config.py"
        ),
        "network_file_sha256": sha256_path(sumo_net_file),
        "traffic_route_file_sha256": sha256_path(effective_route_file),
        "route_id": args.route_id,
        "route_variant_id": args.route_variant_id,
        "sumo_seed": args.sumo_seed,
        "traffic_route_seed": args.traffic_route_seed,
        "context_policy": args.context_policy,
        "adaptive_selector_artifact_sha256": (
            adaptive_selector.artifact["artifact_sha256"]
            if adaptive_selector is not None else ""
        ),
        "experience_buffer_limit": args.experience_buffer_limit,
        "action_contract_repair_attempts": args.action_contract_repair_attempts,
        "safety_shield_enabled": not args.disable_safety_shield,
        "deadlock_mode": args.deadlock_termination_mode,
        "history_storage_order": HISTORY_STORAGE_ORDER,
        "history_render_order": HISTORY_RENDER_ORDER,
    }
    domain_config["policy_provenance"] = policy_provenance
    manifest_path = output_prefix.parent / f"{output_prefix.name}_domain_manifest.json"
    write_domain_manifest(manifest_path, args, domain_config)
    print(
        f"sumo_seed={args.sumo_seed} is effective via traci.start --seed."
    )
    print(
        f"traffic_density_scale={domain_config['traffic_density_scale']} "
        f"effective={domain_config['traffic_density_effective']} "
        f"method={domain_config['traffic_density_method']}"
    )
    print(
        f"weather_id={domain_config['weather_id']} is prompt-observed numeric weather; "
        "weather_sim_effective=False for this SUMO-only run path."
    )
    print(f"Wrote domain manifest: {manifest_path}")

    # step1. init LLM_Driver
    model = Model(
        egoID=ego_id, netFile=sumo_net_file, rouFile=sumo_rou_file,
        cfgFile=sumo_cfg_file, dataBase=database, SUMOGUI=sumo_gui,
        CARLACosim=False, carla_host=carla_host, carla_port=carla_port,
        sumo_seed=args.sumo_seed,
    )
    model.weather.update(domain_config["weather_params"])
    planner = TrafficManager(model)
    safety_shield = None
    if not args.disable_safety_shield:
        safety_shield = SafetyShield(
            horizon=args.shield_horizon,
            dt=args.shield_dt,
            longitudinal_margin=args.shield_longitudinal_margin,
            lateral_margin=args.shield_lateral_margin,
        )
    agent = LLMAgent(
        use_memory=False,
        model_name=args.llm_model,
        max_tokens=args.llm_max_tokens,
        max_retries=args.llm_max_retries,
        retry_sleep=args.llm_retry_sleep,
        max_retry_sleep=args.llm_max_retry_sleep,
    )
    descriptor = ICRLEnvDescription()
    collision_checker = CollisionChecker()
    model.start()
    experience_buffer_limit = args.experience_buffer_limit
    gui = GUI(model)
    gui.start()

    action_list = []
    rewards: list[Any] = []
    previous_prompts: list[Any] = []
    previous_decisions: list[Any] = []
    total_start_time = time.time()
    decision_index = 0
    selected_k_active: list[int] = []
    selected_k_all: list[int] = []
    selected_mask_counter: Counter[str] = Counter()
    selector_fallback_count = 0
    selector_latency_total = 0.0
    policy_latency_total = 0.0
    prompt_tokens_total = 0
    completion_tokens_total = 0
    total_tokens_total = 0
    api_cost_total = 0.0
    policy_api_call_count_total = 0
    policy_api_request_attempts_total = 0
    policy_api_retry_attempts_total = 0
    model_returned_ids: set[str] = set()
    idle_action_count = 0
    deadlock_indicator_count = 0
    shield_filtered_decision_count = 0
    shield_override_count = 0
    shield_forced_single_action_count = 0
    shield_no_safe_action_count = 0
    shield_raw_unsafe_action_count = 0
    shield_history_write_skipped_count = 0
    shield_prediction_truncated_decision_count = 0
    shield_evaluation_failure_count = 0
    episode_status = "failed"
    failure_stage = ""
    failure_error_type = ""
    failure_error_message = ""
    original_exception_class = ""
    original_exception_message = ""
    # step2. run the model, use LLM to make decision per second. PS: the unit of timeStep is 0.1s
    try:
        while not model.tpEnd:
            model.moveStep()
            collision_checker.CollisionCheck(model)
            if model.tpEnd:
                break
            if model.timeStep % 10 == 0:
                roadgraph, vehicles = model.exportSce()
                if model.tpStart and roadgraph:
                    LLM_logger.info(f"--------------- timestep is {model.timeStep} ---------------")
                    # 2.1 get current description based on rules
                    navInfo = descriptor.getNavigationInfo(roadgraph, vehicles)
                    base_action_ids = descriptor.getAvailableActionIds(
                        roadgraph,
                        vehicles,
                    )
                    if safety_shield is None:
                        shield_result = unshielded_result(base_action_ids)
                    else:
                        try:
                            shield_result = safety_shield.evaluate(
                                planner=planner,
                                sim_time=model.timeStep * step_length,
                                roadgraph=roadgraph,
                                vehicles_info=vehicles,
                                base_action_ids=base_action_ids,
                            )
                        except SafetyShieldEvaluationError:
                            failure_stage = "shield_evaluation_failure"
                            shield_evaluation_failure_count += 1
                            raise
                    shield_prediction_truncated_decision_count += int(
                        shield_result.any_prediction_truncated
                    )
                    displayed_action_ids = list(base_action_ids)
                    actionInfo = descriptor.formatAvailableActions(
                        displayed_action_ids
                    )
                    envInfo, short_prompt = descriptor.getEnvPrompt(
                        roadgraph, vehicles, model, rewards, previous_prompts, previous_decisions
                    )
                    full_envInfo = envInfo
                    h_indices = valid_history_indices(
                        previous_prompts,
                        previous_decisions,
                        rewards,
                    )
                    h_available = len(h_indices)
                    selector_metadata = {
                        "selected_mask": "1" * h_available,
                        "selected_mask_order": SELECTED_MASK_ORDER,
                        "selected_K": h_available,
                        "ensemble_mean_logkl": "",
                        "ensemble_std_logkl": "",
                        "selected_upper_bound": "",
                        "threshold_log1p_tau": (
                            adaptive_selector.threshold if adaptive_selector is not None else ""
                        ),
                        "fallback_to_full": False,
                        "selector_inference_seconds": 0.0,
                        "mask_scores": [],
                    }
                    selected_indices = h_indices
                    if adaptive_selector is not None and h_available == 20:
                        selector_record = {
                            "history_storage_order": HISTORY_STORAGE_ORDER,
                            "current_short_prompt": short_prompt,
                            "previous_prompts": list(previous_prompts),
                            "previous_decisions": list(previous_decisions),
                            "rewards": list(rewards),
                            "navigation": navInfo,
                            "actions": actionInfo,
                        }
                        selector_started = time.time()
                        adaptive_selection = adaptive_selector.select(selector_record)
                        # Runtime masks use canonical oldest-to-newest indices;
                        # live buffers are newest-to-oldest.
                        selected_indices = sorted(
                            19 - index for index in adaptive_selection.canonical_indices
                        )
                        selector_metadata.update({
                            "selected_mask": adaptive_selection.selected_mask,
                            "selected_K": adaptive_selection.selected_K,
                            "selected_upper_bound": (
                                adaptive_selection.selected_upper_bound
                                if adaptive_selection.selected_upper_bound is not None else ""
                            ),
                            "threshold_log1p_tau": adaptive_selection.threshold,
                            "fallback_to_full": adaptive_selection.selected_K == 20,
                            "selector_inference_seconds": time.time() - selector_started,
                            "mask_scores": list(adaptive_selection.steps),
                        })
                    decision_previous_prompts, decision_previous_decisions, decision_rewards = subset_buffers(
                        selected_indices, previous_prompts, previous_decisions, rewards
                    )
                    if adaptive_selector is not None and h_available == 20:
                        envInfo = compose_selected_env_prompt(
                            short_prompt,
                            selected_indices,
                            decision_previous_prompts,
                            decision_previous_decisions,
                            decision_rewards,
                        )
                    ego_speed = getattr(model.ms.ego, "speed", 0.0)
                    potential_last_action, potential_streak_len = recent_action_streak(action_list)
                    potential_deadlock = original_deadlock_condition(
                        action_list,
                        ego_speed,
                        window=args.deadlock_warning_window,
                        speed_threshold=args.deadlock_speed_threshold,
                    )
                    state_id = (
                        f"{domain_config['domain_id']}__{args.map_id}__route{args.route_id}"
                        f"__ego{ego_id}__seed{args.sumo_seed}__ep{episode_id}"
                        f"__frame{model.timeStep}"
                    )
                    context_log_record = {
                        "scene_config_id": domain_config["scene_config_id"],
                        "history_horizon": 20,
                        "domain_id": domain_config["domain_id"],
                        "split_group_candidate": domain_config["split_group_candidate"],
                        "episode_id": episode_id,
                        "scenario_id": args.scenario_id,
                        "map_id": args.map_id,
                        "route_id": args.route_id,
                        "route_variant_id": args.route_variant_id,
                        "ego_id": ego_id,
                        "sumo_seed": args.sumo_seed,
                        "sumo_seed_effective": domain_config["sumo_seed_effective"],
                        "traffic_density_id": domain_config["traffic_density_id"],
                        "traffic_density_scale": domain_config["traffic_density_scale"],
                        "traffic_density_effective": domain_config["traffic_density_effective"],
                        "traffic_density_method": domain_config["traffic_density_method"],
                        "base_sumo_rou_file": domain_config["base_sumo_rou_file"],
                        "generated_sumo_rou_file": domain_config["generated_sumo_rou_file"],
                        "weather_id": domain_config["weather_id"],
                        "weather_level": domain_config["weather_level"],
                        "weather_params": domain_config["weather_params"],
                        "weather_prompt_effective": domain_config["weather_prompt_effective"],
                        "weather_sim_effective": domain_config["weather_sim_effective"],
                        "weather_effective_type": domain_config["weather_effective_type"],
                        "sumo_cfg_file": domain_config["sumo_cfg_file"],
                        "sumo_net_file": domain_config["sumo_net_file"],
                        "sumo_rou_file": domain_config["sumo_rou_file"],
                        "sumo_launch_method": domain_config["sumo_launch_method"],
                        "sumo_cfg_effective": domain_config["sumo_cfg_effective"],
                        "output_prefix": domain_config["output_prefix"],
                        "state_id": state_id,
                        "pair_id": args.pair_id,
                        "context_policy": args.context_policy,
                        "decision_index": decision_index,
                        "frame": model.timeStep,
                        "time_step": model.timeStep,
                        "current_short_prompt": short_prompt,
                        "previous_prompts": list(previous_prompts),
                        "previous_decisions": list(previous_decisions),
                        "rewards": list(rewards),
                        "selected_previous_prompts": list(decision_previous_prompts),
                        "selected_previous_decisions": list(decision_previous_decisions),
                        "selected_rewards": list(decision_rewards),
                        "navigation": navInfo,
                        "actions": actionInfo,
                        "final_description": envInfo,
                        "full_final_description": full_envInfo,
                        "H_available": h_available,
                        **selector_metadata,
                        "deadlock_termination_mode": args.deadlock_termination_mode,
                        "potential_deadlock": potential_deadlock,
                        "potential_deadlock_streak_len": potential_streak_len,
                        "potential_deadlock_last_action": potential_last_action if potential_last_action is not None else -1,
                        "ego_speed": ego_speed,
                        **shield_result.audit_fields(),
                        "raw_llm_action": "",
                        "executed_action": "",
                        "shield_override": "",
                        "history_write_skipped_reason": "",
                        **policy_provenance,
                    }

                    start_time = time.time()
                    # 2.2 make decision by LLM
                    agent.llm_source = {
                        "prompt_tokens": 0, "completion_tokens": 0,
                        "total_tokens": 0, "cost": 0.0,
                    }
                    try:
                        ego_behaviour, response, human_question, fewshot, llm_cost, decision_rewards_after, prompt_debug_metadata = agent.makeDecision([
                            envInfo,
                            actionInfo,
                            navInfo,
                            list(decision_rewards),
                            list(decision_previous_prompts),
                            list(decision_previous_decisions),
                            short_prompt,
                            experience_buffer_limit,
                            list(displayed_action_ids),
                            args.action_contract_repair_attempts,
                        ])
                    except ActionParseException as exc:
                        failure_stage = (
                            "reward_parse"
                            if isinstance(exc, RewardParseException)
                            else "action_parse"
                        )
                        parse_metadata = dict(exc.metadata)
                        llm_source = dict(parse_metadata.get("llm_source", {}))
                        prompt_metadata = dict(parse_metadata.get("prompt_debug_metadata", {}))
                        context_log_record.update(
                            {
                                **prompt_metadata,
                                "choose_action": "",
                                "response": parse_metadata.get("response", ""),
                                "prompt_tokens": llm_source.get("prompt_tokens", ""),
                                "completion_tokens": llm_source.get("completion_tokens", ""),
                                "total_tokens": llm_source.get("total_tokens", ""),
                                "policy_api_call_count": parse_metadata.get("policy_api_call_count", 1),
                                "policy_api_request_attempts": parse_metadata.get("policy_api_request_attempts", ""),
                                "policy_api_retry_attempts": parse_metadata.get("policy_api_retry_attempts", ""),
                                "action_contract_repair_count": parse_metadata.get("action_contract_repair_count", 0),
                                "initial_invalid_action": parse_metadata.get("initial_invalid_action", ""),
                                "action_parse_method": parse_metadata.get("action_parse_method", "failed"),
                                "action_parse_fallback_used": parse_metadata.get("action_parse_fallback_used", True),
                                "action_parse_error": parse_metadata.get("action_parse_error", str(exc)),
                                "reward_parse_method": parse_metadata.get("reward_parse_method", ""),
                                "reward_parse_error": parse_metadata.get("reward_parse_error", ""),
                            }
                        )
                        append_selector_csv(
                            selector_log_path,
                            {
                                "pair_id": args.pair_id,
                                "episode_id": episode_id,
                                "route_id": args.route_id,
                                "domain_id": args.domain_id,
                                "sumo_seed": args.sumo_seed,
                                "context_policy": args.context_policy,
                                "adaptive_selector_artifact_sha256": policy_provenance.get(
                                    "adaptive_selector_artifact_sha256", ""
                                ),
                                "decision_index": decision_index,
                                "H_available": h_available,
                                **selector_metadata,
                                "actual_input_tokens": llm_source.get("prompt_tokens", ""),
                                "completion_tokens": llm_source.get("completion_tokens", ""),
                                "total_tokens": llm_source.get("total_tokens", ""),
                                "action_selected": "",
                                "driving_policy_api_seconds": time.time() - start_time,
                                "policy_api_call_count": parse_metadata.get("policy_api_call_count", 1),
                                "policy_api_request_attempts": parse_metadata.get("policy_api_request_attempts", ""),
                                "policy_api_retry_attempts": parse_metadata.get("policy_api_retry_attempts", ""),
                                "action_contract_repair_count": parse_metadata.get("action_contract_repair_count", 0),
                                "initial_invalid_action": parse_metadata.get("initial_invalid_action", ""),
                                "action_parse_method": parse_metadata.get("action_parse_method", "failed"),
                                "action_parse_fallback_used": parse_metadata.get("action_parse_fallback_used", True),
                                "action_parse_error": parse_metadata.get("action_parse_error", str(exc)),
                                "reward_parse_method": parse_metadata.get("reward_parse_method", ""),
                                "reward_parse_error": parse_metadata.get("reward_parse_error", ""),
                                **shield_log_fields(shield_result),
                            },
                        )
                        write_context_pool_log(context_log_path, context_log_record)
                        policy_api_call_count_total += int(parse_metadata.get("policy_api_call_count", 1))
                        policy_api_request_attempts_total += int(parse_metadata.get("policy_api_request_attempts", 0) or 0)
                        policy_api_retry_attempts_total += int(parse_metadata.get("policy_api_retry_attempts", 0) or 0)
                        raise
                    except Exception as exc:
                        failure_stage = "llm_invoke"
                        failed_call_metadata = {
                            "policy_api_call_count": 1,
                            "policy_api_request_attempts": int(agent.last_invoke_attempts),
                            "policy_api_retry_attempts": int(agent.last_invoke_retry_attempts),
                            "action_contract_repair_count": 0,
                            "initial_invalid_action": "",
                            "action_parse_method": "not_attempted",
                            "action_parse_fallback_used": False,
                            "action_parse_error": f"policy_request_failed: {type(exc).__name__}",
                            "reward_parse_method": "not_attempted",
                            "reward_parse_error": "",
                        }
                        context_log_record.update(failed_call_metadata)
                        append_selector_csv(
                            selector_log_path,
                            {
                                "pair_id": args.pair_id,
                                "episode_id": episode_id,
                                "route_id": args.route_id,
                                "domain_id": args.domain_id,
                                "sumo_seed": args.sumo_seed,
                                "context_policy": args.context_policy,
                                "adaptive_selector_artifact_sha256": policy_provenance.get(
                                    "adaptive_selector_artifact_sha256", ""
                                ),
                                "decision_index": decision_index,
                                "H_available": h_available,
                                **selector_metadata,
                                "actual_input_tokens": "",
                                "completion_tokens": "",
                                "total_tokens": "",
                                "action_selected": "",
                                "driving_policy_api_seconds": time.time() - start_time,
                                **failed_call_metadata,
                                **shield_log_fields(shield_result),
                            },
                        )
                        write_context_pool_log(context_log_path, context_log_record)
                        policy_api_call_count_total += 1
                        policy_api_request_attempts_total += int(agent.last_invoke_attempts)
                        policy_api_retry_attempts_total += int(agent.last_invoke_retry_attempts)
                        raise
                    finally:
                        prompt_tokens_total += int(agent.llm_source["prompt_tokens"])
                        completion_tokens_total += int(agent.llm_source["completion_tokens"])
                        total_tokens_total += int(agent.llm_source["total_tokens"])
                        api_cost_total += float(agent.llm_source.get("cost", 0.0) or 0.0)
                    policy_api_call_count_total += int(prompt_debug_metadata["policy_api_call_count"])
                    policy_api_request_attempts_total += int(prompt_debug_metadata["policy_api_request_attempts"])
                    policy_api_retry_attempts_total += int(prompt_debug_metadata["policy_api_retry_attempts"])
                    accepted_policy_calls = int(
                        prompt_debug_metadata.get("policy_api_call_count", 0)
                    )
                    returned_model = str(
                        prompt_debug_metadata.get("model_returned", "")
                    ).strip()
                    if (
                        args.strict_policy_protocol
                        and returned_model
                        and returned_model != DRIVING_POLICY_RETURNED_MODEL
                    ):
                        raise RuntimeError(
                            "Driving policy model identity mismatch: "
                            f"requested={args.llm_model!r} returned={returned_model!r} "
                            f"expected_returned={DRIVING_POLICY_RETURNED_MODEL!r}"
                        )
                    if returned_model:
                        model_returned_ids.add(returned_model)
                    repair_count = int(
                        prompt_debug_metadata.get(
                            "action_contract_repair_count",
                            0,
                        )
                    )
                    if accepted_policy_calls != 1 + repair_count:
                        raise RuntimeError(
                            "Accepted decisions require one initial policy call "
                            "plus each recorded action-contract repair"
                        )
                    if repair_count not in {0, 1}:
                        raise RuntimeError(
                            "Accepted decision has an invalid action-contract "
                            f"repair count: {repair_count}"
                        )
                    returned_reward = decision_rewards_after[0] if decision_rewards_after else ""
                    raw_llm_action = int(ego_behaviour)
                    executed_action, shield_override = (
                        shield_result.resolve_raw_action(raw_llm_action)
                    )
                    if not shield_override:
                        history_write_skipped_reason = ""
                        update_full_history(
                            previous_prompts,
                            previous_decisions,
                            rewards,
                            short_prompt,
                            executed_action,
                            returned_reward,
                            experience_buffer_limit,
                        )
                    else:
                        history_write_skipped_reason = (
                            "shield_override"
                        )
                    policy_seconds = time.time() - start_time
                    context_log_record.update({
                        **prompt_debug_metadata,
                        "choose_action": executed_action,
                        "raw_llm_action": raw_llm_action,
                        "executed_action": executed_action,
                        "shield_override": shield_override,
                        "history_write_skipped_reason": (
                            history_write_skipped_reason
                        ),
                        "response": response,
                        "prompt_tokens": llm_cost["prompt_tokens"],
                        "completion_tokens": llm_cost["completion_tokens"],
                        "total_tokens": llm_cost["total_tokens"],
                        "total_time": policy_seconds,
                        "decision_reward": returned_reward,
                    })

                    descriptor.decision = executed_action # update the last executed decision

                    # 2.3 
                    current_QA = QuestionAndAnswer(envInfo, navInfo, actionInfo, fewshot, response, llm_cost["prompt_tokens"], llm_cost["completion_tokens"], llm_cost["total_tokens"], time.time()-start_time, executed_action)

                    model.putQA(current_QA)
                    trajectories = planner.plan(
                        model.timeStep * 0.1, roadgraph, vehicles, Behaviour(executed_action), other_plan=False
                    )
                    action_list.append(executed_action)
                    deadlock_last_action, deadlock_streak_len = recent_action_streak(action_list)
                    deadlock_after_action = original_deadlock_condition(
                        action_list,
                        model.ms.ego.speed,
                        window=args.deadlock_warning_window,
                        speed_threshold=args.deadlock_speed_threshold,
                    )
                    context_log_record.update({
                        "deadlock_after_action": deadlock_after_action,
                        "deadlock_after_action_streak_len": deadlock_streak_len,
                        "deadlock_after_action_last_action": deadlock_last_action if deadlock_last_action is not None else -1,
                        "deadlock_after_action_ego_speed": model.ms.ego.speed,
                    })
                    selected_k_all.append(int(selector_metadata["selected_K"]))
                    if h_available == 20:
                        selected_mask_counter.update([str(selector_metadata["selected_mask"])])
                        selected_k_active.append(int(selector_metadata["selected_K"]))
                        selector_fallback_count += int(bool(selector_metadata["fallback_to_full"]))
                    selector_latency_total += float(selector_metadata["selector_inference_seconds"])
                    policy_latency_total += policy_seconds
                    idle_action_count += int(executed_action == 8)
                    deadlock_indicator_count += int(deadlock_after_action)
                    shield_filtered_decision_count += int(
                        shield_result.shield_triggered
                    )
                    shield_override_count += int(shield_override)
                    shield_forced_single_action_count += int(
                        shield_result.forced_single_action
                    )
                    shield_no_safe_action_count += int(
                        shield_result.no_safe_action
                    )
                    shield_raw_unsafe_action_count += int(
                        raw_llm_action not in shield_result.safe_actions
                    )
                    shield_history_write_skipped_count += int(
                        bool(history_write_skipped_reason)
                    )
                    append_selector_csv(
                        selector_log_path,
                        {
                            "pair_id": args.pair_id,
                            "episode_id": episode_id,
                            "route_id": args.route_id,
                            "domain_id": args.domain_id,
                            "sumo_seed": args.sumo_seed,
                            "context_policy": args.context_policy,
                            "adaptive_selector_artifact_sha256": policy_provenance.get(
                                "adaptive_selector_artifact_sha256", ""
                            ),
                            "decision_index": decision_index,
                            "H_available": h_available,
                            **selector_metadata,
                            "actual_input_tokens": llm_cost["prompt_tokens"],
                            "completion_tokens": llm_cost["completion_tokens"],
                            "total_tokens": llm_cost["total_tokens"],
                            "action_selected": executed_action,
                            "driving_policy_api_seconds": policy_seconds,
                            "policy_api_call_count": prompt_debug_metadata["policy_api_call_count"],
                            "policy_api_request_attempts": prompt_debug_metadata["policy_api_request_attempts"],
                            "policy_api_retry_attempts": prompt_debug_metadata["policy_api_retry_attempts"],
                            "action_contract_repair_count": prompt_debug_metadata["action_contract_repair_count"],
                            "initial_invalid_action": prompt_debug_metadata["initial_invalid_action"],
                            "action_parse_method": prompt_debug_metadata["action_parse_method"],
                            "action_parse_fallback_used": prompt_debug_metadata["action_parse_fallback_used"],
                            "action_parse_error": prompt_debug_metadata["action_parse_error"],
                            "reward_parse_method": prompt_debug_metadata["reward_parse_method"],
                            "reward_parse_error": prompt_debug_metadata["reward_parse_error"],
                            **shield_log_fields(
                                shield_result,
                                raw_llm_action=raw_llm_action,
                                executed_action=executed_action,
                                shield_override=shield_override,
                                history_write_skipped_reason=(
                                    history_write_skipped_reason
                                ),
                            ),
                        },
                    )
                    write_context_pool_log(context_log_path, context_log_record)
                    decision_index += 1
                    if args.max_decisions is not None and decision_index >= args.max_decisions:
                        raise MaxDecisionsReached()
                    if args.llm_call_sleep > 0:
                        time.sleep(args.llm_call_sleep)
                    if deadlock_after_action:

                        # if choice = "log_only", no BrainDeadlockException will be raised (details refer to line347)
                        if args.deadlock_termination_mode == "raise":
                            raise BrainDeadlockException()
                        LLM_logger.info(
                            "Potential deadlock logged without termination: last_action=%s streak=%s speed=%s",
                            deadlock_last_action,
                            deadlock_streak_len,
                            model.ms.ego.speed,
                        )
                    if len(action_list) > 100:
                        raise TimeOutException()
                    model.setTrajectories(trajectories)
                else:
                    model.ego.exitControlMode()

            model.updateVeh()

    # step3. record the result
    except (
        CollisionException, LaneChangeException, 
        BrainDeadlockException, TimeOutException
        ) as e:
        episode_status = "failed"
        failure_stage = failure_stage or "simulation"
        original_exception_class = type(e).__name__
        original_exception_message = exception_message(e)
        failure_error_type = classify_failure_error(e)
        failure_error_message = original_exception_message
        record_result(model, total_start_time, False, failure_error_message)
        model.dbBridge.commitData()
    except MaxDecisionsReached as e:
        episode_status = "max_decisions_reached"
        failure_stage = "max_decisions"
        original_exception_class = type(e).__name__
        original_exception_message = f"Reached --max-decisions={args.max_decisions}"
        failure_error_type = type(e).__name__
        failure_error_message = original_exception_message
        record_result(model, total_start_time, False, failure_error_message)
        model.dbBridge.commitData()
    except KeyboardInterrupt:
        episode_status = "failed"
        failure_stage = failure_stage or "keyboard_interrupt"
        original_exception_class = "KeyboardInterrupt"
        original_exception_message = "Interrupted by user"
        failure_error_type = "keyboard_interrupt"
        failure_error_message = original_exception_message
        model.dbBridge.commitData()
        raise
    except Exception as e:
        episode_status = "failed"
        failure_stage = failure_stage or "unknown"
        original_exception_class = type(e).__name__
        original_exception_message = exception_message(e)
        failure_error_type = classify_failure_error(e)
        failure_error_message = original_exception_message
        model.dbBridge.commitData()
        raise
    else:
        episode_status = "success"
        record_result(model, total_start_time, True, None)
    finally:
        try:
            model.dbBridge.commitData()
        except Exception:
            pass
        evaluation_metrics = evaluate_database_metrics(database)
        active_count = len(selected_k_active)
        wall_clock_episode_seconds = time.time() - total_start_time
        simulation_time_seconds = float(model.timeStep) * float(step_length)
        terminal_outcome = normalize_terminal_outcome(
            episode_status,
            failure_error_type,
            failure_stage,
        )
        raw_episode_success = terminal_outcome == "success"
        max_history_used = max(selected_k_active, default=0)
        k_distribution = {
            f"K{k}_count": int(sum(value == k for value in selected_k_active))
            for k in range(21)
        }
        episode_metrics = {
            "pair_id": args.pair_id,
            "context_policy": args.context_policy,
            "scene_config_id": domain_config["scene_config_id"],
            "history_horizon": 20,
            "scenario_id": args.scenario_id,
            "map_id": args.map_id,
            "route_id": args.route_id,
            "route_variant_id": args.route_variant_id,
            "ego_id": ego_id,
            "domain_id": args.domain_id,
            "sumo_seed": args.sumo_seed,
            "sumo_seed_effective": domain_config["sumo_seed_effective"],
            "traffic_density_id": domain_config["traffic_density_id"],
            "traffic_density_scale": domain_config["traffic_density_scale"],
            "traffic_density_effective": domain_config["traffic_density_effective"],
            "traffic_density_method": domain_config["traffic_density_method"],
            "base_sumo_rou_file": domain_config["base_sumo_rou_file"],
            "weather_id": domain_config["weather_id"],
            "weather_level": domain_config["weather_level"],
            "weather_params": json.dumps(domain_config["weather_params"], sort_keys=True),
            "weather_prompt_effective": domain_config["weather_prompt_effective"],
            "weather_sim_effective": domain_config["weather_sim_effective"],
            "weather_effective_type": domain_config["weather_effective_type"],
            "ego_depart_lane_mode": args.ego_depart_lane_mode,
            "sumo_cfg_file": domain_config["sumo_cfg_file"],
            "sumo_net_file": domain_config["sumo_net_file"],
            "sumo_launch_method": domain_config["sumo_launch_method"],
            "sumo_cfg_effective": domain_config["sumo_cfg_effective"],
            "episode_id": episode_id,
            "episode_status": episode_status,
            "terminal_outcome": terminal_outcome,
            "success": raw_episode_success,
            "run_finished_without_exception": raw_episode_success,
            "pilot_terminated_by_max_decisions": terminal_outcome == "max_decisions_reached",
            "collision": failure_error_type == "CollisionException",
            "timeout": failure_error_type == "TimeOutException",
            "lane_change_failure": failure_error_type == "LaneChangeException",
            "deadlock_failure": failure_error_type == "BrainDeadlockException",
            "failure_error_type": failure_error_type,
            "failure_error_message": failure_error_message,
            "original_exception_class": original_exception_class,
            "original_exception_message": original_exception_message,
            "route_completion": evaluation_metrics.get("route_completion", ""),
            "wall_clock_episode_seconds": wall_clock_episode_seconds,
            "simulation_time_seconds": simulation_time_seconds,
            "decision_count": decision_index,
            "idle_action_count": idle_action_count,
            "deadlock_indicator_count": deadlock_indicator_count,
            "safety_shield_enabled": not args.disable_safety_shield,
            "shield_horizon_seconds": args.shield_horizon,
            "shield_dt_seconds": args.shield_dt,
            "shield_longitudinal_margin_m": args.shield_longitudinal_margin,
            "shield_lateral_margin_m": args.shield_lateral_margin,
            "shield_filtered_decision_count": shield_filtered_decision_count,
            "shield_override_count": shield_override_count,
            "forced_single_action_count": (
                shield_forced_single_action_count
            ),
            "no_safe_action_count": shield_no_safe_action_count,
            "raw_unsafe_action_count": (
                shield_raw_unsafe_action_count
            ),
            "shield_history_write_skipped_count": (
                shield_history_write_skipped_count
            ),
            "shield_prediction_truncated_decision_count": (
                shield_prediction_truncated_decision_count
            ),
            "shield_evaluation_failure_count": (
                shield_evaluation_failure_count
            ),
            "Safety": evaluation_metrics.get("safety", ""),
            "Comfort": evaluation_metrics.get("comfort", ""),
            "Efficiency": evaluation_metrics.get("efficiency", ""),
            "evaluation_valid": evaluation_metrics.get("evaluation_valid", False),
            "recorded_episode_success": evaluation_metrics.get("recorded_episode_success"),
            "evaluation_status": evaluation_metrics.get("evaluation_status", "failed"),
            "evaluation_error": evaluation_metrics.get("evaluation_error", ""),
            "num_eval_frames": evaluation_metrics.get("num_eval_frames", ""),
            "mean_selected_K_h20": (
                sum(selected_k_active) / active_count if active_count else ""
            ),
            "median_selected_K_h20": (
                statistics.median(selected_k_active) if active_count else ""
            ),
            "max_selected_K_h20": max_history_used if active_count else "",
            **k_distribution,
            "full_context_fallback_rate": selector_fallback_count / active_count if active_count else "",
            "mean_selected_context_chunks_all_decisions": (
                sum(selected_k_all) / len(selected_k_all) if selected_k_all else ""
            ),
            "total_driving_policy_input_tokens": prompt_tokens_total,
            "mean_input_tokens_per_decision": prompt_tokens_total / decision_index if decision_index else "",
            "completion_tokens": completion_tokens_total,
            "total_tokens": total_tokens_total,
            "api_cost_usd": api_cost_total,
            "campaign_id": args.campaign_id,
            "policy_protocol_id": args.policy_protocol_id,
            "model_requested": args.llm_model,
            "model_returned": json.dumps(sorted(model_returned_ids)),
            "policy_provenance_sha256": sha256_text(json.dumps(
                policy_provenance, sort_keys=True, separators=(",", ":")
            )),
            "policy_api_call_count": policy_api_call_count_total,
            "policy_api_request_attempts": policy_api_request_attempts_total,
            "policy_api_retry_attempts": policy_api_retry_attempts_total,
            "selector_inference_latency_mean_seconds": (
                selector_latency_total / active_count if active_count else ""
            ),
            "driving_policy_api_latency_mean_seconds": (
                policy_latency_total / decision_index if decision_index else ""
            ),
            "selected_mask_distribution": json.dumps(dict(sorted(selected_mask_counter.items()))),
            "selected_mask_order": SELECTED_MASK_ORDER,
            "db_path": database,
            "context_log_path": context_log_path,
            "selector_log_path": str(selector_log_path),
        }
        with episode_metrics_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(episode_metrics))
            writer.writeheader()
            writer.writerow(episode_metrics)
        write_episode_status(
            episode_status_path,
            {
                "status": episode_status,
                "failure_stage": failure_stage,
                "failure_error_type": failure_error_type,
                "failure_error_message": failure_error_message,
                "original_exception_class": original_exception_class,
                "original_exception_message": original_exception_message,
                "terminal_outcome": terminal_outcome,
                "raw_episode_success": raw_episode_success,
                "success": episode_metrics["success"],
                "evaluation_valid": episode_metrics["evaluation_valid"],
                "recorded_episode_success": episode_metrics["recorded_episode_success"],
                "evaluation_status": evaluation_metrics.get("evaluation_status", "failed"),
                "evaluation_error": evaluation_metrics.get("evaluation_error", ""),
                "num_eval_frames": evaluation_metrics.get("num_eval_frames", ""),
                "wall_clock_episode_seconds": wall_clock_episode_seconds,
                "simulation_time_seconds": simulation_time_seconds,
                "decisions_completed": decision_index,
                "safety_shield_enabled": not args.disable_safety_shield,
                "shield_filtered_decision_count": (
                    shield_filtered_decision_count
                ),
                "shield_override_count": shield_override_count,
                "forced_single_action_count": (
                    shield_forced_single_action_count
                ),
                "no_safe_action_count": (
                    shield_no_safe_action_count
                ),
                "raw_unsafe_action_count": (
                    shield_raw_unsafe_action_count
                ),
                "shield_history_write_skipped_count": (
                    shield_history_write_skipped_count
                ),
                "shield_prediction_truncated_decision_count": (
                    shield_prediction_truncated_decision_count
                ),
                "shield_evaluation_failure_count": (
                    shield_evaluation_failure_count
                ),
                "output_prefix": str(output_prefix),
                "domain_id": domain_config["domain_id"],
                "map_id": args.map_id,
                "route_id": args.route_id,
                "route_variant_id": args.route_variant_id,
                "ego_id": ego_id,
                "sumo_seed": args.sumo_seed,
                "episode_id": episode_id,
                "campaign_id": args.campaign_id,
                "policy_protocol_id": args.policy_protocol_id,
                "model_requested": args.llm_model,
                "model_returned": sorted(model_returned_ids),
                "policy_provenance": policy_provenance,
            },
        )
        model.destroy()
        gui.terminate()
        gui.join()

    
