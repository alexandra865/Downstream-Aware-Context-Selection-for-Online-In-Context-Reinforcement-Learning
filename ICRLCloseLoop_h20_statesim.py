#!/usr/bin/env python3

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import sys

from h20_statesim_runtime import StateSimSelector


BASE = (Path(__file__).resolve().parent / 'ICRLCloseLoop_h20_recent_k_strict_h20.py')
BASE_SHA256 = "837a3da284d6fce712b38a2d067b2209501a6e6004e90a03a839d5ac43836fe9"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class StateSimTransform(ast.NodeTransformer):
    def __init__(self) -> None:
        self.policy_constants_replaced = 0
        self.selection_blocks_replaced = 0
        self.policy_help_replaced = 0
        self.provenance_blocks_replaced = 0

    def visit_Constant(self, node: ast.Constant):
        if node.value == "recent_h20":
            self.policy_constants_replaced += 1
            return ast.copy_location(ast.Constant(value="statesim_h20"), node)
        return node

    def visit_Call(self, node: ast.Call):
        node = self.generic_visit(node)
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "parser"
            and node.func.attr == "add_argument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "--context-policy"
        ):
            for keyword in node.keywords:
                if keyword.arg == "help":
                    keyword.value = ast.Constant(
                        value="Use fixed-K nearest-history retrieval in structured-state space."
                    )
                    self.policy_help_replaced += 1
        return node

    def visit_Assign(self, node: ast.Assign):
        node = self.generic_visit(node)
        if not any(
            isinstance(target, ast.Name) and target.id == "policy_provenance"
            for target in node.targets
        ):
            return node
        if not isinstance(node.value, ast.Dict):
            raise RuntimeError("Expected a policy provenance dictionary")
        replaced = 0
        for index, key in enumerate(node.value.keys):
            if isinstance(key, ast.Constant) and key.value == "recent_history_definition":
                node.value.values[index] = ast.Constant(
                    value="standardized_structured_state_l2_nearest_K_recency_tiebreak"
                )
                replaced += 1
        if replaced != 1:
            raise RuntimeError("Expected one history selection definition")
        for key, variable in (
            ("wrapper_sha256", "statesim_wrapper_sha256"),
            ("statesim_runtime_sha256", "statesim_runtime_sha256"),
            ("statesim_scaler_sha256", "statesim_scaler_sha256"),
        ):
            node.value.keys.append(ast.Constant(value=key))
            node.value.values.append(ast.Name(id=variable, ctx=ast.Load()))
        self.provenance_blocks_replaced += 1
        return node

    def visit_If(self, node: ast.If):
        body_source = "\n".join(
            ast.unparse(statement)
            for statement in node.body
            if isinstance(statement, (ast.Assign, ast.AnnAssign, ast.AugAssign))
        )
        target = "selected_indices = h_indices[:int(args.recent_k)]"
        exact_h20_test = (
            isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "h_available"
            and len(node.test.ops) == 1
            and isinstance(node.test.ops[0], ast.Eq)
            and len(node.test.comparators) == 1
            and isinstance(node.test.comparators[0], ast.Constant)
            and node.test.comparators[0].value == 20
        )
        if not exact_h20_test or target not in body_source:
            return self.generic_visit(node)
        self.selection_blocks_replaced += 1
        replacement = ast.parse(
            """
if h_available == 20:
    statesim_history_prompts = [previous_prompts[index] for index in h_indices]
    statesim_selector_started = time.perf_counter()
    statesim_selected_positions, statesim_audit = statesim_selector.select(
        short_prompt,
        statesim_history_prompts,
        int(args.recent_k),
    )
    statesim_selector_seconds = time.perf_counter() - statesim_selector_started
    selected_indices = sorted(
        h_indices[position] for position in statesim_selected_positions
    )
    selected_position_set = set(statesim_selected_positions)
    selector_metadata.update({
        "selected_mask": "".join(
            "1" if position in selected_position_set else "0"
            for position in range(h_available)
        ),
        "selected_K": int(args.recent_k),
        "mask_scores": [statesim_audit],
        "selector_inference_seconds": statesim_selector_seconds,
    })
"""
        ).body[0]
        return ast.copy_location(replacement, node)


def transformed_tree() -> ast.Module:
    if sha256(BASE) != BASE_SHA256:
        raise RuntimeError("Frozen strict-Recent source SHA changed")
    tree = ast.parse(BASE.read_text(), filename=str(BASE))
    transform = StateSimTransform()
    tree = transform.visit(tree)
    ast.fix_missing_locations(tree)
    if transform.policy_constants_replaced != 2:
        raise RuntimeError(
            f"Expected two policy constants, got {transform.policy_constants_replaced}"
        )
    if transform.selection_blocks_replaced != 1:
        raise RuntimeError(
            f"Expected one H20 selection block, got {transform.selection_blocks_replaced}"
        )
    if transform.policy_help_replaced != 1 or transform.provenance_blocks_replaced != 1:
        raise RuntimeError("Expected one policy help and one provenance block")
    return tree


def show_help(tree: ast.Module) -> None:
    import argparse
    from formal_domain_config import ALL_FORMAL_DOMAINS

    # Execute only the frozen parser setup, without simulator or policy imports.
    constants = {"DRIVING_POLICY_MODEL", "DRIVING_POLICY_PROTOCOL_ID"}
    statements = [
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in constants
    ]
    if len(statements) != len(constants):
        raise RuntimeError("Frozen parser defaults changed")
    main_block = tree.body[-1]
    if not isinstance(main_block, ast.If):
        raise RuntimeError("Frozen runner main block changed")
    for node in main_block.body:
        statements.append(node)
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "args"
            for target in node.targets
        ):
            break
    else:
        raise RuntimeError("Frozen argument parsing block missing")
    namespace = {
        "__file__": str(BASE), "Path": Path, "argparse": argparse,
        "ALL_FORMAL_DOMAINS": ALL_FORMAL_DOMAINS,
    }
    exec(compile(ast.Module(body=statements, type_ignores=[]), str(BASE), "exec"), namespace)


def main() -> None:
    if str(BASE.parent) not in sys.path:
        sys.path.insert(0, str(BASE.parent))
    tree = transformed_tree()
    if any(argument in {"-h", "--help"} for argument in sys.argv[1:]):
        show_help(tree)
        return
    selector = StateSimSelector()
    namespace = {
        "__name__": "__main__",
        "__file__": str(BASE),
        "statesim_selector": selector,
        "statesim_wrapper_sha256": sha256(Path(__file__).resolve()),
        "statesim_runtime_sha256": sha256(BASE.parent / "h20_statesim_runtime.py"),
        "statesim_scaler_sha256": sha256(selector.artifact_path),
    }
    exec(compile(tree, str(BASE), "exec"), namespace)


if __name__ == "__main__":
    main()
