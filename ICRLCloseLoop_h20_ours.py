#!/usr/bin/env python3

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
SOURCE = Path(__file__).resolve().parent
CODE = SOURCE / "code"
BASE_ENTRYPOINT = SOURCE / "ICRLCloseLoop_h20_full_context_diagnostic.py"
EXPECTED_BASE_SHA256 = "e337faf3f8a69faca3847903de75f6b9049d3b6a852b52979b24b89b3bb88ee8"
for path in (HERE, SOURCE, CODE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from h20_ours_renderer import compose_lorc_prompt  # noqa: E402
import h20_ours_runtime as runtime  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ReplaceHistoryRenderer(ast.NodeTransformer):
    def __init__(self) -> None:
        self.replacements = 0
        self.renderer_calls = 0
        self.provenance_replacements = 0

    def visit_Call(self, node: ast.Call) -> ast.AST:
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == "compose_selected_env_prompt":
            node.keywords.append(ast.keyword(
                arg="audit_metadata", value=ast.Name(id="selector_metadata", ctx=ast.Load())
            ))
            self.renderer_calls += 1
        return node

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        node = self.generic_visit(node)
        if any(
            isinstance(target, ast.Name) and target.id == "policy_provenance"
            for target in node.targets
        ):
            if not isinstance(node.value, ast.Dict):
                raise RuntimeError("Expected a policy provenance dictionary")
            for key, variable in (
                ("wrapper_sha256", "lorc_wrapper_sha256"),
                ("lorc_renderer_sha256", "lorc_renderer_sha256"),
            ):
                node.value.keys.append(ast.Constant(value=key))
                node.value.values.append(ast.Name(id=variable, ctx=ast.Load()))
            self.provenance_replacements += 1
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        if node.name != "compose_selected_env_prompt":
            return self.generic_visit(node)
        self.replacements += 1
        replacement = ast.parse(
            """
def compose_selected_env_prompt(
    short_prompt, selected_raw_indices, selected_prompts,
    selected_decisions, selected_rewards, *, audit_metadata=None,
):
    prompt, audit = compose_lorc_prompt(
        short_prompt, selected_raw_indices, selected_prompts,
        selected_decisions, selected_rewards,
        lorc_runtime.current_deletion_positions_raw(),
        return_audit=True,
    )
    if audit_metadata is not None:
        audit_metadata["lorc_audit"] = audit
    return prompt
"""
        ).body[0]
        return ast.copy_location(replacement, node)


def main() -> None:
    actual = sha256(BASE_ENTRYPOINT)
    if actual != EXPECTED_BASE_SHA256:
        raise RuntimeError(f"Frozen base changed: {actual}")
    tree = ast.parse(BASE_ENTRYPOINT.read_text(), filename=str(BASE_ENTRYPOINT))
    transformer = ReplaceHistoryRenderer()
    tree = transformer.visit(tree)
    ast.fix_missing_locations(tree)
    if transformer.replacements != 1:
        raise RuntimeError(f"Expected one renderer, found {transformer.replacements}")
    if transformer.renderer_calls != 1 or transformer.provenance_replacements != 1:
        raise RuntimeError("Expected one renderer call and one provenance block")
    scope = {
        "__name__": "__main__", "__file__": str(BASE_ENTRYPOINT),
        "__package__": None,
        "compose_lorc_prompt": compose_lorc_prompt,
        "lorc_runtime": runtime,
        "lorc_wrapper_sha256": sha256(Path(__file__).resolve()),
        "lorc_renderer_sha256": sha256(SOURCE / "h20_ours_renderer.py"),
    }
    exec(compile(tree, str(BASE_ENTRYPOINT), "exec"), scope)


if __name__ == "__main__":
    main()
