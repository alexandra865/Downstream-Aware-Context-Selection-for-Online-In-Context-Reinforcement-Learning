"""Scene fields and parsing for learned order-guided redundancy compression."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


HISTORY_INTRO = (
    "Here are your previous experiences, their true temporal ages, the "
    "decisions you took, and their corresponding rewards:\n\n"
)


WEATHER_PREFIX = (
    "Here are the current weather parameters, with 0 being the least severe "
    "and 100 being the most severe:"
)


STATIC_SECTIONS = frozenset({
    "Current lane description",
    "Next lane description",
})


DYNAMIC_SECTIONS = frozenset({
    "Your current state",
    "Nearby vehicles description",
    "Last decision",
})


SECTION_PATTERN = re.compile(r"(?m)^### ([^\n]+?):[ \t]*\n")


@dataclass(frozen=True)
class ParsedScene:
    original: str
    prefix: str
    sections: tuple[tuple[str, str], ...]
    weather: str
    parse_ok: bool

    def section_map(self) -> dict[str, str]:
        return dict(self.sections)


def _split_weather(value: str) -> tuple[str, str]:
    position = value.find(WEATHER_PREFIX)
    if position < 0:
        return value, ""
    return value[:position], value[position:]


def parse_scene(value: Any) -> ParsedScene:
    original = str(value or "")
    body, weather = _split_weather(original)
    matches = list(SECTION_PATTERN.finditer(body))
    if not matches:
        return ParsedScene(original, body, (), weather, False)

    prefix = body[:matches[0].start()]
    sections: list[tuple[str, str]] = []
    names: set[str] = set()
    for index, match in enumerate(matches):
        name = match.group(1)
        if name in names:
            return ParsedScene(original, body, (), weather, False)
        names.add(name)
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        sections.append((name, body[match.start():end]))

    required = {"Your current state", "Nearby vehicles description"}
    parse_ok = required.issubset(names)
    return ParsedScene(original, prefix, tuple(sections), weather, parse_ok)


FIELDS = (*sorted(STATIC_SECTIONS), "Weather")


def _value(parsed, field: str) -> str:
    return parsed.weather if field == "Weather" else parsed.section_map().get(field, "")
