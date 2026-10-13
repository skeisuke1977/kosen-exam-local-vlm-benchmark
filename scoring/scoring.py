"""Score frozen answers against a separately stored key (math scoring, see SCORING_VERSION).

Two scores are reported side by side; every rule applies identically to every
model and trial (the rule history is in the published SCORING.md):

- math: the primary score. A deterministic normalization (NFKC, markup and
  brackets, Hepburn/Kunrei romaji labels, minus and choice variants, joined
  labels) is applied identically to every model and trial.
- strict: the printed katakana label, one character per box, one line per box
  under the 最終解答 heading.

Only the final message is read, never the reasoning channel. Nothing is
repaired from the key: a value is credited only when the answer text states it.
"""

import re
import unicodedata
from dataclasses import dataclass, field

SCORING_VERSION = "1.4"

_ROMAJI = {
    "a": "ア", "i": "イ", "u": "ウ", "e": "エ", "o": "オ",
    "ka": "カ", "ki": "キ", "ku": "ク", "ke": "ケ", "ko": "コ",
    "sa": "サ", "shi": "シ", "si": "シ", "su": "ス", "se": "セ", "so": "ソ",
    "ta": "タ", "chi": "チ", "ti": "チ", "tsu": "ツ", "tu": "ツ", "te": "テ",
    "to": "ト",
    "na": "ナ", "ni": "ニ", "nu": "ヌ", "ne": "ネ", "no": "ノ",
    "ha": "ハ", "hi": "ヒ", "fu": "フ", "hu": "フ", "he": "ヘ", "ho": "ホ",
    "ma": "マ", "mi": "ミ", "mu": "ム", "me": "メ", "mo": "モ",
    "ya": "ヤ", "yu": "ユ", "yo": "ヨ",
    "ra": "ラ", "ri": "リ", "ru": "ル", "re": "レ", "ro": "ロ",
    "wa": "ワ",
}  # fmt: skip
_MINUS = "−–—‐‑⊖﹣"
# Kanji that look identical to katakana labels (e.g. 工 for エ), read as the
# katakana only when standing alone, never inside a word.
_HOMOGLYPH = {
    "工": "エ",
    "力": "カ",
    "口": "ロ",
    "二": "ニ",
    "夕": "タ",
    "卜": "ト",
    "八": "ハ",
}
_CJK = "\u3040-\u30ff\u3400-\u9fff"
_LONE_HOMOGLYPH = re.compile(rf"(?<![{_CJK}])[{''.join(_HOMOGLYPH)}](?![{_CJK}])")
_DROP = "*`$|<>〈〉《》()[]{}【】「」『』"
_KATA = "[ア-ン]"
_NOT_KATA = "[ァ-ヺー]"
_VALUE = r"-?\d+|-|[A-Za-z]"
# v1.3: the value must be the whole token. "3.9" or "3/4" is not read as 3.
_VALUE_END = r"(?![A-Za-z0-9])(?![./]\d)"
_ROMAJI_ALT = "|".join(sorted(_ROMAJI, key=len, reverse=True))
# Katakana labels may be followed by whitespace alone. Romaji labels need = or :
# (a bare letter is also a choice value) and must open a line or follow a comma,
# so that variables such as `a = -1` in an explanation are not read as labels.
_PAIR = re.compile(
    rf"(?<!{_NOT_KATA})(?P<kana>{_KATA}{{1,4}})(?!{_NOT_KATA})\s*(?:[=:]\s*|\s+)"
    rf"(?P<kv>{_VALUE}){_VALUE_END}"
    rf"|(?:^|(?<=[,、;]))[ \t]*(?:\d+[ \t]+)?(?:[-・•][ \t]+)?"
    rf"(?P<roma>{_ROMAJI_ALT})(?![A-Za-z])[ \t]*[=:][ \t]*"
    rf"(?P<rv>{_VALUE}){_VALUE_END}",
    re.IGNORECASE | re.MULTILINE,
)
_MATH_HEADING = re.compile(r"最終解答|最終回答|final\s+answers?", re.IGNORECASE)
_STRICT_HEADING = re.compile(
    r"(?m)^\s*(?:#{1,6}\s*)?(?:\*\*)?最終解答(?:\*\*)?[\s:：]*$"
)
_STRICT_LINE = re.compile(r"^\s*(?P<label>[ア-ン])\s*=\s*(?P<value>[0-9a-z-])\s*$")


@dataclass
class Group:
    question: int
    subquestion: int
    labels: list[str]
    answer: list[str]
    points: int


@dataclass
class Parsed:
    values: dict[str, str] = field(default_factory=dict)
    ambiguous: set[str] = field(default_factory=set)
    heading_found: bool = False


def load_groups(key: dict) -> list[Group]:
    return [Group(**group) for group in key["groups"]]


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.translate({ord(c): "-" for c in _MINUS})
    text = text.translate({ord(c): " " for c in _DROP})
    text = _LONE_HOMOGLYPH.sub(lambda m: _HOMOGLYPH[m.group(0)], text)
    return re.sub(r"\\[()\[\]]", " ", text)


# v1.4: a value directly followed by a fraction sign or an explicit alternative
# ("3 / 4", "3⁄4", "3 または 4") is not read as its first number.
_NOT_WHOLE = re.compile(
    r"\s*(?:[./⁄÷]|または|もしくは|あるいは|or\b)\s*-?\d", re.IGNORECASE
)


def _pairs(region: str) -> list[tuple[str, str]]:
    found = []
    text = _normalize(region)
    for match in _PAIR.finditer(text):
        if _NOT_WHOLE.match(text, match.end()):
            continue
        if match.group("kana"):
            label, value = match.group("kana"), match.group("kv").lower()
        else:
            label = _ROMAJI[match.group("roma").lower()]
            value = match.group("rv").lower()
        if len(label) > 1 and len(value) == len(label):
            found.extend(zip(label, value))
        else:
            found.append((label, value))
    return found


def parse_math(answer: str) -> Parsed:
    """Extract label/value pairs from the final message.

    With a heading, read after its last occurrence and treat conflicting values
    for one label as ambiguous. Without a heading, or when nothing follows the
    heading, read the whole message and let the last statement win.
    """
    headings = list(_MATH_HEADING.finditer(answer))
    parsed = Parsed(heading_found=bool(headings))
    pairs = _pairs(answer[headings[-1].end() :]) if headings else []
    strict_region = bool(pairs)
    if not pairs:
        pairs = _pairs(answer)
    for label, value in pairs:
        previous = parsed.values.get(label)
        if strict_region and previous is not None and previous != value:
            parsed.ambiguous.add(label)
        parsed.values[label] = value
    return parsed


def parse_strict(answer: str) -> tuple[Parsed, list[str]]:
    """One `ア = 9` line per box after an exact 最終解答 heading."""
    headings = list(_STRICT_HEADING.finditer(answer))
    parsed = Parsed(heading_found=bool(headings))
    violations: list[str] = []
    if not headings:
        return parsed, ["missing_heading"]
    for line in answer[headings[-1].end() :].splitlines():
        if not line.strip():
            continue
        match = _STRICT_LINE.match(unicodedata.normalize("NFKC", line))
        if not match:
            violations.append(line.strip())
            continue
        label, value = match.group("label"), match.group("value")
        if label in parsed.values:
            parsed.ambiguous.add(label)
        parsed.values[label] = value
    return parsed, violations


def _math_correct(group: Group, parsed: Parsed) -> bool:
    if any(label in parsed.ambiguous for label in group.labels):
        return False
    observed = "".join(parsed.values.get(label, "") for label in group.labels)
    return observed == "".join(group.answer)


def _strict_correct(group: Group, parsed: Parsed) -> bool:
    return all(
        label not in parsed.ambiguous and parsed.values.get(label) == expected
        for label, expected in zip(group.labels, group.answer)
    )


def score_answer(answer: str, groups: list[Group]) -> dict:
    """Score one frozen final message against the groups its unit covers."""
    math = parse_math(answer)
    strict, violations = parse_strict(answer)
    rows = []
    for group in groups:
        rows.append(
            {
                "question": group.question,
                "subquestion": group.subquestion,
                "labels": "".join(group.labels),
                "points": group.points,
                "observed_math": {
                    label: math.values.get(label) for label in group.labels
                },
                "math_points": group.points if _math_correct(group, math) else 0,
                "strict_points": (
                    group.points if _strict_correct(group, strict) else 0
                ),
            }
        )
    required = {label for group in groups for label in group.labels}
    return {
        "scoring_version": SCORING_VERSION,
        "max_points": sum(group.points for group in groups),
        "math_points": sum(row["math_points"] for row in rows),
        "strict_points": sum(row["strict_points"] for row in rows),
        "heading_found": math.heading_found,
        "format_compliant": (
            strict.heading_found
            and not violations
            and not strict.ambiguous
            and required <= set(strict.values)
        ),
        "strict_violations": violations,
        "groups": rows,
    }
