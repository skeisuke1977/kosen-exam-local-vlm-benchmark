"""Score non-math subjects (choice and numeric boxes) against a separate key.

Answer lines follow prompts/general_prompt_v1_0.txt: ``問1 = ウ``, ``問3 = ウ、オ``
(several choices), ``問4 アイ = 12`` (numeric katakana boxes) and ``問2(1) = ア``.
As in scoring.py, only the final message is read, a deterministic normalization
is applied identically to every model, and nothing is repaired from the key.

Key groups: ``{"question": 1, "labels": ["1"], "answer": ["ウ"],
"unordered": false, "points": 3}``. Labels are canonical without 問: ``<n>``,
``<n>(<m>)`` or ``<n><box>`` (e.g. ``4ア``). Sub-items may be numbers, letters
(``(a)``, ``A``) or Roman numerals (``Ⅰ``/``Ⅱ`` become ``(i)``/``(ii)``, g1.6). Several answers for one label are
compared in order, or as a set when ``unordered`` is true.
"""

import re
import unicodedata
from dataclasses import dataclass

from .scoring import _HOMOGLYPH, _LONE_HOMOGLYPH, _MATH_HEADING, _NOT_WHOLE, Parsed

SCORING_VERSION = "g2.0"

# A label is "問3", "(3)", "3" or "3." optionally followed by a sub-item
# ("(1)", or a bare number / circled number before "=") and a katakana box.
# Without 問, a label must open a line (or follow a comma) so that numbers inside
# explanations are not read as labels.
_LABEL = (
    r"(?:(?:(?P<part>[AB])[ \t]*)?問\s*|(?:^|(?<=[,、;]))[ \t]*\(?[ \t]*)"
    r"(?P<num>\d+)[ \t]*[.)]?[ \t]*"
    r"(?:[(\[]\s*(?P<sub>\d+|III|II|iii|ii|[A-Za-z])\s*[)\]]"
    r"|(?P<sub2>\d+|III|II|iii|ii|[A-Za-z])(?=[ \t]*[=:]))?[ \t]*"
)
_VALUE = r"(?P<value>(?:(?!問\s*\d)[^\n])*)"
_LINE = re.compile(
    _LABEL + r"(?P<box>[ア-ン]{1,4})?[ \t]*[=:][ \t]*" + _VALUE, re.MULTILINE
)
# g1.1: inside the final list only, "問2 エ、ク" or "1.エ" without "=" is read too.
_LIST_LINE = re.compile(
    _LABEL + r"(?:(?P<box>[ア-ン]{1,4})(?=[ \t]*[=:]))?[ \t]*[=:]?[ \t]*" + _VALUE,
    re.MULTILINE,
)
_CHOICE = re.compile(r"(?<![ァ-ヺー])[ア-ン](?![ァ-ヺー])")
_NUMBER = re.compile(r"-?\d+")
_MARK = re.compile(r"[○×]")
_LATIN = re.compile(r"(?<![A-Za-z])[A-Za-z](?![A-Za-z])")
# g1.6: a circled-number choice ("③", already turned into "(3)") as the value.
_CIRCLED_VALUE = re.compile(r"\(\s*(\d+)\s*\)")
_FRACTION = re.compile(r"\d\s*[./⁄÷]\s*\d")
# g1.9: _NOT_WHOLE (shared with the math scoring v1.4) rejects a number directly
# followed by a fraction sign or an explicit alternative ("28 / 9", "28⁄9",
# "28 または 29"); g2.0 (math v1.5) also an expression ("28+1"). g2.0: a number
# glued to a letter ("28e2") is not read as its digits either.
_GLUED = re.compile(r"[A-Za-z]")


@dataclass
class GeneralGroup:
    question: int
    labels: list[str]
    answer: list[str]
    points: int
    unordered: bool = False
    # g1.7: each correct element earns points / len(answer): as a set when
    # unordered ("順序は問わない" with points per item), else position by position
    # (named items scored separately). More elements than expected earn nothing.
    partial: bool = False
    # Set when a unit's subquestion numbering differs from the label (e.g. the
    # English 大問5 B part, whose 問1 is listed as subquestion 3 of the unit plan).
    subquestion: int | None = None


def _normalize(text: str) -> str:
    # g1.3: circled numbers mark sub-items; NFKC would fuse "問1①" into "問11".
    # g1.6: a circled number already in brackets ("問1(①)") keeps one pair.
    text = re.sub(
        r"[(（]\s*([①-⑳])\s*[)）]|[①-⑳]",
        lambda m: f"({ord(m.group(1) or m.group(0)) - 0x245F})",
        text,
    )
    # g1.6: the simplified form of 問.
    text = text.replace("问", "問")
    text = unicodedata.normalize("NFKC", text)
    text = text.translate({ord(c): "-" for c in "−–—‐‑﹣"})
    text = text.translate({ord(c): " " for c in "*`$|<>〈〉【】「」『』"})
    return _LONE_HOMOGLYPH.sub(lambda m: _HOMOGLYPH[m.group(0)], text)


# g1.7: inside the final list, a line "label = value" is read with a tokenizer, so
# nested items ("問1 2(i)", "問2-1(i)") and named items ("問5 反応A", "問3[成分]")
# keep their structure: "1(2)(i)", "2(1)(i)", "5[反応A]", "3[成分]".
_HEAD = re.compile(
    r"[ \t]*(?:(?P<part>[AB])[ \t]*)?(?:\d+[ \t]*(?=問))?(?:問\s*|\(?[ \t]*)"
    r"(?P<num>\d+)(?P<rest>[^=:\n]*)[=:]"
)
_TOKEN = re.compile(
    r"[(\[]\s*(?P<br>[^()\[\]]+?)\s*[)\]]"
    r"|(?P<sep>[\s\-.)]+)"
    r"|(?P<box>[ア-ン]{1,4})\s*$"
    r"|(?P<num>\d+)"
    r"|(?P<roman>III|II|iii|ii)(?![A-Za-z])"
    r"|(?P<latin>[A-Za-z])(?![A-Za-z])"
    r"|(?P<name>[^\s()\[\]\-]+)"
)
_ROMAN = re.compile(r"III|II|I|iii|ii|i")
_LIST_VALUE = re.compile(r"(?:(?!問\s*\d)[^\n])*")


def _sub(token: str) -> str:
    if token.isdigit() or _ROMAN.fullmatch(token) or re.fullmatch(r"[A-Za-z]", token):
        return f"({token.lower()})"
    return f"[{token}]"


def _tokenized(line: str):
    """(part, base, box, value) for a "label = value" line, or None."""
    head = _HEAD.match(line)
    if head is None or line.count("=") + line.count(":") != 1:
        return None
    base, box = head.group("num"), None
    for token in _TOKEN.finditer(head.group("rest")):
        if token.group("sep"):
            continue
        if token.group("box"):
            box = token.group("box")
            continue
        text = next(v for k, v in token.groupdict().items() if v and k != "sep")
        base += _sub(text.strip())
    value = _LIST_VALUE.match(line, head.end()).group(0)
    return head.group("part"), base, box, value


def _from_match(m) -> tuple:
    base = m.group("num")
    sub = m.group("sub") or m.group("sub2")
    if sub:
        base += f"({sub.lower()})"
    return m.group("part"), base, m.group("box"), m.group("value")


def _list_entries(text: str) -> list[tuple]:
    entries = []
    for line in text.splitlines():
        one = _tokenized(line)
        if one is not None:
            entries.append(one)
        else:
            entries.extend(_from_match(m) for m in _LIST_LINE.finditer(line))
    return entries


def parse_general(answer: str) -> Parsed:
    headings = list(_MATH_HEADING.finditer(answer))
    parsed = Parsed(heading_found=bool(headings))
    region = answer[headings[-1].end() :] if headings else ""
    matches = _list_entries(_normalize(region))
    strict_region = bool(matches)
    if not matches:
        matches = [_from_match(m) for m in _LINE.finditer(_normalize(answer))]
    # g1.4: some units hold two parts (A and B) that both number from 問1. A part
    # is taken from an explicit "A問1"/"B問1", or, in the final list, from the
    # numbering restarting at 問1. Labels then carry the part ("A1", "B2(1)").
    entries: list[tuple[int, str, str | None]] = []
    part, seen, explicit = 0, set(), False
    for part_name, base, box, value in matches:
        if part_name:
            explicit = True
            this_part = "AB".index(part_name)
        else:
            if strict_region and base.split("(")[0] == "1" and base in seen:
                part, seen = part + 1, set()
            seen.add(base)
            this_part = part
        pairs = _pairs(base, box, value)
        if pairs is None:
            # g2.0: written, but with a value that cannot be read.
            entries.extend((this_part, label, None) for label in _boxes(base, box))
            continue
        for label, one in pairs:
            entries.append((this_part, label, one))
    prefixed = explicit or part > 0
    invalid = set()
    for this_part, label, one in entries:
        if prefixed and this_part < 2:
            label = "AB"[this_part] + label
        if one is None:
            invalid.add(label)
            continue
        previous = parsed.values.get(label)
        if strict_region and previous is not None and previous != one:
            parsed.ambiguous.add(label)
        parsed.values[label] = one
    # g2.0: a label also written with an unreadable value in the final list is not
    # credited from its readable value alone.
    if strict_region:
        parsed.ambiguous |= invalid & set(parsed.values)
    return parsed


def _boxes(base: str, box: str | None) -> list[str]:
    if box and len(box) > 1:
        return [base + b for b in box] + [base + box]
    return [base + (box or "")]


def _pairs(base: str, box: str | None, value: str) -> list[tuple[str, str]] | None:
    """Readable pairs; None when the value is written but cannot be read (g2.0)."""
    value = value.strip()
    # g1.8: decimals and fractions are not read as their integer part. g2.0: the
    # label counts as written-but-unreadable only when the value itself starts
    # with the number ("29/3"); a fraction in a comment or in prose that happens
    # to start a line ("エ（0.5 倍なので）", "1/R ratio …") is dropped as before.
    if _FRACTION.search(value):
        return None if _NUMBER.match(value) else []
    if box:
        number = _NUMBER.match(value)
        if not number:
            return []
        if _NOT_WHOLE.match(value, number.end()) or _GLUED.match(value, number.end()):
            return None
        digits = number.group(0)
        if len(box) > 1 and len(digits) == len(box):
            return [(base + b, d) for b, d in zip(box, digits)]
        return [(base + box, digits)]
    # Katakana choices, ○/× marks, or numbered choices ("6、9").
    choices = _CHOICE.findall(value) or _MARK.findall(value)
    if not choices:
        # g1.2: some items use Latin letters (A-D) as choice labels.
        choices = [c.upper() for c in _LATIN.findall(value)]
    if not choices and _NUMBER.match(value):
        if any(_NOT_WHOLE.match(value, m.end()) for m in _NUMBER.finditer(value)):
            return None
        choices = _NUMBER.findall(value)
    if not choices and _CIRCLED_VALUE.match(value):
        choices = _CIRCLED_VALUE.findall(value)
    return [(base, "、".join(choices))] if choices else []


def _resolve(parsed: Parsed, label: str, question: int | None = None) -> str:
    """A key label without a part also matches a single "A…"/"B…" answer, for
    units that hold one part but whose answer names the part. g1.5: a plain
    item number "3" also matches "<大問>(3)", when an answer prefixes the 大問."""
    if label in parsed.values or not label[:1].isdigit():
        return label
    found = [p + label for p in "AB" if p + label in parsed.values]
    if len(found) == 1:
        return found[0]
    # Only when every answer carries that 大問 prefix, so that "問1③" (item 1,
    # sub-item 3) is never read as item 3.
    head = f"{question}("
    if (
        question is not None
        and label.isdigit()
        and parsed.values
        and all(key.startswith(head) for key in parsed.values)
    ):
        prefixed = head + label + ")"
        if prefixed in parsed.values:
            return prefixed
    return label


def _lookup(parsed: Parsed, label: str, question: int | None = None) -> str | None:
    return parsed.values.get(_resolve(parsed, label, question))


def _named(group: GeneralGroup, parsed: Parsed) -> tuple[list[str], list[str]] | None:
    """g1.7: answers for named items ("5[反応A]", "5[反応B]"). Taken from the same
    names, else from the parent's named lines in order ("問5 反応A = …"), else
    from one combined line on the parent ("問5 = エ、ア"). The count must match."""
    labels = group.labels
    if not all("[" in label for label in labels):
        return None
    parent = labels[0][: labels[0].index("[")]
    if any(label[: label.index("[")] != parent for label in labels):
        return None
    if all(label in parsed.values for label in labels):
        return [parsed.values[label] for label in labels], list(labels)
    children = [
        key
        for key in parsed.values
        if key.startswith(parent + "[")
        and key.endswith("]")
        and key[len(parent) :].count("[") == 1
        and "(" not in key[len(parent) :]
    ]
    if len(children) == len(labels):
        return [parsed.values[key] for key in children], children
    combined = parsed.values.get(parent)
    if combined and len(combined.split("、")) == len(labels):
        return combined.split("、"), [parent]
    return None


def _joined_boxes(
    labels: list[str], parsed: Parsed
) -> tuple[list[str], list[str]] | None:
    """g1.7: numeric boxes written as one number ("問6(1) = 57" for 6(1)ア, 6(1)イ),
    as in the math scoring (v1.2). Digits only, one per box."""
    bases = {re.sub(r"[ア-ン]$", "", label) for label in labels}
    if len(bases) != 1 or not all(re.search(r"[ア-ン]$", lab) for lab in labels):
        return None
    base = bases.pop()
    value = parsed.values.get(base)
    if value and value.isdigit() and len(value) == len(labels):
        return list(value), [base]
    return None


def _scoped(parsed: Parsed, groups: list[GeneralGroup]) -> Parsed:
    """g1.7: in a unit that covers a single 問 n, sub-items written without the 問
    number ("1 = ク" under 問4) are read as n's sub-items ("4(1)"), but only when
    no answer line carries n itself. When every item of 問n is a numbered
    sub-item, n alone answers nothing, so only "n(…)" lines block this."""
    labels = [label for g in groups for label in g.labels]
    starts = [re.match(r"\d+", label) for label in labels]
    if not starts or not all(starts) or not parsed.values:
        return parsed
    heads = {m.group(0) for m in starts}
    if len(heads) != 1:
        return parsed
    n = heads.pop()
    only_subs = all(label.startswith(f"{n}(") for label in labels)
    blocking = rf"[AB]?{n}\(" if only_subs else rf"[AB]?{n}(?!\d)"
    if any(re.match(blocking, key) for key in parsed.values):
        return parsed
    # g1.8: answers that all carry the 大問 number ("1(2) = ウ") are read by g1.5.
    questions = {g.question for g in groups}
    if len(questions) == 1:
        head = f"{questions.pop()}("
        if all(key.startswith(head) for key in parsed.values):
            return parsed
    remapped = Parsed(heading_found=parsed.heading_found)
    for key, value in parsed.values.items():
        m = re.match(r"(\d+)(.*)", key)
        new = f"{n}({m.group(1)}){m.group(2)}" if m else key
        remapped.values[new] = value
        if key in parsed.ambiguous:
            remapped.ambiguous.add(new)

    # g1.8: only when a re-read label is a label of this unit (or the parent of
    # one, such as "4(1)" for the named items "4(1)[直列]").
    def fits(key: str) -> bool:
        return any(
            label == key or (label.startswith(key) and not label[len(key)].isdigit())
            for label in labels
        )

    if not any(fits(key) for key in remapped.values):
        return parsed
    return remapped


def _observe(group: GeneralGroup, parsed: Parsed) -> tuple[list[str], list[str]]:
    """The answers used for a group and the parsed labels they came from."""
    q = group.question
    named = _named(group, parsed)
    if named is not None:
        values, sources = named
        return [v.upper() for v in values], sources
    if len(group.labels) == 1 and len(group.answer) > 1:
        # Several choices for one label: a set ("二つ選べ") or an ordered
        # list (e.g. the 3rd and 5th words of a reordering question).
        source = _resolve(parsed, group.labels[0], q)
        return (parsed.values.get(source) or "").upper().split("、"), [source]
    sources = [_resolve(parsed, lab, q) for lab in group.labels]
    observed = [(parsed.values.get(s) or "").upper() for s in sources]
    joined = _joined_boxes(group.labels, parsed)
    if joined is not None and not any(observed):
        return joined
    return observed, sources


def _earned(group: GeneralGroup, observed: list[str], ambiguous: bool) -> float:
    # g1.8: an ambiguous source label earns nothing on every path.
    if ambiguous:
        return 0
    expected = [a.upper() for a in group.answer]
    if group.partial:
        if len([o for o in observed if o]) > len(expected):
            return 0
        if group.unordered:
            hits = len(set(observed) & set(expected))
        else:
            hits = sum(o == e for o, e in zip(observed, expected))
        return group.points * hits / len(expected)
    if group.unordered:
        return group.points if sorted(observed) == sorted(expected) else 0
    return group.points if observed == expected else 0


def score_general(answer: str, groups: list[GeneralGroup]) -> dict:
    parsed = _scoped(parse_general(answer), groups)
    rows = []
    for g in groups:
        observed, sources = _observe(g, parsed)
        ambiguous = any(s in parsed.ambiguous for s in sources)
        rows.append(
            {
                "question": g.question,
                "labels": " ".join(g.labels),
                "points": g.points,
                # g1.9: one label with several answers keeps them all ("イ、ウ").
                "observed": (
                    {g.labels[0]: "、".join(observed)}
                    if len(g.labels) == 1
                    else dict(zip(g.labels, observed))
                ),
                "sources": sources,
                "ambiguous": ambiguous,
                "math_points": _earned(g, observed, ambiguous),
            }
        )
    return {
        "scoring_version": SCORING_VERSION,
        "max_points": sum(g.points for g in groups),
        "math_points": sum(r["math_points"] for r in rows),
        "heading_found": parsed.heading_found,
        "groups": rows,
    }
