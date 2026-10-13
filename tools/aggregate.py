"""Check results/per_question/*.csv, rebuild results/summary.csv from it and compare.

Checks on the per-question rows (rows with superseded=True are left out):
- every subject has its 5 exam files;
- no send unit appears twice for the same model and trial;
- every model and trial has exactly the units listed in unit_plans/ for that exam
  (excluded_units are not sent), so a missing row cannot lower the denominator;
- the points of each exam add up to its maximum (100; English 2025 makeup 79,
  science 2026 main 97).

Rule: the percent for one exam is sum(math_points) / sum(max_points); the mean is
over the 5 exams of a subject, then over the trials. Minutes per exam and status
counts use the same trials.

Every (subject, model) row of results/summary.csv must be reproduced in both
directions: number of trials, per-trial percents, mean, minutes per exam and
status counts. summary.json, summary_table.md and README.md are not checked.

    python tools/aggregate.py
"""

import csv
import json
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUBJECTS = ["math", "english", "social", "japanese", "science"]
EXAMS = [
    "2026_{s}_main",
    "2024_{s}_main",
    "2024_{s}_makeup",
    "2025_{s}_makeup",
    "2026_{s}_makeup",
]
MAX_POINTS = {"2025_english_makeup": 79, "2026_science_main": 97}


def planned_units(exam: str) -> set[tuple[str, str]]:
    """(question, subquestion) of every sent unit, as written in per_question."""
    year, subject, kind = exam.split("_")
    plan = json.loads(
        (ROOT / "unit_plans" / f"{subject}_{year}_{kind}_units_v1.json").read_text(
            "utf-8"
        )
    )
    units = set()
    for unit in plan["units"]:
        sub = unit["subquestions"]
        units.add((str(unit["question"]), "all" if sub == "all" else str(sub[0])))
    return units


def rebuild() -> tuple[dict, list[str]]:
    problems = []
    rows = defaultdict(list)  # (subject, model, trial, exam) -> rows
    for s in SUBJECTS:
        for e in EXAMS:
            exam = e.format(s=s)
            path = ROOT / "results" / "per_question" / f"{exam}.csv"
            if not path.exists():
                problems.append(f"missing per-question file: {path.name}")
                continue
            with path.open(encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    if r["superseded"] == "True":
                        continue
                    rows[(r["subject"], r["model"], int(r["trial"]), r["exam"])].append(
                        r
                    )
    if problems:
        raise SystemExit("\n".join(problems))
    plans = {
        e.format(s=s): planned_units(e.format(s=s)) for s in SUBJECTS for e in EXAMS
    }
    complete = set()
    for (subject, model, trial, exam), rs in rows.items():
        units = Counter((r["question"], r["subquestion"]) for r in rs)
        twice = sorted(u for u, n in units.items() if n > 1)
        if twice:
            problems.append(f"{exam} {model} trial {trial}: units twice {twice}")
        if set(units) != plans[exam]:
            problems.append(
                f"{exam} {model} trial {trial}: units differ from unit_plans "
                f"(missing {sorted(plans[exam] - set(units))}, "
                f"extra {sorted(set(units) - plans[exam])})"
            )
        total = sum(float(r["max_points"]) for r in rs)
        if not twice and total != MAX_POINTS.get(exam, 100):
            problems.append(f"{exam} {model} trial {trial}: max points {total:g}")
    for subject, model, trial in {k[:3] for k in rows}:
        exams = [e.format(s=subject) for e in EXAMS]
        if all((subject, model, trial, e) in rows for e in exams):
            complete.add((subject, model, trial))
        else:
            problems.append(f"{subject} {model} trial {trial}: not all 5 exams")
    out = {}
    for subject, model in sorted({k[:2] for k in complete}):
        exams = [e.format(s=subject) for e in EXAMS]
        trials = sorted(t for s, m, t in complete if (s, m) == (subject, model))
        per_trial, minutes, status = [], [], Counter()
        for t in trials:
            percents = []
            for e in exams:
                rs = rows[(subject, model, t, e)]
                points = sum(float(r["math_points"]) for r in rs)
                maximum = sum(float(r["max_points"]) for r in rs)
                percents.append(100 * points / maximum)
                minutes.append(sum(float(r["wall_seconds"]) for r in rs) / 60)
                status.update(r["status"] for r in rs)
            per_trial.append(st.mean(percents))
        out[(subject, model)] = {
            "trials": str(len(per_trial)),
            "mean_percent": f"{st.mean(per_trial):.1f}",
            "per_trial_percent": " / ".join(f"{x:.1f}" for x in per_trial),
            "minutes_per_exam": f"{st.mean(minutes):.1f}",
            "status_counts": dict(status),
        }
    return out, problems


def main() -> int:
    rebuilt, problems = rebuild()
    with (ROOT / "results" / "summary.csv").open(encoding="utf-8") as f:
        published = {
            (r["subject"], r["model"]): {
                **r,
                "status_counts": json.loads(r["status_counts"]),
            }
            for r in csv.DictReader(f)
        }
    for key in sorted(set(rebuilt) | set(published)):
        if key not in rebuilt or key not in published:
            problems.append(
                f"{key}: only in {'rebuilt' if key in rebuilt else 'summary.csv'}"
            )
            continue
        for field, value in rebuilt[key].items():
            given = published[key][field]
            if field in ("mean_percent", "minutes_per_exam") and given:
                given = f"{float(given):.1f}"
            if given != value:
                problems.append(
                    f"{key} {field}: summary.csv={given!r} rebuilt={value!r}"
                )
    for (subject, model), v in sorted(rebuilt.items()):
        print(
            f"{subject:9s} {model:40s} {v['mean_percent']:>6s}  ({v['trials']} trials)"
        )
    if problems:
        print("PROBLEMS:")
        print("\n".join(problems))
        return 1
    print(
        "per-question rows checked (files, duplicates, units against unit_plans,"
        " maximum points); all"
        f" {len(rebuilt)} rows of results/summary.csv reproduced (trials, per-trial"
        " percents, mean, minutes, status counts)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
