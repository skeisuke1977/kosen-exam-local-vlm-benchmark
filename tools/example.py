"""Score one answer text with the published rules and keys.

    python tools/example.py

To try other cases, change ANSWER (a final message), the key file name
(keys/<exam>_v1.json) and the filter on question / subquestion below.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scoring.scoring import load_groups, score_answer
from scoring.scoring_general import GeneralGroup, score_general

# Math: 2024 main, 大問1 (1)(2), written in the requested format.
math_key = json.loads((ROOT / "keys" / "2024_math_main_v1.json").read_text("utf-8"))
groups = [
    g for g in load_groups(math_key) if g.question == 1 and g.subquestion in (1, 2)
]
ANSWER = "（考え方）…\n最終解答\nア = 3\nイ = 2\nウ = 3"
result = score_answer(ANSWER, groups)
print(
    "math:",
    result["math_points"],
    "/",
    result["max_points"],
    "strict:",
    result["strict_points"],
)

# Other subjects: 2026 social main, 大問1, with a full-width and a 工-for-エ typo.
social_key = json.loads((ROOT / "keys" / "2026_social_main_v1.json").read_text("utf-8"))
groups = [GeneralGroup(**g) for g in social_key["groups"] if g["question"] == 1]
ANSWER = "最終解答\n問１ = 工\n問2 = ア\n問3 = ウ\n問4 = イ"
result = score_general(ANSWER, groups)
print("social:", result["math_points"], "/", result["max_points"])
