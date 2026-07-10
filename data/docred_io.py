"""Raw DocRED JSON loading + relation-id mapping. Shared by every team member's branch."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "docred_data" / "data"

SPLITS = ["train_annotated", "train_distant", "dev", "test"]


def load_split(name: str) -> list[dict]:
    """name: one of SPLITS. Returns the raw list of DocRED documents."""
    with open(DATA_DIR / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)


def load_rel_info() -> dict[str, str]:
    """P-code -> human-readable relation name, e.g. {"P17": "country", ...}."""
    with open(DATA_DIR / "rel_info.json", encoding="utf-8") as f:
        return json.load(f)


def build_rel2id() -> dict[str, int]:
    """Deterministic P-code -> class-id mapping, shared across all branches so
    everyone's classifier output dimension lines up the same way.
    Index 0 is reserved for "no relation" (Na)."""
    rel_info = load_rel_info()
    rel2id = {"Na": 0}
    for i, p_code in enumerate(sorted(rel_info.keys())):
        rel2id[p_code] = i + 1
    return rel2id


NUM_CLASSES = 97  # Na + 96 relation types
