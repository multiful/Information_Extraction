"""Entity-pair candidate + multi-hot label construction for dk's model branch.

Reuses the shared rel2id mapping from data/docred_io.py (read-only import,
not modified) so class indices stay consistent with whatever the team's
other tracks use.

New file. Does not modify any shared module under data/.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "data"))
import docred_io  # shared, read-only

REL2ID = docred_io.build_rel2id()
ID2REL = {v: k for k, v in REL2ID.items()}
NUM_CLASSES = docred_io.NUM_CLASSES


def build_pairs_and_labels(doc: dict, num_entities: int) -> tuple[list[tuple[int, int]], list[list[float]]]:
    """doc: raw DocRED dict with 'labels' (list of {"r","h","t","evidence"}), possibly [].
    Returns (hts, labels):
      hts    = all ordered (h, t) index pairs, h != t
      labels = one NUM_CLASSES-dim multi-hot float vector per pair (index 0 = Na, left 0)
    """
    gold: dict[tuple[int, int], set[int]] = {}
    for lab in doc.get("labels", []):
        gold.setdefault((lab["h"], lab["t"]), set()).add(REL2ID[lab["r"]])

    hts: list[tuple[int, int]] = []
    labels: list[list[float]] = []
    for h in range(num_entities):
        for t in range(num_entities):
            if h == t:
                continue
            hts.append((h, t))
            vec = [0.0] * NUM_CLASSES
            for rid in gold.get((h, t), ()):
                vec[rid] = 1.0
            labels.append(vec)
    return hts, labels


NUM_DIST_BUCKETS = 6


def _dist_bucket(dist: int) -> int:
    if dist == 0:
        return 0
    if dist == 1:
        return 1
    if dist == 2:
        return 2
    if dist <= 4:
        return 3
    if dist <= 8:
        return 4
    return 5


def compute_dist_buckets(doc: dict, hts: list[tuple[int, int]]) -> list[int]:
    """Bucketed *minimum* sentence distance between any mention of h and any
    mention of t, for each pair in hts (same order). Bucket 0 = they co-occur
    in at least one sentence together; higher buckets = further apart. Merges
    the PRD's separate "Distance Embedding" and "Sentence Position Embedding"
    ideas into one feature, since both describe this same underlying signal."""
    entity_sents = [{m["sent_id"] for m in cluster} for cluster in doc["vertexSet"]]
    return [
        _dist_bucket(min(abs(sh - st) for sh in entity_sents[h] for st in entity_sents[t]))
        for h, t in hts
    ]
