"""Common DocRED scorer — F1 and Ign F1. Shared by every track (our models and
ATLOP) so the comparison in PRD.md is apples-to-apples.

Ported from thunlp/DocRED's official `evaluation.py` (MIT license). The metric:

  * A prediction (title, h_idx, t_idx, r) is correct iff it is in the gold set.
  * F1        = harmonic mean of precision/recall over all predictions.
  * Ign F1    = same, but correct predictions whose (head_name, tail_name, r)
                fact already appears in train_annotated are removed from BOTH
                the correct count and the submission count — it measures how well
                the model finds facts NOT memorizable from the training set.

Prediction format (PRD section 4), one relation per row, r as a P-code:
    [{"title": "...", "h_idx": 0, "t_idx": 4, "r": "P17"}, ...]
"""

from collections import defaultdict


def _gen_train_facts(train_docs) -> set:
    """(head_mention_name, tail_mention_name, relation) triples seen in training.
    Uses every mention-name pairing of the head/tail clusters, matching the
    official scorer."""
    facts = set()
    for doc in train_docs:
        vertex_set = doc["vertexSet"]
        for label in doc.get("labels", []):
            rel = label["r"]
            for n1 in vertex_set[label["h"]]:
                for n2 in vertex_set[label["t"]]:
                    facts.add((n1["name"], n2["name"], rel))
    return facts


def evaluate(predictions, dev_docs, train_docs=None) -> dict:
    """Score `predictions` against `dev_docs` gold labels.

    predictions : list of {"title","h_idx","t_idx","r"} (r = P-code).
    dev_docs    : the gold split (list of raw DocRED docs, e.g. from
                  DocREDataset("dev")).
    train_docs  : train_annotated docs, used only for Ign F1. If None, Ign
                  metrics are returned equal to the plain metrics.

    Returns precision / recall / f1 / ign_precision / ign_f1 plus raw counts.
    """
    title2doc = {doc["title"]: doc for doc in dev_docs}

    # Gold triple set + per-title vertexSet for the Ign name lookup.
    gold = set()
    for doc in dev_docs:
        title = doc["title"]
        for label in doc.get("labels", []):
            gold.add((title, label["h"], label["t"], label["r"]))
    tot_gold = len(gold)

    fact_in_train = _gen_train_facts(train_docs) if train_docs else set()

    # De-duplicate submission (title, h, t, r).
    seen = set()
    submission = []
    for p in predictions:
        key = (p["title"], p["h_idx"], p["t_idx"], p["r"])
        if key in seen:
            continue
        seen.add(key)
        submission.append(p)

    correct = 0
    correct_in_train = 0
    for p in submission:
        title, h_idx, t_idx, r = p["title"], p["h_idx"], p["t_idx"], p["r"]
        if (title, h_idx, t_idx, r) in gold:
            correct += 1
            doc = title2doc[title]
            in_train = False
            for n1 in doc["vertexSet"][h_idx]:
                for n2 in doc["vertexSet"][t_idx]:
                    if (n1["name"], n2["name"], r) in fact_in_train:
                        in_train = True
            if in_train:
                correct_in_train += 1

    n_sub = len(submission)
    precision = correct / n_sub if n_sub else 0.0
    recall = correct / tot_gold if tot_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    ign_denom = n_sub - correct_in_train
    ign_precision = (correct - correct_in_train) / ign_denom if ign_denom else 0.0
    ign_f1 = (
        2 * ign_precision * recall / (ign_precision + recall)
        if (ign_precision + recall) else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "ign_precision": ign_precision,
        "ign_f1": ign_f1,
        "num_correct": correct,
        "num_correct_in_train": correct_in_train,
        "num_submitted": n_sub,
        "num_gold": tot_gold,
    }


def per_relation_f1(predictions, dev_docs) -> dict:
    """Optional breakdown: micro F1 per relation P-code. Not required by the PRD,
    handy for error analysis."""
    gold = defaultdict(set)
    for doc in dev_docs:
        for label in doc.get("labels", []):
            gold[label["r"]].add((doc["title"], label["h"], label["t"]))

    pred = defaultdict(set)
    for p in predictions:
        pred[p["r"]].add((p["title"], p["h_idx"], p["t_idx"]))

    out = {}
    for r in set(gold) | set(pred):
        g, pr = gold[r], pred[r]
        tp = len(g & pr)
        precision = tp / len(pr) if pr else 0.0
        recall = tp / len(g) if g else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        out[r] = {"precision": precision, "recall": recall, "f1": f1, "gold": len(g), "pred": len(pr)}
    return out
