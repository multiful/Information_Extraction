"""Official DocRED evaluation, ported from thunlp/DocRED's code/evaluation.py
(MIT licensed: https://github.com/thunlp/DocRED/blob/master/code/evaluation.py).
Computes RE F1, Ign F1 (the metric papers like ATLOP actually report), and
Evidence F1. Ported line-for-line rather than reimplemented from the metric's
description, so it matches literature numbers exactly (e.g. it faithfully
reproduces the official script's asymmetry where "ignore" filtering only
applies to precision, not recall).

"Ign F1" as usually cited in papers (e.g. ATLOP's BERT-base Dev Ign F1 =
59.22) is ign_annotated_f1 below -- it excludes (head, tail, relation) facts
that already appear in train_annotated.json from the precision computation,
since a model could get those "right" via memorizing entity-name co-occurrence
across documents rather than reading the document. ign_distant_f1 (the same
idea but against train_distant.json) is not typically reported in papers
since most models don't train on distant data at all, but is included since
it's part of the official script and this project's models do use distant
data.

Shared team infra -- any branch's predictions.json ([{"title","h_idx","t_idx",
"r"}, ...], the format every branch's train script already writes) can be
scored with this.

Usage:
    python Scripts/docred_scorer.py <predictions.json> [--dev_split dev]
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.docred_io import DATA_DIR, load_split

FACT_CACHE_DIR = DATA_DIR / "_fact_cache"


def gen_train_facts(split_name: str) -> set[tuple]:
    """(head_mention_name, tail_mention_name, relation) triples over every
    mention-pair of every labeled relation in a train split. Cached to disk
    (train_distant has 101,873 docs, not worth rebuilding every run)."""
    FACT_CACHE_DIR.mkdir(exist_ok=True)
    cache_path = FACT_CACHE_DIR / f"{split_name}.fact.json"
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            return set(tuple(x) for x in json.load(f))

    facts = set()
    for doc in load_split(split_name):
        vertex_set = doc["vertexSet"]
        for label in doc["labels"]:
            rel = label["r"]
            for n1 in vertex_set[label["h"]]:
                for n2 in vertex_set[label["t"]]:
                    facts.add((n1["name"], n2["name"], rel))

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(list(facts), f)
    return facts


def official_evaluate(predictions: list[dict], dev_docs: list[dict],
                       fact_in_train_annotated: set[tuple], fact_in_train_distant: set[tuple]) -> dict:
    """predictions: [{"title","h_idx","t_idx","r"}, ...] (+ optional "evidence").
    dev_docs: raw DocRED documents (e.g. load_split("dev")), must have gold labels."""
    std = {}
    tot_evidences = 0
    title2vertexset = {}
    for doc in dev_docs:
        title = doc["title"]
        title2vertexset[title] = doc["vertexSet"]
        for label in doc["labels"]:
            std[(title, label["r"], label["h"], label["t"])] = set(label["evidence"])
            tot_evidences += len(label["evidence"])
    tot_relations = len(std)

    tmp = sorted(predictions, key=lambda x: (x["title"], x["h_idx"], x["t_idx"], x["r"]))
    deduped = [tmp[0]] if tmp else []
    for i in range(1, len(tmp)):
        x, y = tmp[i], tmp[i - 1]
        if (x["title"], x["h_idx"], x["t_idx"], x["r"]) != (y["title"], y["h_idx"], y["t_idx"], y["r"]):
            deduped.append(x)

    correct_re = correct_evidence = pred_evi = 0
    correct_in_train_annotated = correct_in_train_distant = 0
    for x in deduped:
        title, h_idx, t_idx, r = x["title"], x["h_idx"], x["t_idx"], x["r"]
        if title not in title2vertexset:
            continue
        vertex_set = title2vertexset[title]
        evi = set(x.get("evidence", []))
        pred_evi += len(evi)
        if (title, r, h_idx, t_idx) in std:
            correct_re += 1
            correct_evidence += len(std[(title, r, h_idx, t_idx)] & evi)
            in_annotated = in_distant = False
            for n1 in vertex_set[h_idx]:
                for n2 in vertex_set[t_idx]:
                    if (n1["name"], n2["name"], r) in fact_in_train_annotated:
                        in_annotated = True
                    if (n1["name"], n2["name"], r) in fact_in_train_distant:
                        in_distant = True
            if in_annotated:
                correct_in_train_annotated += 1
            if in_distant:
                correct_in_train_distant += 1

    n_pred = len(deduped)
    re_p = correct_re / n_pred if n_pred else 0.0
    re_r = correct_re / tot_relations if tot_relations else 0.0
    re_f1 = 2 * re_p * re_r / (re_p + re_r) if (re_p + re_r) else 0.0

    evi_p = correct_evidence / pred_evi if pred_evi else 0.0
    evi_r = correct_evidence / tot_evidences if tot_evidences else 0.0
    evi_f1 = 2 * evi_p * evi_r / (evi_p + evi_r) if (evi_p + evi_r) else 0.0

    # NOTE (faithful to the official script, not a bug): only precision is
    # adjusted for the "ignore" sets below -- recall (re_r) is reused as-is.
    denom_a = n_pred - correct_in_train_annotated
    re_p_ign_a = (correct_re - correct_in_train_annotated) / denom_a if denom_a else 0.0
    re_f1_ign_a = 2 * re_p_ign_a * re_r / (re_p_ign_a + re_r) if (re_p_ign_a + re_r) else 0.0

    denom_d = n_pred - correct_in_train_distant
    re_p_ign_d = (correct_re - correct_in_train_distant) / denom_d if denom_d else 0.0
    re_f1_ign_d = 2 * re_p_ign_d * re_r / (re_p_ign_d + re_r) if (re_p_ign_d + re_r) else 0.0

    return {
        "precision": re_p, "recall": re_r, "f1": re_f1,
        "evidence_f1": evi_f1,
        "ign_annotated_f1": re_f1_ign_a,  # this is the "Ign F1" papers usually report
        "ign_distant_f1": re_f1_ign_d,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=str, help="path to a predictions .json ([{title,h_idx,t_idx,r},...])")
    parser.add_argument("--dev_split", type=str, default="dev")
    args = parser.parse_args()

    with open(args.predictions, encoding="utf-8") as f:
        predictions = json.load(f)
    dev_docs = load_split(args.dev_split)

    print("building train fact sets (cached under docred_data/data/_fact_cache/ after first run)...")
    fact_annotated = gen_train_facts("train_annotated")
    fact_distant = gen_train_facts("train_distant")

    results = official_evaluate(predictions, dev_docs, fact_annotated, fact_distant)
    print(f"F1:               {results['f1']:.4f}  (P={results['precision']:.4f} R={results['recall']:.4f})")
    print(f"Ign F1 (annot.):  {results['ign_annotated_f1']:.4f}  <- standard 'Ign F1' reported in papers")
    print(f"Ign F1 (distant): {results['ign_distant_f1']:.4f}")
    print(f"Evidence F1:      {results['evidence_f1']:.4f}")


if __name__ == "__main__":
    main()
