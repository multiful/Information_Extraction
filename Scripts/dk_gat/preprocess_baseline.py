"""Feature construction for the plain ATLOP-reproduction baseline
(Scripts/dk_gat/model_baseline.py) -- no graph, no sentence nodes, no
evidence loss, just what BERT + Entity Marker/logsumexp pooling + Localized
Context Pooling + Grouped Bilinear need.

Self-contained re-implementation of the same marker-insertion scheme as
Scripts/atlop/preprocess.py (same `*` markers, same entity_pos semantics --
start = `*` start-marker subword index, end = one past the `*` end marker,
both pre-[CLS] so the model adds offset=1) -- written fresh here rather than
imported, since Scripts/atlop is a teammate's track (see
Scripts/dk_gat/README.md's jurisdiction note).
"""

import sys
from pathlib import Path
from typing import Optional

from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data.docred_io import build_rel2id  # noqa: E402

MARKER = "*"


def _encode_with_markers(doc: dict, tokenizer: PreTrainedTokenizerBase):
    sents = doc["sents"]
    vertex_set = doc["vertexSet"]

    entity_start, entity_end = set(), set()
    for entity in vertex_set:
        for m in entity:
            sid = m["sent_id"]
            start_w, end_w = m["pos"]
            entity_start.add((sid, start_w))
            entity_end.add((sid, end_w - 1))

    tokens: list = []
    sent_map: list = []
    for sid, sent in enumerate(sents):
        smap = {}
        for wi, word in enumerate(sent):
            wp = tokenizer.tokenize(word)
            if (sid, wi) in entity_start:
                wp = [MARKER] + wp
            if (sid, wi) in entity_end:
                wp = wp + [MARKER]
            smap[wi] = len(tokens)
            tokens.extend(wp)
        smap[len(sent)] = len(tokens)
        sent_map.append(smap)

    entity_pos = []
    for entity in vertex_set:
        spans = []
        for m in entity:
            sid = m["sent_id"]
            start_w, end_w = m["pos"]
            spans.append((sent_map[sid][start_w], sent_map[sid][end_w]))
        entity_pos.append(spans)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]
    return input_ids, entity_pos


def build_baseline_features(
    docs,
    tokenizer: PreTrainedTokenizerBase,
    rel2id: Optional[dict] = None,
    show_progress: bool = True,
) -> list:
    if rel2id is None:
        rel2id = build_rel2id()

    features = []
    it = tqdm(docs, desc="preprocess-baseline") if show_progress else docs
    for doc in it:
        input_ids, entity_pos = _encode_with_markers(doc, tokenizer)
        n_ent = len(doc["vertexSet"])

        triples = {}
        for label in doc.get("labels", []):
            key = (label["h"], label["t"])
            triples.setdefault(key, []).append(rel2id[label["r"]])

        hts, labels = [], []
        for h in range(n_ent):
            for t in range(n_ent):
                if h == t:
                    continue
                hts.append((h, t))
                labels.append(sorted(set(triples[(h, t)])) if (h, t) in triples else [0])

        features.append({
            "input_ids": input_ids,
            "entity_pos": entity_pos,
            "hts": hts,
            "labels": labels,
            "title": doc["title"],
            "num_entities": n_ent,
        })
    return features
