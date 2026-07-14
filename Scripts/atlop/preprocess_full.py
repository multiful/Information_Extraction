"""Feature construction for the integrated model (re_model_full.DocREModelFull).

Superset of preprocess.build_features: same `*`-marker tokenization and
entity-pair / label construction, PLUS two extra fields the DREEAM-style
evidence-guided local context needs:

  sent_pos : list[(start, end)]  token span of each sentence (pre-[CLS], the
             model adds +offset just like entity_pos). Lets the model aggregate
             per-pair token attention up to sentence level.
  evidence : list[list[int]]     per entity pair (hts order), the union of gold
             evidence sentence ids across that pair's relations. Empty list for
             pairs with no gold relation, and for every pair in train_distant
             (distant supervision carries no evidence) -> evidence loss then
             simply has nothing to supervise and is skipped.

baseline preprocess.py is left untouched; the marker loop is re-implemented here
(not imported) because the private helper does not expose the sentence map this
model requires. Marker/label semantics are kept identical to preprocess.py.
"""

import sys
from pathlib import Path
from typing import Optional

from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data.docred_io import build_rel2id  # noqa: E402

NUM_CLASSES = 97
MARKER = "*"


def _encode_with_markers_and_sents(doc: dict, tokenizer: PreTrainedTokenizerBase):
    """Same marker tokenization as preprocess._encode_with_markers, but also
    returns per-sentence token spans. Returns (input_ids, entity_pos, sent_pos).

    Positions (entity_pos, sent_pos) exclude the leading [CLS]/<s>; the model
    compensates with +offset, exactly as in the baseline."""
    sents = doc["sents"]
    vertex_set = doc["vertexSet"]

    entity_start, entity_end = set(), set()
    for entity in vertex_set:
        for m in entity:
            sid = m["sent_id"]
            start_w, end_w = m["pos"]
            entity_start.add((sid, start_w))
            entity_end.add((sid, end_w - 1))

    tokens: list[str] = []
    sent_map: list[dict[int, int]] = []
    for sid, sent in enumerate(sents):
        smap: dict[int, int] = {}
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

    # sentence token spans: [first word's first subword, one past last token)
    sent_pos = [(sent_map[sid][0], sent_map[sid][len(sents[sid])]) for sid in range(len(sents))]

    entity_pos: list[list[tuple[int, int]]] = []
    for entity in vertex_set:
        spans: list[tuple[int, int]] = []
        for m in entity:
            sid = m["sent_id"]
            start_w, end_w = m["pos"]
            spans.append((sent_map[sid][start_w], sent_map[sid][end_w]))
        entity_pos.append(spans)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]
    return input_ids, entity_pos, sent_pos


def build_features_full(
    docs,
    tokenizer: PreTrainedTokenizerBase,
    rel2id: Optional[dict] = None,  # dict[str, int]; Optional for py3.9 compat
    show_progress: bool = True,
) -> list[dict]:
    """DocRED docs -> feature dicts for DocREModelFull (adds sent_pos, evidence)."""
    if rel2id is None:
        rel2id = build_rel2id()

    features: list[dict] = []
    it = tqdm(docs, desc="preprocess-full") if show_progress else docs
    for doc in it:
        input_ids, entity_pos, sent_pos = _encode_with_markers_and_sents(doc, tokenizer)
        n_ent = len(doc["vertexSet"])
        n_sent = len(doc["sents"])

        triples: dict[tuple[int, int], list[int]] = {}
        evi_map: dict[tuple[int, int], set] = {}
        for label in doc.get("labels", []):
            key = (label["h"], label["t"])
            triples.setdefault(key, []).append(rel2id[label["r"]])
            # union evidence across the pair's relations; clamp to valid sents
            evi_map.setdefault(key, set()).update(
                s for s in label.get("evidence", []) if 0 <= s < n_sent
            )

        hts, labels, evidence = [], [], []
        for h in range(n_ent):
            for t in range(n_ent):
                if h == t:
                    continue
                if (h, t) in triples:
                    pos = sorted(set(triples[(h, t)]))
                    evi = sorted(evi_map.get((h, t), set()))
                else:
                    pos, evi = [0], []
                hts.append((h, t))
                labels.append(pos)
                evidence.append(evi)

        features.append(
            {
                "input_ids": input_ids,
                "entity_pos": entity_pos,
                "sent_pos": sent_pos,
                "hts": hts,
                "labels": labels,
                "evidence": evidence,
                "title": doc["title"],
                "num_entities": n_ent,
            }
        )
    return features
