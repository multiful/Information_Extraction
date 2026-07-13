"""ATLOP-style feature construction for DocRED.

Consumes raw documents from the team's shared loader
(`data.docred_dataset.DocREDataset`) — we do NOT re-implement data loading and
we reuse the shared `data.docred_io.build_rel2id` mapping, per PRD.md's
"모든 트랙이 DocREDataset을 그대로 입력으로" rule.

The only ATLOP-specific step is inserting a `"*"` marker token immediately
before and after every mention, then recording the subword index of each start
marker. ATLOP represents an entity by log-sum-exp pooling over its mentions'
start-marker hidden states, so these positions are what the model slices.
Re-implemented from wzhouad/ATLOP (prepro.py, read_docred); the repo's license
is unspecified so nothing is copied verbatim.

Marker positions here are recorded BEFORE the encoder's special tokens are
added; the model adds a `+offset` (1 for the leading [CLS]) when indexing.
"""

import sys
from pathlib import Path

from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data.docred_io import build_rel2id  # noqa: E402

NUM_CLASSES = 97  # Na/TH + 96 relation types
MARKER = "*"


def _encode_with_markers(doc: dict, tokenizer: PreTrainedTokenizerBase):
    """Tokenize a document word-by-word, wrapping each mention in `*` markers.

    Returns (input_ids, entity_pos) where entity_pos[e] is a list of
    (start, end) subword spans (start = the `*` start-marker index), one per
    mention, in vertexSet order. Positions exclude the [CLS]/[SEP] added at the
    end (the model compensates with +offset).
    """
    sents = doc["sents"]
    vertex_set = doc["vertexSet"]

    # (sent_id, word_idx) sets where a `*` marker must be inserted.
    entity_start, entity_end = set(), set()
    for entity in vertex_set:
        for m in entity:
            sid = m["sent_id"]
            start_w, end_w = m["pos"]  # word-level [start, end) within the sentence
            entity_start.add((sid, start_w))
            entity_end.add((sid, end_w - 1))

    tokens: list[str] = []
    # sent_map[sid][word_idx] -> index in `tokens`; also holds len(sent) as a
    # sentinel so a mention ending on the last word maps cleanly.
    sent_map: list[dict[int, int]] = []
    for sid, sent in enumerate(sents):
        smap: dict[int, int] = {}
        for wi, word in enumerate(sent):
            wp = tokenizer.tokenize(word)
            if (sid, wi) in entity_start:
                wp = [MARKER] + wp
            if (sid, wi) in entity_end:
                wp = wp + [MARKER]
            smap[wi] = len(tokens)   # index of this word's first subword (or its `*` start marker)
            tokens.extend(wp)
        smap[len(sent)] = len(tokens)
        sent_map.append(smap)

    entity_pos: list[list[tuple[int, int]]] = []
    for entity in vertex_set:
        spans: list[tuple[int, int]] = []
        for m in entity:
            sid = m["sent_id"]
            start_w, end_w = m["pos"]
            start = sent_map[sid][start_w]   # the `*` start marker
            end = sent_map[sid][end_w]        # one past the `*` end marker
            spans.append((start, end))
        entity_pos.append(spans)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    # Wrap with the encoder's leading/trailing special tokens ([CLS]..[SEP] /
    # <s>..</s>). The single leading token is why the model uses offset=1 when
    # indexing marker positions recorded above.
    input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]
    return input_ids, entity_pos


def build_features(
    docs,
    tokenizer: PreTrainedTokenizerBase,
    rel2id: dict[str, int] | None = None,
    show_progress: bool = True,
) -> list[dict]:
    """Turn an iterable of raw DocRED docs into ATLOP feature dicts.

    Each feature:
      input_ids   : list[int]  (with [CLS]/[SEP], `*` markers inserted)
      entity_pos  : list[list[(start, end)]]  marker-based spans per mention
      hts         : list[(h, t)]              every ordered entity pair, h != t
      labels      : list[list[int]]           per pair, the positive class ids
                                              (Na pairs -> [0]); the collate_fn
                                              expands these to a (pairs, 97)
                                              multi-hot tensor. Stored sparsely
                                              so train_distant (100k docs) fits
                                              in memory.
      title       : str
      num_entities: int
    """
    if rel2id is None:
        rel2id = build_rel2id()

    features: list[dict] = []
    it = tqdm(docs, desc="preprocess") if show_progress else docs
    for doc in it:
        input_ids, entity_pos = _encode_with_markers(doc, tokenizer)
        n_ent = len(doc["vertexSet"])

        # gold relations keyed by (head_idx, tail_idx)
        triples: dict[tuple[int, int], list[int]] = {}
        for label in doc.get("labels", []):
            key = (label["h"], label["t"])
            triples.setdefault(key, []).append(rel2id[label["r"]])

        hts, labels = [], []
        for h in range(n_ent):
            for t in range(n_ent):
                if h == t:
                    continue
                if (h, t) in triples:
                    pos = sorted(set(triples[(h, t)]))
                else:
                    pos = [0]  # Na / TH
                hts.append((h, t))
                labels.append(pos)

        features.append(
            {
                "input_ids": input_ids,
                "entity_pos": entity_pos,
                "hts": hts,
                "labels": labels,
                "title": doc["title"],
                "num_entities": n_ent,
            }
        )
    return features
