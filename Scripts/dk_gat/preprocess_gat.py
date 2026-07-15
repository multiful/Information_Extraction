"""Feature construction for the dk EGAT model.

Extends the marker-insertion scheme of Scripts/atlop/preprocess.py (same `*`
markers, same entity_pos semantics — start = `*` start-marker subword index,
end = one past the `*` end marker, both pre-[CLS] so the model adds offset=1)
with the extra per-document structure the GAT and the evidence loss need.

Graph is now **heterogeneous** (2 node types) instead of entity-only:

  node 0..num_entities-1              : entity nodes (order = vertexSet)
  node num_entities..num_entities+S-1 : sentence nodes (order = doc["sents"])

so downstream code always knows entities are the first `num_entities` rows
of any (node, ...) tensor and can slice them back out after the GAT.

  sent_spans   : list[(start, end)] token span of each sentence (pre-[CLS])
  entity_types : list[int]          per node -- real type (0-5)/unk(6) for
                                     entities, a single shared pseudo-type
                                     (7) for every sentence node
  edge_cat     : (N+S, N+S) int     4=self, 3=entity-sentence ("appears in"),
                                     2=entity-entity mention overlap,
                                     1=entity-entity same sentence, 0=otherwise
  edge_dist    : (N+S, N+S) int     bucketed min sentence distance, only
                                     meaningful for entity-entity pairs
                                     (0,1,2,3,4,5+); 0 for any connected
                                     entity-sentence/self edge
  adj          : (N+S, N+S) bool    sparse graph: entity-entity edge iff
                                     edge_cat>0 or dist bucket<=2;
                                     entity-sentence edge iff the entity has
                                     a mention in that sentence; self-loops
                                     always on. No sentence-sentence edges
                                     (multi-hop between entities routes
                                     through a shared sentence node instead,
                                     e.g. Steve Jobs -[S1]- Apple -[S2]- California)
  evidence     : dict[(h,t)] -> list[int]  union of gold evidence sent ids
                                     (empty dict for train_distant)

hts/labels are identical to the ATLOP features (all ordered pairs, sparse
positive-id labels, Na -> [0]).
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

ENTITY_TYPES = {"PER": 0, "ORG": 1, "LOC": 2, "TIME": 3, "NUM": 4, "MISC": 5}
SENTENCE_TYPE_ID = 7
NUM_ENTITY_TYPES = 8  # 6 known + 1 unk(entity) + 1 shared sentence pseudo-type
EDGE_CATS = 5         # 0 none / 1 ent-ent same-sentence / 2 ent-ent mention overlap
                      # / 3 entity-sentence "appears in" / 4 self
NUM_DIST_BUCKETS = 6  # 0,1,2,3,4,5+


def _dist_bucket(d: int) -> int:
    return min(d, NUM_DIST_BUCKETS - 1)


def _encode_with_markers_and_sents(doc: dict, tokenizer: PreTrainedTokenizerBase):
    """Same as atlop.preprocess._encode_with_markers, plus per-sentence token
    spans (needed for sentence embeddings in the evidence contrastive loss)."""
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
    sent_spans: list = []
    for sid, sent in enumerate(sents):
        smap = {}
        sent_tok_start = len(tokens)
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
        sent_spans.append((sent_tok_start, len(tokens)))

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
    return input_ids, entity_pos, sent_spans


def _graph_structure(doc: dict):
    """Heterogeneous Entity+Sentence graph: nodes 0..n_ent-1 are entities
    (vertexSet order), nodes n_ent..n_ent+n_sent-1 are sentences (doc order).
    Returns (edge_cat, edge_dist, adj, n_ent, n_sent), all sized (n_ent+n_sent)^2.

    Entities with no direct entity-entity edge can still exchange information
    after 2 GAT layers by routing through a shared sentence node they both
    appear in (e.g. Steve Jobs -[S1]- Apple -[S2]- California), which is the
    whole point of adding sentence nodes -- see module docstring."""
    vs = doc["vertexSet"]
    n_ent = len(vs)
    n_sent = len(doc["sents"])
    n = n_ent + n_sent
    ent_sents = [set(m["sent_id"] for m in e) for e in vs]
    # word-level spans per (sent, entity) for overlap detection
    ent_word_spans = [[(m["sent_id"], m["pos"][0], m["pos"][1]) for m in e] for e in vs]

    edge_cat = [[0] * n for _ in range(n)]
    edge_dist = [[NUM_DIST_BUCKETS - 1] * n for _ in range(n)]
    adj = [[False] * n for _ in range(n)]

    # entity-entity block (unchanged logic from the entity-only graph)
    for i in range(n_ent):
        edge_cat[i][i] = 4
        edge_dist[i][i] = 0
        adj[i][i] = True
        for j in range(n_ent):
            if i == j:
                continue
            d = min(abs(a - b) for a in ent_sents[i] for b in ent_sents[j])
            edge_dist[i][j] = _dist_bucket(d)
            cat = 0
            if ent_sents[i] & ent_sents[j]:
                cat = 1
                for (s1, a1, b1) in ent_word_spans[i]:
                    for (s2, a2, b2) in ent_word_spans[j]:
                        if s1 == s2 and a1 < b2 and a2 < b1:
                            cat = 2
                            break
                    if cat == 2:
                        break
            edge_cat[i][j] = cat
            adj[i][j] = cat > 0 or edge_dist[i][j] <= 2  # sparse graph

    # sentence self-loops
    for s in range(n_sent):
        node = n_ent + s
        edge_cat[node][node] = 4
        edge_dist[node][node] = 0
        adj[node][node] = True

    # entity-sentence edges: "entity appears in sentence" (symmetric).
    # No sentence-sentence edges -- two entities in different sentences
    # reach each other via a shared sentence node instead (see docstring).
    for i in range(n_ent):
        for sid in ent_sents[i]:
            node = n_ent + sid
            edge_cat[i][node] = edge_cat[node][i] = 3
            edge_dist[i][node] = edge_dist[node][i] = 0
            adj[i][node] = adj[node][i] = True

    return edge_cat, edge_dist, adj, n_ent, n_sent


def build_gat_features(
    docs,
    tokenizer: PreTrainedTokenizerBase,
    rel2id: Optional[dict] = None,
    show_progress: bool = True,
) -> list:
    if rel2id is None:
        rel2id = build_rel2id()

    features = []
    it = tqdm(docs, desc="preprocess-gat") if show_progress else docs
    for doc in it:
        input_ids, entity_pos, sent_spans = _encode_with_markers_and_sents(doc, tokenizer)
        vs = doc["vertexSet"]
        n_ent = len(vs)

        triples, evidence = {}, {}
        for label in doc.get("labels", []):
            key = (label["h"], label["t"])
            triples.setdefault(key, []).append(rel2id[label["r"]])
            if label.get("evidence"):
                evidence.setdefault(key, set()).update(label["evidence"])
        evidence = {k: sorted(v) for k, v in evidence.items()}

        hts, labels = [], []
        for h in range(n_ent):
            for t in range(n_ent):
                if h == t:
                    continue
                hts.append((h, t))
                labels.append(sorted(set(triples[(h, t)])) if (h, t) in triples else [0])

        edge_cat, edge_dist, adj, n_ent_chk, n_sent = _graph_structure(doc)
        assert n_ent_chk == n_ent
        # node_types: real type per entity (0-5 known / 6 unk), then the
        # shared sentence pseudo-type (7) for every sentence node -- so
        # node_types has exactly n_ent + n_sent entries, matching edge_cat.
        node_types = [ENTITY_TYPES.get(e[0].get("type"), 6) for e in vs] + [SENTENCE_TYPE_ID] * n_sent

        features.append({
            "input_ids": input_ids,
            "entity_pos": entity_pos,
            "sent_spans": sent_spans,
            "entity_types": node_types,
            "edge_cat": edge_cat,
            "edge_dist": edge_dist,
            "adj": adj,
            "hts": hts,
            "labels": labels,
            "evidence": evidence,
            "title": doc["title"],
            "num_entities": n_ent,
            "num_sentences": n_sent,
        })
    return features
