"""DocRED 원본 JSON을 다루는 공통 유틸 — export_triples.py / load_ground_truth.py /
export_pinecone.py / export_postgres.py가 공유한다.

핵심은 evidence 보완 로직: DocRED 라벨 중 evidence가 비어있는 경우,
1) head/tail이 같은 문장에 함께 언급되면 그 문장을 추론해서 채우고
   ("inferred_cooccurrence"),
2) 공존 문장이 없으면(multi-hop) head 문장과 tail 문장을 잇는 **공유 개체가
   있는 문장 쌍**을 찾아 채우고("inferred_bridge", find_bridge_sentences 참고),
3) 그마저도 없으면(공유 개체 없는 순수 다단 추론) 억지로 채우지 않는다
   ("unresolved_multihop").
"""

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "docred_data" / "data"


def normalize_name(name):
    return " ".join(name.split())


def load_split(name):
    with open(DATA_DIR / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)


def load_rel_info():
    with open(DATA_DIR / "rel_info.json", encoding="utf-8") as f:
        return json.load(f)


def cluster_canonical(cluster):
    """멘션 클러스터에서 가장 흔한 이름/타입을 대표값으로 뽑는다."""
    names = [normalize_name(m["name"]) for m in cluster]
    types = [m["type"] for m in cluster]
    name = Counter(names).most_common(1)[0][0]
    type_ = Counter(types).most_common(1)[0][0]
    return name, type_


def sentence_text(sent_tokens):
    return " ".join(sent_tokens)


def find_bridge_sentences(h_idx, t_idx, mention_sents_h, mention_sents_t, sent_entities):
    """head/tail이 공존하는 문장이 없을 때, head가 언급된 문장과 tail이 언급된
    문장을 잇는 **공유하는 제3의 개체가 있는 문장 쌍**을 찾는다 (multi-hop 근거
    추론). 후보가 여럿이면 두 문장 사이 거리(|h_sent - t_sent|)가 가장 가까운
    쌍을 택한다. 공유 개체가 있는 쌍이 하나도 없으면 None."""
    best = None
    best_dist = None
    for hs in mention_sents_h:
        h_ents = sent_entities.get(hs, set()) - {h_idx}
        if not h_ents:
            continue
        for ts in mention_sents_t:
            t_ents = sent_entities.get(ts, set()) - {t_idx}
            if h_ents & t_ents:
                dist = abs(hs - ts)
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best = (hs, ts)
    if best is None:
        return None
    return sorted(set(best))


def resolve_evidence(label, h_idx, t_idx, mention_sents, sent_entities, sents):
    """(evidence_sent_ids, evidence_texts, evidence_source)를 반환."""
    evidence_sent_ids = label.get("evidence", [])
    evidence_source = "annotated"

    if not evidence_sent_ids:
        mention_sents_h = mention_sents[h_idx]
        mention_sents_t = mention_sents[t_idx]
        cooccur = sorted(set(mention_sents_h) & set(mention_sents_t))
        if cooccur:
            evidence_sent_ids = cooccur
            evidence_source = "inferred_cooccurrence"
        else:
            bridge = find_bridge_sentences(
                h_idx, t_idx, mention_sents_h, mention_sents_t, sent_entities
            )
            if bridge:
                evidence_sent_ids = bridge
                evidence_source = "inferred_bridge"
            else:
                evidence_source = "unresolved_multihop"

    evidence_texts = [
        sentence_text(sents[sid]) for sid in evidence_sent_ids if sid < len(sents)
    ]
    return evidence_sent_ids, evidence_texts, evidence_source


def iter_doc_records(splits):
    """split마다 문서를 순회하며 (split, doc, vertex_meta, mention_sents,
    sent_entities)를 yield.

    vertex_meta[i] = (canonical_name, type), mention_sents[i] = 그 개체가
    언급된 문장 id 정렬 리스트, sent_entities[sent_id] = 그 문장에 언급된
    개체(vertexSet 인덱스) 집합 (find_bridge_sentences가 사용).
    """
    for split in splits:
        docs = load_split(split)
        for doc in docs:
            vertex_meta = [cluster_canonical(c) for c in doc["vertexSet"]]
            mention_sents = [
                sorted(set(m["sent_id"] for m in cluster)) for cluster in doc["vertexSet"]
            ]
            sent_entities = {}
            for vidx, sids in enumerate(mention_sents):
                for sid in sids:
                    sent_entities.setdefault(sid, set()).add(vidx)
            yield split, doc, vertex_meta, mention_sents, sent_entities


def global_entity_id(name, type_):
    return f"{name}::{type_}"


def is_revised_split(split):
    """`train_revised`/`dev_revised`처럼 사람이 재정제(Re-DocRED)한 split인지.

    `_revised` 접미사가 없는 split(train_annotated/dev/train_distant)은 False —
    향후 2단계(모델 예측 triple)도 여기서 False로 떨어지므로 별도 처리 불필요."""
    return split.endswith("_revised")
