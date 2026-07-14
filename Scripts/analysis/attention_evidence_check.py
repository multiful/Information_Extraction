"""Check whether attention-based token pruning would disproportionately drop
evidence-sentence tokens.

Motivation (Daily_Standup): a candidate way to handle long documents /
speed up training is to prune low-attention-score tokens before/instead of
the sliding-window approach in `Scripts.atlop.long_input`. The risk is that
the encoder's attention -- especially before any RE-specific fine-tuning --
may not reliably highlight evidence-sentence tokens, so pruning by score
could remove the exact signal needed to learn a relation.

This script samples docs from train_annotated (which has gold evidence
sentence ids per label), runs the ATLOP encoder (bert-base-cased by default,
eager attention) once per doc, and compares the "attention received" score
(last layer, averaged over heads and query positions) of tokens inside a
gold evidence sentence vs. tokens outside one. It also simulates a few
per-document pruning rates and reports what fraction of evidence tokens
each rate would remove.

No training happens here -- just a forward pass over a subset of docs, per
the team's "데이터 전부 다 돌려야 하는거 아니고 일부만 train 시켜서 해도 됨" note.

Run from the project root, e.g.

    python -m Scripts.analysis.attention_evidence_check --num_docs 100
"""

import argparse
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import mannwhitneyu
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data.docred_dataset import DocREDataset            # noqa: E402
from Scripts.atlop.long_input import process_long_input  # noqa: E402

OUT_DIR = ROOT / "EDA"
FIG_DIR = OUT_DIR / "figures"
MARKER = "*"
PRUNE_RATES = [0.10, 0.25, 0.50]  # fraction of lowest-scoring tokens dropped


def encode_with_sent_map(doc: dict, tokenizer) -> tuple[list[int], list[int]]:
    """Same marker-insertion scheme as Scripts.atlop.preprocess._encode_with_markers,
    but also returns each subword token's source sentence id (markers inherit
    their mention's sentence; [CLS]/[SEP] get sent_id -1)."""
    entity_start, entity_end = set(), set()
    for entity in doc["vertexSet"]:
        for m in entity:
            sid = m["sent_id"]
            start_w, end_w = m["pos"]
            entity_start.add((sid, start_w))
            entity_end.add((sid, end_w - 1))

    tokens: list[str] = []
    token_sent_ids: list[int] = []
    for sid, sent in enumerate(doc["sents"]):
        for wi, word in enumerate(sent):
            wp = tokenizer.tokenize(word)
            if (sid, wi) in entity_start:
                wp = [MARKER] + wp
            if (sid, wi) in entity_end:
                wp = wp + [MARKER]
            tokens.extend(wp)
            token_sent_ids.extend([sid] * len(wp))

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]
    token_sent_ids = [-1] + token_sent_ids + [-1]
    return input_ids, token_sent_ids


@torch.no_grad()
def attention_received(model, input_ids: list[int], device) -> np.ndarray:
    """Per-token score = mean attention received (last encoder layer),
    averaged over heads and over query positions. A standard proxy signal
    for attention-based token pruning (Power-BERT / TR-BERT style)."""
    ids = torch.tensor([input_ids], device=device)
    mask = torch.ones_like(ids)
    start_tokens = [model.config.cls_token_id]
    end_tokens = [model.config.sep_token_id]
    _, attention = process_long_input(model, ids, mask, start_tokens, end_tokens)
    attn = attention[0].mean(0)      # (seq_q, seq_k), averaged over heads
    received = attn.mean(0)          # (seq_k,), averaged over queries
    return received.cpu().numpy()


def evidence_sent_ids(doc: dict) -> set[int]:
    ids: set[int] = set()
    for label in doc.get("labels", []):
        ids.update(label.get("evidence", []))
    return ids


def per_doc_prune_impact(scores: np.ndarray, is_evidence: np.ndarray, rate: float) -> tuple[float, float]:
    """At this per-doc prune rate (bottom-`rate` fraction of tokens by score
    dropped), returns (fraction of evidence tokens dropped, fraction of
    non-evidence tokens dropped). NaN if the doc has none of one group."""
    cutoff = np.percentile(scores, rate * 100)
    dropped = scores < cutoff
    ev_frac = dropped[is_evidence].mean() if is_evidence.any() else np.nan
    non_ev_frac = dropped[~is_evidence].mean() if (~is_evidence).any() else np.nan
    return ev_frac, non_ev_frac


def main():
    p = argparse.ArgumentParser(description="Evidence-vs-attention pruning risk check")
    p.add_argument("--split", default="train_annotated", help="needs gold evidence sentences")
    p.add_argument("--num_docs", type=int, default=100)
    p.add_argument("--model_name_or_path", default="bert-base-cased")
    p.add_argument("--seed", type=int, default=66)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    random.seed(args.seed)
    device = torch.device(args.device)

    docs = list(DocREDataset(args.split))
    docs = [d for d in docs if d.get("labels")]
    sample = random.sample(docs, min(args.num_docs, len(docs)))
    print(f"[data] sampled {len(sample)}/{len(docs)} labeled docs from {args.split}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    model = AutoModel.from_pretrained(
        args.model_name_or_path, config=config, attn_implementation="eager"
    ).to(device)
    model.eval()

    evidence_scores: list[float] = []
    non_evidence_scores: list[float] = []
    prune_impacts = {r: {"evidence": [], "non_evidence": []} for r in PRUNE_RATES}

    for doc in tqdm(sample, desc="scoring"):
        input_ids, token_sent_ids = encode_with_sent_map(doc, tokenizer)
        scores = attention_received(model, input_ids, device)
        sent_ids = np.array(token_sent_ids)
        real = sent_ids != -1  # drop [CLS]/[SEP]

        ev_sents = evidence_sent_ids(doc)
        is_evidence = np.isin(sent_ids, list(ev_sents)) & real

        doc_scores = scores[real]
        doc_is_evidence = is_evidence[real]
        evidence_scores.extend(doc_scores[doc_is_evidence].tolist())
        non_evidence_scores.extend(doc_scores[~doc_is_evidence].tolist())

        for rate in PRUNE_RATES:
            ev_frac, non_ev_frac = per_doc_prune_impact(doc_scores, doc_is_evidence, rate)
            if not np.isnan(ev_frac):
                prune_impacts[rate]["evidence"].append(ev_frac)
            if not np.isnan(non_ev_frac):
                prune_impacts[rate]["non_evidence"].append(non_ev_frac)

    evidence_scores = np.array(evidence_scores)
    non_evidence_scores = np.array(non_evidence_scores)
    print(f"[tokens] evidence={len(evidence_scores)} non_evidence={len(non_evidence_scores)}")

    u_stat, p_value = mannwhitneyu(evidence_scores, non_evidence_scores, alternative="two-sided")

    def pct(a, q):
        return np.percentile(a, q)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    bins = np.linspace(
        min(evidence_scores.min(), non_evidence_scores.min()),
        max(evidence_scores.max(), non_evidence_scores.max()),
        40,
    )
    plt.hist(non_evidence_scores, bins=bins, alpha=0.6, density=True,
             label="non-evidence tokens", color="#C44E52")
    plt.hist(evidence_scores, bins=bins, alpha=0.6, density=True,
             label="evidence tokens", color="#4C72B0")
    plt.xlabel("attention received (last layer, mean over heads/queries)")
    plt.ylabel("density")
    plt.title("Attention score: evidence vs. non-evidence tokens")
    plt.legend()
    plt.tight_layout()
    hist_path = FIG_DIR / "attention_evidence_hist.png"
    plt.savefig(hist_path, dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    x = np.arange(len(PRUNE_RATES))
    width = 0.35
    ev_means = [np.mean(prune_impacts[r]["evidence"]) * 100 for r in PRUNE_RATES]
    non_ev_means = [np.mean(prune_impacts[r]["non_evidence"]) * 100 for r in PRUNE_RATES]
    plt.bar(x - width / 2, ev_means, width, label="evidence tokens dropped", color="#4C72B0")
    plt.bar(x + width / 2, non_ev_means, width, label="non-evidence tokens dropped", color="#C44E52")
    plt.xticks(x, [f"prune bottom {int(r * 100)}%" for r in PRUNE_RATES])
    plt.ylabel("% of tokens dropped (per-doc avg)")
    plt.title("Simulated per-doc attention pruning: evidence vs. non-evidence")
    plt.legend()
    plt.tight_layout()
    impact_path = FIG_DIR / "attention_evidence_pruning_impact.png"
    plt.savefig(impact_path, dpi=150)
    plt.close()

    lines = [
        "# Attention-Based Pruning Risk to Evidence Sentences",
        "",
        f"> **최종 업데이트**: 2026-07-13: `train_annotated`에서 {len(sample)}개 문서를 샘플링해 "
        f"(사전학습된) `{args.model_name_or_path}` 인코더의 마지막 레이어 attention을 측정하고, "
        "evidence 문장 토큰과 비-evidence 토큰의 attention 점수 분포 및 가상 pruning 시나리오에서의 "
        "제거 비율을 비교함.",
        "",
        "생성 스크립트: `Scripts/analysis/attention_evidence_check.py`",
        "",
        "## 배경",
        "",
        "긴 문서를 처리할 때 attention 점수가 낮은 토큰을 쳐내는 방식(예: Power-BERT/TR-BERT류 token "
        "pruning)을 고려할 수 있는데, 이때 relation의 근거가 되는 evidence sentence의 토큰까지 함께 "
        "잘려나갈 위험이 있음. 이 스크립트는 그 위험을 실제로 정량화한다.",
        "",
        "**주의**: 아래 attention은 RE 태스크로 파인튜닝되지 않은 사전학습 encoder(`bert-base-cased`)의 "
        "attention이다. 즉 pruning을 학습 전/파이프라인 앞단에 적용했을 때 볼 수 있는 상황에 해당한다. "
        "학습된 체크포인트가 있다면 `--model_name_or_path`에 로컬 경로를 넘겨 재실행해 비교할 수 있다.",
        "",
        "## 설정",
        "",
        f"- split: `{args.split}` (`--num_docs {args.num_docs}`, `--seed {args.seed}`)",
        f"- encoder: `{args.model_name_or_path}` (attn_implementation=eager, last layer)",
        f"- 점수 정의: 토큰 k가 받는 평균 attention (모든 head/모든 query 위치에 대한 평균)",
        f"- evidence 여부: 문서의 모든 label에 걸친 evidence 문장 id의 합집합에 속하는 문장의 토큰",
        "",
        "## 결과: 점수 분포",
        "",
        "| group | n_tokens | mean | median | p10 | p90 |",
        "|---|---|---|---|---|---|",
        f"| evidence | {len(evidence_scores)} | {evidence_scores.mean():.5f} | "
        f"{pct(evidence_scores, 50):.5f} | {pct(evidence_scores, 10):.5f} | {pct(evidence_scores, 90):.5f} |",
        f"| non-evidence | {len(non_evidence_scores)} | {non_evidence_scores.mean():.5f} | "
        f"{pct(non_evidence_scores, 50):.5f} | {pct(non_evidence_scores, 10):.5f} | "
        f"{pct(non_evidence_scores, 90):.5f} |",
        "",
        f"Mann-Whitney U test (two-sided): U={u_stat:.1f}, p={p_value:.3g} "
        f"({'유의미한 차이' if p_value < 0.05 else '유의미한 차이 없음'}, alpha=0.05).",
        "",
        f"![score distribution]({hist_path.relative_to(OUT_DIR).as_posix()})",
        "",
        "## 결과: 가상 pruning 시나리오",
        "",
        "문서별로 최하위 attention 점수 토큰부터 일정 비율을 쳐낸다고 가정했을 때, evidence/비-evidence "
        "토큰이 각각 얼마나 잘려나가는지 (문서별 비율의 평균):",
        "",
        "| prune rate | evidence tokens dropped | non-evidence tokens dropped |",
        "|---|---|---|",
    ]
    for r, ev, non_ev in zip(PRUNE_RATES, ev_means, non_ev_means):
        lines.append(f"| bottom {int(r * 100)}% | {ev:.1f}% | {non_ev:.1f}% |")
    lines += [
        "",
        f"![pruning impact]({impact_path.relative_to(OUT_DIR).as_posix()})",
        "",
        "evidence tokens dropped 비율이 해당 prune rate보다 유의미하게 높다면, attention 기반 pruning이 "
        "evidence sentence를 비례 이상으로 잘라낸다는 뜻 -- pruning을 도입할 경우 evidence 문장의 토큰을 "
        "보호하는 별도 규칙(예: 문장 단위 최소 보존, evidence 후보 문장 가중치 보정)이 필요할 수 있음.",
    ]
    summary_path = OUT_DIR / "attention_evidence_check.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[saved] {summary_path}")
    print(f"[saved] {hist_path}")
    print(f"[saved] {impact_path}")


if __name__ == "__main__":
    main()
