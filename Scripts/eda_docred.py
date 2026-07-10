"""DocRED EDA: computes dataset statistics and saves plots/summary under EDA/."""

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "docred_data" / "data"
OUT_DIR = ROOT / "EDA"
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = ["train_annotated", "train_distant", "dev", "test"]


def load_split(name):
    with open(DATA_DIR / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)


def load_rel_info():
    with open(DATA_DIR / "rel_info.json", encoding="utf-8") as f:
        return json.load(f)


def doc_stats(doc):
    n_sents = len(doc["sents"])
    n_tokens = sum(len(s) for s in doc["sents"])
    n_entities = len(doc["vertexSet"])
    n_mentions = sum(len(v) for v in doc["vertexSet"])
    n_labels = len(doc.get("labels", []))
    return n_sents, n_tokens, n_entities, n_mentions, n_labels


def analyze_split(name, docs):
    rows = [doc_stats(d) for d in docs]
    df = pd.DataFrame(rows, columns=["n_sents", "n_tokens", "n_entities", "n_mentions", "n_labels"])

    entity_types = Counter()
    for d in docs:
        for cluster in d["vertexSet"]:
            for mention in cluster:
                entity_types[mention["type"]] += 1

    relation_types = Counter()
    intra_sent = 0
    inter_sent = 0
    evidence_lens = []
    for d in docs:
        for label in d.get("labels", []):
            relation_types[label["r"]] += 1
            evidence_lens.append(len(label.get("evidence", [])))
            head_sents = {m["sent_id"] for m in d["vertexSet"][label["h"]]}
            tail_sents = {m["sent_id"] for m in d["vertexSet"][label["t"]]}
            if head_sents & tail_sents:
                intra_sent += 1
            else:
                inter_sent += 1

    return {
        "name": name,
        "n_docs": len(docs),
        "df": df,
        "entity_types": entity_types,
        "relation_types": relation_types,
        "intra_sent": intra_sent,
        "inter_sent": inter_sent,
        "evidence_lens": evidence_lens,
    }


def plot_hist(series, title, xlabel, path, bins=30):
    plt.figure(figsize=(6, 4))
    plt.hist(series, bins=bins, color="#4C72B0", edgecolor="white")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_bar(counter, title, path, top_n=20):
    items = counter.most_common(top_n)
    labels, values = zip(*items)
    plt.figure(figsize=(8, 4.5))
    plt.bar(labels, values, color="#55A868")
    plt.title(title)
    plt.ylabel("count")
    plt.xticks(rotation=60, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    rel_info = load_rel_info()
    results = {}

    for split in SPLITS:
        print(f"loading {split} ...")
        docs = load_split(split)
        results[split] = analyze_split(split, docs)
        print(f"  -> {len(docs)} docs analyzed")

    # ---- plots ----
    doc_counts = {k: v["n_docs"] for k, v in results.items()}
    plt.figure(figsize=(6, 4))
    plt.bar(doc_counts.keys(), doc_counts.values(), color="#C44E52")
    plt.title("Number of documents per split")
    plt.ylabel("documents")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "doc_counts.png", dpi=150)
    plt.close()

    for split in ["train_annotated", "train_distant", "dev", "test"]:
        df = results[split]["df"]
        plot_hist(df["n_sents"], f"[{split}] Sentences per document", "sentences",
                   FIG_DIR / f"{split}_sents_hist.png")
        plot_hist(df["n_entities"], f"[{split}] Entities per document", "entities",
                   FIG_DIR / f"{split}_entities_hist.png")

    for split in ["train_annotated", "train_distant", "dev"]:
        df = results[split]["df"]
        plot_hist(df["n_labels"], f"[{split}] Relations per document", "relations",
                   FIG_DIR / f"{split}_relations_hist.png")

    # entity type distribution (train_annotated as reference)
    plot_bar(results["train_annotated"]["entity_types"],
              "Entity type distribution (train_annotated)",
              FIG_DIR / "entity_types_train_annotated.png", top_n=10)

    # relation type distribution (train_annotated, mapped to readable names)
    rel_counter = results["train_annotated"]["relation_types"]
    named_counter = Counter({rel_info.get(k, k): v for k, v in rel_counter.items()})
    plot_bar(named_counter, "Top-20 relation types (train_annotated)",
              FIG_DIR / "relation_types_train_annotated.png", top_n=20)

    # intra vs inter-sentence relations
    plt.figure(figsize=(5, 4))
    labels = ["train_annotated", "dev"]
    intra = [results[s]["intra_sent"] for s in labels]
    inter = [results[s]["inter_sent"] for s in labels]
    x = np.arange(len(labels))
    plt.bar(x - 0.2, intra, width=0.4, label="intra-sentence", color="#4C72B0")
    plt.bar(x + 0.2, inter, width=0.4, label="inter-sentence", color="#DD8452")
    plt.xticks(x, labels)
    plt.title("Intra- vs inter-sentence relations")
    plt.ylabel("relation count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "intra_vs_inter_sentence.png", dpi=150)
    plt.close()

    # evidence length distribution (train_annotated)
    plot_hist(results["train_annotated"]["evidence_lens"],
              "[train_annotated] Evidence sentences per relation", "evidence sentence count",
              FIG_DIR / "train_annotated_evidence_len_hist.png", bins=15)

    # ---- summary markdown ----
    lines = []
    lines.append("# DocRED EDA Summary\n")
    lines.append(f"생성 스크립트: `Scripts/eda_docred.py`\n")

    lines.append("## 1. 스플릿별 문서 수\n")
    lines.append("| split | documents |")
    lines.append("|---|---|")
    for split in SPLITS:
        lines.append(f"| {split} | {results[split]['n_docs']:,} |")
    lines.append("")

    lines.append("## 2. 문서 구조 통계 (문장/토큰/엔티티/관계 수, 문서당 평균±표준편차)\n")
    lines.append("| split | sents/doc | tokens/doc | entities/doc | mentions/doc | relations/doc |")
    lines.append("|---|---|---|---|---|---|")
    for split in SPLITS:
        df = results[split]["df"]
        rel_col = f"{df['n_labels'].mean():.1f}±{df['n_labels'].std():.1f}" if split != "test" else "-"
        lines.append(
            f"| {split} "
            f"| {df['n_sents'].mean():.1f}±{df['n_sents'].std():.1f} "
            f"| {df['n_tokens'].mean():.1f}±{df['n_tokens'].std():.1f} "
            f"| {df['n_entities'].mean():.1f}±{df['n_entities'].std():.1f} "
            f"| {df['n_mentions'].mean():.1f}±{df['n_mentions'].std():.1f} "
            f"| {rel_col} |"
        )
    lines.append("")

    lines.append("## 3. 엔티티 타입 분포 (train_annotated)\n")
    lines.append("| type | count |")
    lines.append("|---|---|")
    for t, c in results["train_annotated"]["entity_types"].most_common():
        lines.append(f"| {t} | {c:,} |")
    lines.append("")

    lines.append("## 4. 관계(Relation) 통계\n")
    for split in ["train_annotated", "train_distant", "dev"]:
        n_unique_rel = len(results[split]["relation_types"])
        lines.append(f"- **{split}**: 총 relation label {sum(results[split]['relation_types'].values()):,}개, "
                      f"고유 relation type {n_unique_rel}개 (전체 96개 중)")
    lines.append("")

    lines.append("### Top-10 relation types (train_annotated)\n")
    lines.append("| relation | id | count |")
    lines.append("|---|---|---|")
    for rid, c in results["train_annotated"]["relation_types"].most_common(10):
        lines.append(f"| {rel_info.get(rid, '?')} | {rid} | {c:,} |")
    lines.append("")

    lines.append("## 5. Intra- vs Inter-sentence 관계 비율\n")
    for split in ["train_annotated", "dev"]:
        intra = results[split]["intra_sent"]
        inter = results[split]["inter_sent"]
        total = intra + inter
        lines.append(f"- **{split}**: intra-sentence {intra:,} ({intra/total:.1%}), "
                      f"inter-sentence {inter:,} ({inter/total:.1%}) — 총 {total:,}개")
    lines.append("")
    lines.append("> inter-sentence 비율이 높을수록 문서 전체를 읽어야 관계를 추론할 수 있는 "
                  "multi-hop 성격이 강하다는 뜻입니다.\n")

    lines.append("## 6. Evidence 문장 수 분포 (train_annotated)\n")
    ev = np.array(results["train_annotated"]["evidence_lens"])
    lines.append(f"- 평균 {ev.mean():.2f}개, 중앙값 {np.median(ev):.0f}개, 최대 {ev.max()}개\n")

    lines.append("## 7. 생성된 그래프\n")
    lines.append("`EDA/figures/` 폴더 참고:")
    for p in sorted(FIG_DIR.glob("*.png")):
        lines.append(f"- {p.name}")

    (OUT_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nDone. See {OUT_DIR / 'summary.md'} and {FIG_DIR}/")


if __name__ == "__main__":
    main()
