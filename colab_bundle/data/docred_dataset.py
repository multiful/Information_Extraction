"""Shared DocRED Dataset -- this IS the team's branch point.

Returns raw, un-tokenized document structure only (title / sents / vertexSet /
labels). No tokenizer, no model choice baked in anywhere here -- that's
intentional. Everyone forks off from this same raw data.

From here, each branch picks its own tokenizer/model and does its own
tokenization + subword alignment + entity-pair construction + classifier.
`tokenization.py` in this same folder has a reusable, tokenizer-agnostic
subword-alignment helper (`tokenize_document`) you can call in *your own*
script with whatever tokenizer you chose (RoBERTa, BERT, DeBERTa, ...) -- it
is not called automatically here.

Usage (run from the project root):

    from data.docred_dataset import DocREDataset

    ds = DocREDataset("train_annotated")
    doc = ds[0]
    doc["title"], doc["sents"], doc["vertexSet"], doc["labels"]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import docred_io

from torch.utils.data import Dataset


class DocREDataset(Dataset):
    def __init__(self, split: str):
        assert split in docred_io.SPLITS, f"split must be one of {docred_io.SPLITS}"
        self.split = split
        self.docs = docred_io.load_split(split)

    def __len__(self) -> int:
        return len(self.docs)

    def __getitem__(self, idx: int) -> dict:
        doc = self.docs[idx]
        return {
            "doc_id": idx,
            "title": doc.get("title"),
            "sents": doc["sents"],            # list[sentence][word] -- raw word tokens
            "vertexSet": doc["vertexSet"],     # list[entity][mention] = {"name","sent_id","pos","type"}
            # raw DocRED format, vertexSet-index based: [{"r": "P17", "h": 0, "t": 4, "evidence": [...]}]
            # test split has no "labels" key at all -> [].
            "labels": doc.get("labels", []),
        }


if __name__ == "__main__":
    # quick smoke test: python data/docred_dataset.py
    ds = DocREDataset("dev")
    print(f"dev split: {len(ds)} docs")
    doc = ds[0]
    print("title:", doc["title"])
    print("num_entities:", len(doc["vertexSet"]), "num_sents:", len(doc["sents"]))
    print("vertexSet[0][0]:", doc["vertexSet"][0][0])
    print("labels[0]:", doc["labels"][0] if doc["labels"] else None)
