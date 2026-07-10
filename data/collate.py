"""Optional DataLoader collate_fn, for use AFTER you've tokenized examples
yourself (e.g. via tokenization.py::tokenize_document in your own branch).
Not wired into DocREDataset -- that dataset yields raw, un-tokenized docs.

Dynamic-pads input_ids/attention_mask to the batch's longest doc.
entity_pos/entity_types/labels stay as plain Python lists (ragged across
docs), since how they're consumed depends on each branch's model."""

import torch


def make_collate_fn(pad_token_id: int):
    def collate(batch: list[dict]) -> dict:
        max_len = max(len(ex["input_ids"]) for ex in batch)
        input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        for i, ex in enumerate(batch):
            length = len(ex["input_ids"])
            input_ids[i, :length] = torch.tensor(ex["input_ids"], dtype=torch.long)
            attention_mask[i, :length] = torch.tensor(ex["attention_mask"], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "doc_id": [ex["doc_id"] for ex in batch],
            "title": [ex["title"] for ex in batch],
            "entity_pos": [ex["entity_pos"] for ex in batch],
            "entity_types": [ex["entity_types"] for ex in batch],
            "num_entities": [ex["num_entities"] for ex in batch],
            "labels": [ex["labels"] for ex in batch],
        }

    return collate
