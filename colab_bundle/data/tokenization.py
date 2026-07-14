"""Optional, per-branch tokenization helper -- NOT used by DocREDataset.

DocREDataset (docred_dataset.py) is the one thing every team member shares;
it hands out raw, un-tokenized documents. Tokenizer/model choice is where
branches diverge, so it lives here instead, as a plain function you call
yourself with whatever tokenizer *you* picked (RoBERTa, BERT, DeBERTa, ...).

`tokenize_document()` itself is tokenizer-agnostic: it only relies on the
standard HF fast-tokenizer contract (`is_split_into_words=True` +
`.word_ids()`), so it works unmodified regardless of which model you use it
with. It concatenates a document's sentences into one flat word sequence,
tokenizes once, and for each gold mention (vertexSet entry) computes where
its words ended up in the subword sequence -- so any downstream pooling
strategy (mention average, entity marker, attention, ...) can slice
`input_ids` / hidden states directly. This module only produces indices, it
does not run any model.

`load_tokenizer()` below is just an example for a RoBERTa branch -- swap
`model_name` (and drop `add_prefix_space`, which only RoBERTa/GPT-2-family
byte-level-BPE tokenizers need for pre-tokenized input) for your own model,
or skip this helper entirely and call `AutoTokenizer.from_pretrained(...)`
yourself.
"""

import logging

from transformers import AutoTokenizer, PreTrainedTokenizerFast

logger = logging.getLogger(__name__)

MAX_LENGTH = 512


def load_tokenizer(model_name: str) -> PreTrainedTokenizerFast:
    # add_prefix_space=True: required by RoBERTa/GPT-2-style byte-level BPE
    # tokenizers when feeding them pre-tokenized word lists
    # (is_split_into_words=True). Drop it if you're using a WordPiece
    # tokenizer (BERT, DeBERTa, ...) -- it's not needed there.
    return AutoTokenizer.from_pretrained(model_name, use_fast=True, add_prefix_space=True)


def _flatten_words(sents: list[list[str]]) -> tuple[list[str], list[tuple[int, int]]]:
    """Concatenate all sentences into one word list.
    Returns (words, word_to_sentpos) where word_to_sentpos[global_idx] = (sent_id, idx_in_sent).
    """
    words: list[str] = []
    word_to_sentpos: list[tuple[int, int]] = []
    for sent_id, sent in enumerate(sents):
        for idx_in_sent, w in enumerate(sent):
            words.append(w)
            word_to_sentpos.append((sent_id, idx_in_sent))
    return words, word_to_sentpos


def tokenize_document(
    doc: dict,
    tokenizer: PreTrainedTokenizerFast,
    max_length: int = MAX_LENGTH,
) -> dict:
    """Tokenize one DocRED document (whole document as a single sequence).

    Returns a dict with:
      input_ids, attention_mask       : list[int], length <= max_length
      truncated                       : bool, True if the doc did not fit in max_length
      entity_pos                      : list[list[(start, end)]]
                                         outer = vertexSet order, inner = per-mention
                                         subword span [start, end) into input_ids.
                                         A mention that fell outside the truncated
                                         region is simply omitted from its entity's list.
      entity_types                    : list[str], one type per entity (first mention's type)
      dropped_mentions                : int, mentions lost to truncation (0 for the
                                         large majority of docs -- see EDA/summary.md,
                                         ~0.5-0.8% of docs exceed 512 subwords)
    """
    words, word_to_sentpos = _flatten_words(doc["sents"])
    sentpos_to_word = {sp: i for i, sp in enumerate(word_to_sentpos)}

    encoding = tokenizer(
        words,
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
    )
    word_ids = encoding.word_ids()

    # global word idx -> (first subword idx, last subword idx + 1), only for words
    # that survived truncation.
    word_subword_span: dict[int, tuple[int, int]] = {}
    for subword_idx, wid in enumerate(word_ids):
        if wid is None:
            continue
        if wid not in word_subword_span:
            word_subword_span[wid] = (subword_idx, subword_idx + 1)
        else:
            start, _ = word_subword_span[wid]
            word_subword_span[wid] = (start, subword_idx + 1)

    entity_pos: list[list[tuple[int, int]]] = []
    entity_types: list[str] = []
    dropped_mentions = 0

    for cluster in doc["vertexSet"]:
        mention_spans: list[tuple[int, int]] = []
        for mention in cluster:
            sent_id = mention["sent_id"]
            start_w, end_w = mention["pos"]  # word-level [start, end) within that sentence
            global_start = sentpos_to_word[(sent_id, start_w)]
            global_end = sentpos_to_word[(sent_id, end_w - 1)]  # inclusive last word

            if global_start not in word_subword_span or global_end not in word_subword_span:
                dropped_mentions += 1
                continue

            sub_start = word_subword_span[global_start][0]
            sub_end = word_subword_span[global_end][1]
            mention_spans.append((sub_start, sub_end))

        entity_pos.append(mention_spans)
        entity_types.append(cluster[0]["type"])

    truncated = len(words) != sum(len(s) for s in doc["sents"]) or dropped_mentions > 0
    if truncated:
        logger.warning(
            "doc %r truncated at max_length=%d, dropped %d mention(s)",
            doc.get("title"), max_length, dropped_mentions,
        )

    return {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "truncated": truncated,
        "entity_pos": entity_pos,
        "entity_types": entity_types,
        "dropped_mentions": dropped_mentions,
    }


if __name__ == "__main__":
    # example: how a RoBERTa branch would combine the shared DocREDataset
    # with its own tokenizer choice. python data/tokenization.py
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from docred_dataset import DocREDataset

    ds = DocREDataset("dev")
    doc = ds[0]  # raw, un-tokenized

    tokenizer = load_tokenizer("roberta-base")
    tok = tokenize_document(doc, tokenizer, MAX_LENGTH)
    print("title:", doc["title"])
    print("input_ids len:", len(tok["input_ids"]))
    start, end = tok["entity_pos"][0][0]
    print("entity 0 mention 0 decoded:", repr(tokenizer.decode(tok["input_ids"][start:end]).strip()))
