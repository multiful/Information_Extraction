"""pinecone_upsert.jsonl의 {id, text, metadata}를 OpenAI 임베딩으로 벡터화해
Pinecone 인덱스에 upsert한다.

사전 준비:
    .env 에 아래 두 줄 추가
        OPENAI_API_KEY=...
        PINECONE_API_KEY=...

사용법:
    python Scripts/kg/upsert_pinecone.py
"""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent.parent
JSONL_PATH = ROOT / "pinecone_upsert.jsonl"

INDEX_NAME = "informationrag"
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536
BATCH_SIZE = 100


def load_rows():
    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def batched(iterable, size):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def main():
    from openai import OpenAI
    from pinecone import Pinecone, ServerlessSpec

    load_dotenv(ROOT / ".env")
    openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

    if INDEX_NAME not in [idx["name"] for idx in pc.list_indexes()]:
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(INDEX_NAME).status["ready"]:
            time.sleep(1)

    # 인덱스가 이미 존재하면 그 차원에 맞춰 임베딩해야 함 (다른 값으로 만들어져 있을 수 있음)
    dimensions = pc.describe_index(INDEX_NAME).dimension
    index = pc.Index(INDEX_NAME)

    n = 0
    for batch in batched(load_rows(), BATCH_SIZE):
        texts = [row["text"] for row in batch]
        resp = openai_client.embeddings.create(
            model=EMBED_MODEL, input=texts, dimensions=dimensions
        )
        vectors = [
            {
                "id": row["id"],
                "values": emb.embedding,
                "metadata": row["metadata"],
            }
            for row, emb in zip(batch, resp.data)
        ]
        index.upsert(vectors=vectors)
        n += len(vectors)
        print(f"upserted {n}")

    print(f"done: {n}개 벡터를 '{INDEX_NAME}' 인덱스에 upsert")


if __name__ == "__main__":
    main()
