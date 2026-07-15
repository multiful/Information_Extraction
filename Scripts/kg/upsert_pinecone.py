"""`pinecone_upsert.jsonl`(export_pinecone.py 산출물)을 OpenAI로 임베딩해서
Pinecone 인덱스에 업서트한다.

사용법:
    python Scripts/kg/upsert_pinecone.py                # 전체 업서트
    python Scripts/kg/upsert_pinecone.py --limit 50      # 앞 50개만 (테스트용)
    python Scripts/kg/upsert_pinecone.py --dry-run       # API 호출 없이 건수/토큰만 확인

필요 환경변수 (`.env`): OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME.
선택: PINECONE_CLOUD(기본 aws), PINECONE_REGION(기본 us-east-1) — 인덱스가
아직 없을 때 생성할 서버리스 인덱스의 위치.
"""

import argparse
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INPUT = ROOT / "pinecone_upsert.jsonl"

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536


def read_rows(path, limit=None):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def chunked(items, size):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def embed_batch(client, texts, dimensions, retries=3):
    for attempt in range(retries):
        try:
            resp = client.embeddings.create(
                model=EMBED_MODEL, input=texts, dimensions=dimensions
            )
            return [d.embedding for d in resp.data]
        except Exception as e:
            if attempt == retries - 1:
                raise
            wait = 2**attempt
            print(f"  임베딩 실패 ({e}), {wait}초 후 재시도")
            time.sleep(wait)


def upsert_batch(index, vectors, retries=3):
    for attempt in range(retries):
        try:
            index.upsert(vectors=vectors)
            return
        except Exception as e:
            if attempt == retries - 1:
                raise
            wait = 2**attempt
            print(f"  업서트 실패 ({e}), {wait}초 후 재시도")
            time.sleep(wait)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")

    rows = read_rows(args.input, args.limit)
    print(f"{args.input.name}: {len(rows)}개 로드")

    if args.dry_run:
        approx_tokens = sum(len(r["text"]) for r in rows) // 4
        print(f"[dry-run] 임베딩/업서트를 실제로 호출하지 않음. 예상 토큰 수(대략): {approx_tokens:,}")
        return

    from openai import OpenAI
    from pinecone import Pinecone, ServerlessSpec

    openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index_name = os.environ["PINECONE_INDEX_NAME"]

    if index_name not in pc.list_indexes().names():
        print(f"Pinecone 인덱스 '{index_name}' 없음 -> 생성")
        pc.create_index(
            name=index_name,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=os.environ.get("PINECONE_CLOUD", "aws"),
                region=os.environ.get("PINECONE_REGION", "us-east-1"),
            ),
        )
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)

    dimensions = pc.describe_index(index_name).dimension
    print(f"인덱스 '{index_name}' 차원: {dimensions}")
    index = pc.Index(index_name)

    n = 0
    for batch in chunked(rows, args.batch_size):
        texts = [r["text"] for r in batch]
        embeddings = embed_batch(openai_client, texts, dimensions)
        vectors = [
            {"id": r["id"], "values": emb, "metadata": r["metadata"]}
            for r, emb in zip(batch, embeddings)
        ]
        upsert_batch(index, vectors)
        n += len(batch)
        print(f"  {n}/{len(rows)} 업서트 완료")

    print(f"완료: {n}개를 '{index_name}' 인덱스에 업서트")


if __name__ == "__main__":
    main()
