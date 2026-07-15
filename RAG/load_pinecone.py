"""docred_data/data/pinecone_upsert_revised.jsonl 을 임베딩해 Pinecone 인덱스에 적재.

API 키는 코드에 하드코딩하지 않고 .env(레포 루트)에서 읽는다
(OPENAI_API_KEY, PINECONE_API_KEY). .env는 .gitignore에 등록되어 있음.

사용법:
    python RAG/load_pinecone.py
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

INDEX_NAME = "informationextraction"  # Pinecone 인덱스 이름
EMBED_DIMENSIONS = 512  # Pinecone 인덱스가 512차원으로 생성되어 있어 임베딩도 맞춰서 축소
JSONL_PATH = ROOT / "docred_data" / "data" / "pinecone_upsert_revised.jsonl"

# OpenAI 클라이언트는 OPENAI_API_KEY 환경변수를 자동으로 읽음
openai_client = OpenAI()
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(INDEX_NAME)


def get_embeddings(texts, model="text-embedding-3-small"):
    """텍스트 리스트를 한 번의 API 호출로 임베딩 벡터 리스트로 변환 (OpenAI 호출 횟수 절감)"""
    response = openai_client.embeddings.create(
        input=texts, model=model, dimensions=EMBED_DIMENSIONS
    )
    return [d.embedding for d in response.data]


def load_to_pinecone(limit=None):
    print("🚀 Pinecone 벡터 데이터베이스 적재 시작...")
    # 텍스트가 짧아(평균 ~50토큰) OpenAI 호출은 크게 묶고(300개),
    # Pinecone 업서트는 자체 권장 배치 크기(100개)로 잘게 나눠서 보낸다.
    embed_batch_size = 300
    upsert_batch_size = 100
    success_count = 0
    buffer = []  # [(doc_id, text, metadata), ...]

    def flush(rows):
        nonlocal success_count
        if not rows:
            return
        try:
            vectors = get_embeddings([text for _, text, _ in rows])
        except Exception as e:
            print(f"❌ 임베딩 배치 에러 발생 ({len(rows)}건 스킵): {e}")
            return

        for i in range(0, len(rows), upsert_batch_size):
            chunk_rows = rows[i:i + upsert_batch_size]
            chunk_vectors = vectors[i:i + upsert_batch_size]
            upsert_data = [
                (doc_id, vector, metadata)
                for (doc_id, _, metadata), vector in zip(chunk_rows, chunk_vectors)
            ]
            try:
                index.upsert(vectors=upsert_data)
                success_count += len(upsert_data)
                print(f"🔄 {success_count}개 벡터 적재 성공...")
            except Exception as e:
                print(f"❌ 업서트 배치 에러 발생 ({len(upsert_data)}건 스킵): {e}")

    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            if limit is not None and line_num >= limit:
                break

            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
                doc_id = item["id"]
                text_content = item["text"]
                metadata = item["metadata"]

                # 중요: Pinecone 메타데이터에는 실제 text(본문)도 속성으로 들고 있어야
                # 나중에 검색했을 때 원본 문장을 읽어서 RAG 답변을 만들 수 있습니다.
                metadata["text"] = text_content

                buffer.append((doc_id, text_content, metadata))

                if len(buffer) >= embed_batch_size:
                    flush(buffer)
                    buffer = []

            except Exception as e:
                print(f"❌ 에러 발생: {e}")
                continue

        # 남은 데이터가 있다면 마저 처리
        flush(buffer)

    print(f"✨ Pinecone 적재 완료! 총 {success_count}개 벡터가 저장되었습니다.")


if __name__ == "__main__":
    load_to_pinecone()
