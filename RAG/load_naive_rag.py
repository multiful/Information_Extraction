"""train_rag.csv(train_revised) + dev_revised.json 을 임베딩해 naive RAG용
Pinecone 인덱스(informationrag)에 적재.

informationextraction 인덱스(GraphRAG, relation triple 단위)와 대비되는 naive RAG
데모용 인덱스: 문서를 relation으로 구조화하지 않고 문장 하나하나를 그대로 청크로 삼아
임베딩한다. 그래서 단일홉 질의는 두 인덱스 모두 잘 찾지만, 멀티홉 질의는 문장 간
연결 정보가 없는 이 인덱스에서는 못 찾는 것을 데모에서 보여주는 용도.

주의: train_rag.csv는 train_annotated 기준 3,053개 문서만 담고 있어 dev_revised의
500개 문서(예: ROKETSAN, Lappeenranta)가 빠져 있다. informationextraction 인덱스는
train_revised+dev_revised(3,553개 문서)를 모두 담고 있으므로, 두 인덱스를 공정하게
비교하려면 dev_revised.json도 같은 방식(문장 단위, NLTK TreebankWordDetokenizer로
detokenize)으로 flatten해서 함께 적재해야 한다.

API 키는 코드에 하드코딩하지 않고 .env(레포 루트)에서 읽는다
(OPENAI_API_KEY, PINECONE_API_KEY). .env는 .gitignore에 등록되어 있음.

사용법:
    python RAG/load_naive_rag.py
"""

import csv
import hashlib
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from nltk.tokenize.treebank import TreebankWordDetokenizer
from openai import OpenAI
from pinecone import Pinecone

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

INDEX_NAME = "informationrag"  # Pinecone 인덱스 이름 (naive RAG 데모용)
EMBED_DIMENSIONS = 512  # 인덱스가 512차원으로 생성되어 있어 임베딩도 맞춰서 축소
CSV_PATH = ROOT / "docred_data" / "data" / "train_rag.csv"
DEV_REVISED_PATH = ROOT / "docred_data" / "data" / "dev_revised.json"

detokenizer = TreebankWordDetokenizer()

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


def iter_rows():
    """(title, sent_id, sentence, split) 튜플을 train_rag.csv + dev_revised.json 순서로 yield."""
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["sentence"].strip():
                yield row["title"], row["sent_id"], row["sentence"], "train_revised"

    with open(DEV_REVISED_PATH, "r", encoding="utf-8") as f:
        for doc in json.load(f):
            for sent_id, tokens in enumerate(doc["sents"]):
                sentence = detokenizer.detokenize(tokens)
                if sentence.strip():
                    yield doc["title"], str(sent_id), sentence, "dev_revised"


def load_to_pinecone(limit=None):
    print("🚀 Pinecone(informationrag) naive RAG 적재 시작...")
    # 문장이 짧아 OpenAI 호출은 크게 묶고(300개),
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

    for row_num, (title, sent_id, sentence, split) in enumerate(iter_rows()):
        if limit is not None and row_num >= limit:
            break

        try:
            original_id = f"{title}::{sent_id}"
            # Pinecone 벡터 ID는 ASCII만 허용하는데 title에는 비ASCII 문자가
            # 섞여 있어(예: József Mindszenty, Björk Digital) 그대로 쓸 수 없다.
            # ASCII 안전한 해시로 바꾸고 원본은 metadata에 보존한다.
            doc_id = hashlib.sha1(original_id.encode("utf-8")).hexdigest()

            metadata = {
                "title": title,
                "sent_id": int(sent_id),
                "original_id": original_id,
                "split": split,
                "text": sentence,
            }

            buffer.append((doc_id, sentence, metadata))

            if len(buffer) >= embed_batch_size:
                flush(buffer)
                buffer = []

        except Exception as e:
            print(f"❌ 에러 발생: {e}")
            continue

    # 남은 데이터가 있다면 마저 처리
    flush(buffer)

    print(f"✨ Pinecone(informationrag) 적재 완료! 총 {success_count}개 벡터가 저장되었습니다.")


if __name__ == "__main__":
    load_to_pinecone()
