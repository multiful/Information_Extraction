"""LLM 호출 결과 로컬 캐싱 — 실험을 반복 실행할 때 같은 입력에 대한 API 재호출 방지.

주의: 재현성 검증을 위해 같은 질문을 n번 반복 샘플링(다수결 판정)할 때는 호출부에서
repeat 인덱스를 키에 반드시 포함시켜야 한다. 안 그러면 첫 호출 결과가 캐싱되어
2번째/3번째 반복이 새 샘플이 아니라 같은 답을 재사용하게 되고, "답변이 흔들리는지"
검증 자체가 무의미해진다.

사용법:
    from cache import cache_key, cached

    def my_llm_call(question, repeat=0):
        def compute():
            return openai_client.chat.completions.create(...).choices[0].message.content
        return cached(cache_key("my_llm_call", question, repeat), compute)
"""

import hashlib
import json
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)


def cache_key(*parts):
    """캐시 키를 구성할 값들(문자열/숫자/리스트 등, JSON 직렬화 가능해야 함)을 해시."""
    raw = json.dumps(parts, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def cached(key, compute_fn):
    """key로 캐시를 조회하고, 없으면 compute_fn()을 호출해 결과를 저장 후 반환."""
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    result = compute_fn()
    path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return result


def cache_stats():
    files = list(CACHE_DIR.glob("*.json"))
    return {"n_entries": len(files), "total_bytes": sum(f.stat().st_size for f in files)}
