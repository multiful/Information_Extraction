"""[후처리] triple confidence 기준 3분류 + 중간대 LLM(OpenAI) 배치 검증.

extract_triples가 만든 리치 메타데이터 triple을 confidence로 나눈다:
  - conf >= --high (기본 0.95)          : 바로 통과 (LLM 없이 채택)
  - --low <= conf < --high (0.80~0.95)  : LLM(OpenAI) 배치 검증 -> accept/reject
  - conf < --low (기본 0.80)            : 바로 폐기

최종 채택 = (통과분) + (LLM이 accept한 중간대). 폐기분·LLM reject분은 제외.
결과는 --out(기본 ..._v3.json)에 저장. 각 triple에 "filter" 메타(band/action/llm_reason) 부착.

LLM 검증은 (head, relation, tail)이 evidence 문장에 비추어 성립하는지 판단하며,
여러 triple을 한 번에 묶어(--batch) 호출해 API 호출 수를 줄인다.

API 키: 아래 OPENAI_API_KEY 를 채우거나 환경변수 OPENAI_API_KEY 로 주입.
키가 없으면 통과분만 저장하고 중간대는 건너뛴다(경고). --dry_run 은 분류만.

예:
    # 분류만 확인
    python -m Scripts.atlop.filter_triples --dry_run
    # 전체(키 필요): 통과 + LLM검증 -> v3
    OPENAI_API_KEY=sk-... python -m Scripts.atlop.filter_triples
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# ==================== 사용자 설정 ====================
OPENAI_API_KEY = ""              # <- 본인 OpenAI API 키 입력 (또는 환경변수 OPENAI_API_KEY)
MODEL = "gpt-5.4-mini"           # <- 사용할 OpenAI 모델명 (정확한 식별자로 수정)
# ====================================================

DEFAULT_IN = ROOT / "results" / "final" / "atlop_dream_revised_test_revised_triples_v2.json"


def _load_key():
    """키 우선순위: 코드 상수 -> 환경변수 -> 프로젝트 .env 파일(gitignore)."""
    if OPENAI_API_KEY:
        return OPENAI_API_KEY
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]
    envf = ROOT / ".env"
    if envf.exists():
        for line in envf.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("OPENAI_API_KEY") and "=" in line:
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def _evi_text(t):
    ev = t.get("evidence", [])
    return " ".join(ev) if isinstance(ev, list) else str(ev)


def verify_batch(client, model, items):
    """items: triple dict 리스트 -> [{"accept": bool, "reason": str}, ...] (같은 순서)."""
    lines = []
    for i, t in enumerate(items):
        lines.append(
            f'{i}. ({t["head"]["name"]}) --[{t["relation"]["name"]}]--> ({t["tail"]["name"]})\n'
            f'   근거: {_evi_text(t)}'
        )
    user_msg = (
        "아래 각 관계 triple이 주어진 '근거' 문장으로 뒷받침되어 사실로 성립하는지 판단하라.\n"
        "근거가 그 관계를 명확히 뒷받침하면 accept=true, 아니면(근거 부족/무관/오류) accept=false.\n"
        '반드시 JSON만 출력: {"results":[{"idx":0,"accept":true,"reason":"간단한 근거"}, ...]}\n\n'
        + "\n".join(lines)
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "너는 문서 관계추출 검증기다. 근거 문장만으로 관계의 참/거짓을 엄격히 판단한다."},
            {"role": "user", "content": user_msg},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    data = json.loads(resp.choices[0].message.content)
    by_idx = {r.get("idx"): r for r in data.get("results", [])}
    out = []
    for i in range(len(items)):
        r = by_idx.get(i, {})
        out.append({"accept": bool(r.get("accept", False)), "reason": r.get("reason", "no verdict")})
    return out


def build_argparser():
    p = argparse.ArgumentParser(description="triple confidence 3분류 + 중간대 LLM 검증")
    p.add_argument("--in", dest="inp", default=str(DEFAULT_IN), help="입력 리치 triple json (v2)")
    p.add_argument("--out", default="", help="출력 json (기본: 입력의 v2->v3)")
    p.add_argument("--high", type=float, default=0.95, help="이 값 이상은 바로 통과")
    p.add_argument("--low", type=float, default=0.80, help="이 값 미만은 바로 폐기")
    p.add_argument("--model", default=MODEL, help="OpenAI 모델명")
    p.add_argument("--batch", type=int, default=20, help="LLM 한 번에 검증할 triple 수")
    p.add_argument("--dry_run", action="store_true", help="LLM 호출 없이 분류 개수만")
    return p


def main():
    args = build_argparser().parse_args()
    inp = Path(args.inp)
    triples = json.load(open(inp, encoding="utf-8"))

    high = [t for t in triples if t["confidence"] >= args.high]
    mid = [t for t in triples if args.low <= t["confidence"] < args.high]
    low = [t for t in triples if t["confidence"] < args.low]
    print(f"[분류] 전체 {len(triples)}  |  >={args.high} 통과 {len(high)}  |  "
          f"[{args.low},{args.high}) LLM검증 {len(mid)}  |  <{args.low} 폐기 {len(low)}")

    kept = []
    for t in high:
        t["filter"] = {"band": "high", "action": "pass"}
        kept.append(t)

    if args.dry_run:
        print("[dry-run] 분류만 — LLM 호출·저장 생략.")
        return

    if mid:
        key = _load_key()
        if not key:
            print(f"[warn] OpenAI 키 없음 -> 중간대 {len(mid)}개 검증 건너뜀(v3엔 통과분만). "
                  f"키 넣고 다시 실행하면 검증분이 합쳐집니다.")
        else:
            from openai import OpenAI
            client = OpenAI(api_key=key)
            acc = rej = 0
            for s in range(0, len(mid), args.batch):
                chunk = mid[s: s + args.batch]
                try:
                    verdicts = verify_batch(client, args.model, chunk)
                except Exception as e:  # noqa: BLE001
                    print(f"  [LLM][err] {s}-{s + len(chunk)} 실패: {str(e)[:120]} -> 이 배치 reject 처리")
                    verdicts = [{"accept": False, "reason": f"llm_error: {str(e)[:60]}"}] * len(chunk)
                for t, v in zip(chunk, verdicts):
                    t["filter"] = {"band": "mid", "action": "accept" if v["accept"] else "reject",
                                   "llm_reason": v["reason"]}
                    if v["accept"]:
                        kept.append(t)
                        acc += 1
                    else:
                        rej += 1
                print(f"  [LLM] {min(s + args.batch, len(mid))}/{len(mid)}  (accept {acc} / reject {rej})", flush=True)
            print(f"[LLM] 검증 완료: accept {acc} / reject {rej}")

    # low(폐기)는 kept에 넣지 않음
    out = Path(args.out) if args.out else inp.with_name(inp.stem.replace("_v2", "") + "_v3.json"
                                                        if "_v2" in inp.stem else inp.stem + "_v3.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fp:
        json.dump(kept, fp, ensure_ascii=False, indent=1)
    n_high = sum(1 for t in kept if t["filter"]["band"] == "high")
    n_mid = sum(1 for t in kept if t["filter"]["band"] == "mid")
    print(f"[saved] {out}  (최종 채택 {len(kept)}: 통과 {n_high} + LLM검증통과 {n_mid})")


if __name__ == "__main__":
    main()
