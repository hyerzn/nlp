"""
사용자 정의 도구를 사용하는 에이전트.

동작 방식
---------
1. tools.py 의 JSON 도구 정의(TOOL_SCHEMAS)를 시스템 프롬프트에 포함시킨다.
2. 사용자의 질문을 모델에 보낸다.
3. 모델이 도구가 필요하다고 판단하면, 아래 형식의 JSON 만 출력한다.
       {"name": "도구이름", "arguments": {...}}
4. 이 코드가 JSON 을 파싱해 실제 파이썬 함수를 실행하고,
   그 결과를 다시 모델에 전달해 최종 답변을 생성한다.

사전 준비
---------
    # 1) Ollama 설치 후 모델 내려받기
    ollama pull llama3.1
    # 2) Ollama 서버 실행 (보통 자동 실행됨)
    ollama serve

실행
----
    python agent.py          # 5개 도구를 모두 호출하는 테스트 실행
    python agent.py "직접 입력한 질문"
"""

import json
import re
import sys
import urllib.request

try:
    from dotenv import find_dotenv, load_dotenv
    load_dotenv(find_dotenv())
except ImportError:
    pass

from tools import TOOL_REGISTRY, TOOL_SCHEMAS

try:
    from langsmith import traceable
except ImportError:
    def traceable(*dargs, **dkwargs):
        def decorator(func):
            return func
        # @traceable 와 @traceable(run_type=...) 두 형태 모두 지원
        if len(dargs) == 1 and callable(dargs[0]):
            return dargs[0]
        return decorator

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5"


SYSTEM_PROMPT = """당신은 도구를 사용할 수 있는 한국어 AI 비서입니다.
사용할 수 있는 도구 목록(JSON 스키마)은 다음과 같습니다.

{tools}

지침:
- 사용자의 요청을 처리하기 위해 도구가 필요하면, 다른 설명 없이 아래 형식의 JSON 한 줄만 출력하세요.
  {{"name": "<도구이름>", "arguments": {{<인자>}}}}
- 도구가 필요 없으면 평범한 자연어로 답하세요.
- 한 번에 하나의 도구만 호출하세요.
""".format(tools=json.dumps(TOOL_SCHEMAS, ensure_ascii=False, indent=2))


@traceable(run_type="llm", name="ollama_chat")
def call_llama(messages: list) -> str:
    """Ollama chat API 를 호출해 assistant 메시지 내용을 반환한다."""
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0, "top_p": 1},
    }
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["message"]["content"]


def extract_tool_call(text: str):
    """모델 출력에서 {"name":..., "arguments":...} JSON 을 추출한다."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if isinstance(obj, dict) and "name" in obj and obj["name"] in TOOL_REGISTRY:
        return obj
    return None


@traceable(run_type="chain", name="tool_agent")
def run(user_input: str) -> str:
    """사용자 입력을 받아 (필요하면) 도구를 호출하고 최종 답변을 만든다."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]

    first = call_llama(messages)
    call = extract_tool_call(first)

    if call is None:
        return first  # 도구가 필요 없는 일반 답변

    name = call["name"]
    args = call.get("arguments", {}) or {}
    print(f"  [도구 호출] {name}({args})")

    # 각 도구 호출도 추적에 별도 span 으로 남도록 traceable 로 감싼다.
    traced_tool = traceable(run_type="tool", name=name)(TOOL_REGISTRY[name])
    result = traced_tool(**args)
    print(f"  [도구 결과] {result}")

    # 도구 실행 결과를 모델에 다시 전달해 자연어 답변을 생성
    messages.append({"role": "assistant", "content": first})
    messages.append(
        {"role": "user", "content": f"도구 '{name}'의 실행 결과: {result}\n이 결과를 바탕으로 사용자에게 한국어로 답변하세요."}
    )
    return call_llama(messages)


# ---------------------------------------------------------------------------
# 각 도구가 호출되도록 유도하는 테스트 질문
# ---------------------------------------------------------------------------
TEST_QUESTIONS = [
    "위키백과에서 '자연어 처리'에 대해 알려줘.",                       # query_wikipedia
    "arXiv에서 'large language model' 관련 논문을 찾아줘.",            # query_arxiv
    "섭씨 36.5도는 화씨로 몇 도야?",                                   # celsius_to_fahrenheit
    "'안녕하세요, 도구 테스트입니다.' 라는 내용을 note.txt 파일로 저장해줘.",  # save_text_file
    "note.txt 파일을 note_backup.txt 로 복사해줘.",                   # copy_file
]


def main():
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        print(f"[질문] {question}")
        print(f"[답변] {run(question)}")
        return

    for q in TEST_QUESTIONS:
        print("=" * 70)
        print(f"[질문] {q}")
        try:
            print(f"[답변] {run(q)}")
        except Exception as exc:  # noqa: BLE001
            print(f"[오류] {exc}  (Ollama 서버와 llama3.1 모델이 실행 중인지 확인하세요)")


if __name__ == "__main__":
    main()
