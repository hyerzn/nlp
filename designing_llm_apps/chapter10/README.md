# Chapter 10 — 사용자 정의 도구(Custom Tools)

Llama 3.1(혹은 로컬에서 호출 가능한 다른 모델)에서 직접 만든 도구를 사용하도록 구성한 예제입니다.
다섯 개의 도구를 **JSON 형식**으로 정의해 **시스템 프롬프트에 포함**시키고, 모델이 출력한 JSON
도구 호출을 파싱해 실제 파이썬 함수를 실행합니다.

## 파일 구성

| 파일 | 설명 |
|------|------|
| `tools.py` | 5개 도구의 구현 함수 + JSON 스키마(`TOOL_SCHEMAS`) + 디스패치 매핑(`TOOL_REGISTRY`) |
| `agent.py` | Ollama(Llama 3.1) 호출, 도구 호출 파싱/실행, 테스트 질문 실행 |

## 도구 목록

1. `query_wikipedia` — 위키백과 API 질의
2. `query_arxiv` — arXiv API 질의
3. `celsius_to_fahrenheit` — 섭씨 → 화씨 변환
4. `save_text_file` — 입력을 텍스트 파일로 저장
5. `copy_file` — 파일 복사

## 실행 방법

```bash
# 1) Ollama 설치 후 모델 내려받기
ollama pull llama3.1

# 2) Ollama 서버 실행 (대부분 자동 실행됨)
ollama serve

# 3) 5개 도구를 각각 유도하는 테스트 질문 실행
python agent.py

# 또는 직접 질문 입력
python agent.py "섭씨 100도는 화씨로 몇 도야?"
```

> 도구 함수(`tools.py`)만 따로 테스트하려면 Ollama 없이도 실행할 수 있습니다.
> ```bash
> python -c "import tools; print(tools.celsius_to_fahrenheit(36.5))"
> ```

## 각 도구를 유도하는 테스트 질문

`agent.py`의 `TEST_QUESTIONS`에 정의되어 있습니다.

| 질문 | 호출되는 도구 |
|------|---------------|
| 위키백과에서 '자연어 처리'에 대해 알려줘. | `query_wikipedia` |
| arXiv에서 'large language model' 관련 논문을 찾아줘. | `query_arxiv` |
| 섭씨 36.5도는 화씨로 몇 도야? | `celsius_to_fahrenheit` |
| '...' 라는 내용을 note.txt 파일로 저장해줘. | `save_text_file` |
| note.txt 파일을 note_backup.txt 로 복사해줘. | `copy_file` |
