"""
사용자 정의 도구(custom tools) 모음.

각 도구는 단일 함수로 구현되어 있으며, Llama 3.1 같은 로컬 모델이
호출할 수 있도록 JSON 스키마(`TOOL_SCHEMAS`)로도 정의되어 있습니다.

- query_wikipedia : 위키백과 API 질의
- query_arxiv     : arXiv API 질의
- celsius_to_fahrenheit : 섭씨 -> 화씨 변환
- save_text_file  : 입력을 텍스트 파일로 저장
- copy_file       : 파일 복사
"""

import os
import shutil
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET


# ---------------------------------------------------------------------------
# 1) 위키백과 API 질의 도구
# ---------------------------------------------------------------------------
def query_wikipedia(query: str, lang: str = "ko") -> str:
    """위키백과에서 검색어에 해당하는 문서 요약을 가져온다."""
    title = urllib.parse.quote(query)
    url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{title}"
    req = urllib.request.Request(url, headers={"User-Agent": "nlp-study-tool/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            import json

            data = json.loads(resp.read().decode("utf-8"))
        return data.get("extract", "요약을 찾을 수 없습니다.")
    except Exception as exc:  # noqa: BLE001
        return f"위키백과 질의 실패: {exc}"


# ---------------------------------------------------------------------------
# 2) arXiv API 질의 도구
# ---------------------------------------------------------------------------
def query_arxiv(query: str, max_results: int = 3) -> str:
    """arXiv에서 논문을 검색해 제목과 요약을 반환한다."""
    base = "http://export.arxiv.org/api/query"
    params = urllib.parse.urlencode(
        {"search_query": f"all:{query}", "start": 0, "max_results": max_results}
    )
    url = f"{base}?{params}"
    try:
        with urllib.request.urlopen(url, timeout=20) as resp:
            xml_text = resp.read().decode("utf-8")
    except Exception as exc:  # noqa: BLE001
        return f"arXiv 질의 실패: {exc}"

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(xml_text)
    entries = root.findall("atom:entry", ns)
    if not entries:
        return "검색 결과가 없습니다."

    lines = []
    for i, entry in enumerate(entries, 1):
        title = entry.findtext("atom:title", default="", namespaces=ns).strip()
        summary = entry.findtext("atom:summary", default="", namespaces=ns).strip()
        lines.append(f"{i}. {title}\n   {summary[:300]}...")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3) 섭씨 -> 화씨 변환 도구
# ---------------------------------------------------------------------------
def celsius_to_fahrenheit(celsius: float) -> str:
    """섭씨 온도를 화씨로 변환한다."""
    fahrenheit = float(celsius) * 9 / 5 + 32
    return f"{celsius}°C = {fahrenheit:.1f}°F"


# ---------------------------------------------------------------------------
# 4) 입력을 텍스트 파일로 저장하는 도구
# ---------------------------------------------------------------------------
def save_text_file(content: str, filename: str = "output.txt") -> str:
    """주어진 텍스트를 파일로 저장한다."""
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        return f"'{filename}' 파일로 저장했습니다. ({len(content)}자)"
    except Exception as exc:  # noqa: BLE001
        return f"파일 저장 실패: {exc}"


# ---------------------------------------------------------------------------
# 5) 파일을 복사하는 도구
# ---------------------------------------------------------------------------
def copy_file(src: str, dst: str) -> str:
    """원본 파일을 대상 경로로 복사한다."""
    if not os.path.exists(src):
        return f"원본 파일이 없습니다: {src}"
    try:
        shutil.copy2(src, dst)
        return f"'{src}' -> '{dst}' 복사 완료."
    except Exception as exc:  # noqa: BLE001
        return f"파일 복사 실패: {exc}"


# ---------------------------------------------------------------------------
# 도구 이름 -> 실제 함수 매핑 (모델이 호출하면 여기서 디스패치)
# ---------------------------------------------------------------------------
TOOL_REGISTRY = {
    "query_wikipedia": query_wikipedia,
    "query_arxiv": query_arxiv,
    "celsius_to_fahrenheit": celsius_to_fahrenheit,
    "save_text_file": save_text_file,
    "copy_file": copy_file,
}


# ---------------------------------------------------------------------------
# JSON 도구 정의
# OpenAI / Llama 3.1 의 function-calling 스키마 형식을 따른다.
# ---------------------------------------------------------------------------
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "query_wikipedia",
            "description": "위키백과에서 주제를 검색해 문서 요약을 가져온다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "검색할 주제"},
                    "lang": {
                        "type": "string",
                        "description": "위키백과 언어 코드 (예: ko, en)",
                        "default": "ko",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_arxiv",
            "description": "arXiv에서 논문을 검색해 제목과 요약을 반환한다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "검색 키워드"},
                    "max_results": {
                        "type": "integer",
                        "description": "가져올 논문 수",
                        "default": 3,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "celsius_to_fahrenheit",
            "description": "섭씨 온도를 화씨 온도로 변환한다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "celsius": {"type": "number", "description": "섭씨 온도"}
                },
                "required": ["celsius"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_text_file",
            "description": "주어진 텍스트를 텍스트 파일로 저장한다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "저장할 텍스트"},
                    "filename": {
                        "type": "string",
                        "description": "저장할 파일 이름",
                        "default": "output.txt",
                    },
                },
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "copy_file",
            "description": "원본 파일을 대상 경로로 복사한다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "src": {"type": "string", "description": "원본 파일 경로"},
                    "dst": {"type": "string", "description": "복사할 대상 경로"},
                },
                "required": ["src", "dst"],
            },
        },
    },
]
