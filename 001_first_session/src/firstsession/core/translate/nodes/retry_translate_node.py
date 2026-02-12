"""재번역 노드 모듈."""

from __future__ import annotations

from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from firstsession.core.translate.state.translation_state import TranslationState


class RetryTranslateNode:
    """재번역을 담당하는 노드.

    QC 실패 시 더 엄격한 지시(누락/언어/형식/환각 방지)로 번역을 복구한다.
    """

    _MODEL_ID = "gemini-3-flash-preview"

    def __init__(self) -> None:
        self._llm = ChatGoogleGenerativeAI(model=self._MODEL_ID, temperature=0)

    def run(self, state: TranslationState) -> TranslationState:
        print("[NODE] RetryTranslateNode")
        print("[DBG] service_state_id:", id(state))


        text = self._get(state, "text", default="") or ""
        prev = self._get(state, "translated_text", default="") or ""
        src = (self._get(state, "source_language") or "auto").strip()
        tgt = (self._get(state, "target_language") or "en").strip()

        # retry_count 갱신 (루프 안전장치)
        retry_count = int(self._get(state, "retry_count", default=0) or 0) + 1
        self._set(state, "retry_count", retry_count)

        # 입력이 비어있으면 그대로
        if not text.strip():
            self._set(state, "translated_text", "")
            return state

        improved = self._retry_translate_with_gemini(
            source_text=text,
            previous_translation=prev,
            source_language=src,
            target_language=tgt,
        )

        self._set(state, "translated_text", improved)
        self._set(state, "last_translation", improved)
        print(state)
        return state

    def _retry_translate_with_gemini(
        self,
        source_text: str,
        previous_translation: str,
        source_language: str,
        target_language: str,
    ) -> str:
        prompt = f"""
You are a senior professional translator fixing a failed translation.

Goal:
Produce a HIGH-QUALITY translation from {source_language} to {target_language}.

Rules (very important):
- Output ONLY the final corrected translation text.
- No explanations, no bullet points, no quotes, no code fences.
- Must be in {target_language}.
- Preserve ALL meaning; do not omit details.
- Do not add information that is not in the source.
- Preserve formatting (line breaks, lists) as much as possible.

SOURCE TEXT:
{source_text}

PREVIOUS TRANSLATION (for reference; it failed QC):
{previous_translation}

Now output the corrected translation:
""".strip()

        resp = self._llm.invoke([HumanMessage(content=prompt)])
        out = self._extract_text(resp)
        out = out.replace("```", "").strip()
        return self._strip_wrapping_quotes(out)
    
    def _extract_text(self, resp) -> str:
        # langchain 응답은 보통 AIMessage
        content = getattr(resp, "content", resp)

        # 1) 이미 문자열이면 끝
        if isinstance(content, str):
            return content.strip()

        # 2) list[dict] 형태(지금 네 케이스)
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text" and isinstance(item.get("text"), str):
                        parts.append(item["text"])
                elif isinstance(item, str):
                    parts.append(item)
            return "\n".join(parts).strip()

        # 3) dict면 text 키 우선
        if isinstance(content, dict) and isinstance(content.get("text"), str):
            return content["text"].strip()

        # 4) fallback
        return str(content).strip()
    
    def _strip_wrapping_quotes(self, s: str) -> str:
        if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
            return s[1:-1].strip()
        return s

    # -------------------------
    # dict/object 공용 get/set
    # -------------------------

    def _get(self, state: Any, key: str, default: Any = None) -> Any:
        if isinstance(state, dict):
            return state.get(key, default)
        return getattr(state, key, default)

    def _set(self, state: Any, key: str, value: Any) -> None:
        if isinstance(state, dict):
            state[key] = value
        else:
            setattr(state, key, value)
