"""번역 수행 노드 모듈."""

from __future__ import annotations

from typing import Any, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from firstsession.core.translate.state.translation_state import TranslationState


class TranslateNode:
    """번역 수행을 담당하는 노드.

    - 전략 패턴: (모델 기반 번역) 전략을 메서드로 캡슐화
    - 파이프라인 노드: state를 읽고 state에만 기록
    """

    _MODEL_ID = "gemini-3-flash-preview"

    def __init__(self) -> None:
        self._llm = ChatGoogleGenerativeAI(model=self._MODEL_ID, temperature=0)

    def run(self, state: TranslationState) -> TranslationState:
        print("[NODE] TranslateNode")
        print("[DBG] service_state_id:", id(state))

        text = self._get(state, "text", default="") or ""
        src = self._get(state, "source_language")
        tgt = self._get(state, "target_language")

        # 안전장치: 입력 없으면 빈 번역
        if not text.strip():
            self._set(state, "translated_text", "")
            return state

        # 언어 코드가 없으면 최소한의 기본값
        src = (src or "auto").strip()
        tgt = (tgt or "en").strip()

        translated = self._translate_with_gemini(text=text, source_language=src, target_language=tgt)

        # 상태 기록 규칙
        self._set(state, "translated_text", translated)
        self._set(state, "last_translation", translated)  # 재시도 로직에서 쓰기 좋게(있으면)
        print(state)
        return state

    def _translate_with_gemini(self, text: str, source_language: str, target_language: str) -> str:
        prompt = f"""
You are a strict translation engine.

Translate the text from {source_language} to {target_language}.

Rules:
- Output ONLY the translated text.
- Do NOT add explanations, notes, examples, or extra words.
- Do NOT add quotes, code fences, or markdown.
- Preserve meaning, tone, nuance, and formatting (line breaks, punctuation).
- Keep slang and profanity natural; do not censor or soften.
- If {source_language} is "auto", detect the source language automatically.
- If the input is already in {target_language}, return it unchanged.

Text:
{text}
""".strip()
        resp = self._llm.invoke([HumanMessage(content=prompt)])

        out = self._extract_text(resp)
        out = out.replace("```", "").strip()
        out = self._strip_wrapping_quotes(out)
        return out

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
