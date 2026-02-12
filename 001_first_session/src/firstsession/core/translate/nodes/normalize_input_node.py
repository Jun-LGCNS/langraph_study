"""입력 정규화 노드 모듈."""

from __future__ import annotations

import json
import re
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from firstsession.core.translate.state.translation_state import TranslationState


class NormalizeInputNode:
    """입력 정규화를 담당하는 노드."""

    _MODEL_ID = "gemini-3-flash-preview"
    _MAX_TEXT_CHARS = 10_000

    _LANG_ALIAS = {
        "kr": "ko",
        "kor": "ko",
        "ko-kr": "ko",
        "en-us": "en",
        "en-uk": "en",
        "jp": "ja",
        "zh-cn": "zh-Hans",
        "zh-tw": "zh-Hant",
    }

    _RE_MULTI_SPACE = re.compile(r"[ \t]+")
    _RE_MULTI_NEWLINE = re.compile(r"\n{3,}")

    def __init__(self) -> None:
        self._llm = ChatGoogleGenerativeAI(
            model=self._MODEL_ID,
            temperature=0,
        )

    def run(self, state: TranslationState) -> TranslationState:
        print("[NODE] NormalizeInputNode")
        print("[DBG] service_state_id:", id(state))

        # ✅ dict 기반으로 읽기 (LangGraph가 dict로 넘기는 케이스 대응)
        text = self._get(state, "text", default="") or ""
        src = self._get(state, "source_language")
        tgt = self._get(state, "target_language")

        # 1) 공백 정리
        text = text.strip()
        text = self._RE_MULTI_SPACE.sub(" ", text)
        text = self._RE_MULTI_NEWLINE.sub("\n\n", text)

        # 2) 길이 제한
        if len(text) > self._MAX_TEXT_CHARS:
            text = text[: self._MAX_TEXT_CHARS]
            self._set(state, "warning_message", "Input text truncated by max length rule.")

        # 3) 언어 코드 정규화
        src = self._normalize_lang_code(src)
        tgt = self._normalize_lang_code(tgt)

        # 4) source_language 없으면 Gemini로 감지
        if not src and text:
            detected = self._detect_language(text)
            if detected:
                src = detected

        # 5) state 반영 (✅ dict에 쓰기)
        self._set(state, "text", text)
        self._set(state, "source_language", src)
        self._set(state, "target_language", tgt)

        return state

    # -------------------------
    # Helpers
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

    def _normalize_lang_code(self, lang: Optional[str]) -> Optional[str]:
        if not lang:
            return None

        lang = lang.strip().lower()
        lang = self._LANG_ALIAS.get(lang, lang)

        if not re.fullmatch(r"[a-z]{2,3}(?:-[A-Za-z]{2,8})?", lang):
            return None

        return lang

    def _detect_language(self, text: str) -> Optional[str]:
        prompt = (
            "Detect the language of the following text.\n"
            "Return ONLY valid JSON like:\n"
            '{"language": "ko"}\n\n'
            f"Text:\n{text}"
        )

        try:
            response = self._llm.invoke(
                [HumanMessage(content=prompt)]
            )

            content = response.content.strip()

            # LLM이 가끔 ```json 블록 감싸는 경우 대비
            content = content.replace("```json", "").replace("```", "").strip()

            data = json.loads(content)
            lang = data.get("language")

            return self._normalize_lang_code(lang)

        except Exception:
            return None
