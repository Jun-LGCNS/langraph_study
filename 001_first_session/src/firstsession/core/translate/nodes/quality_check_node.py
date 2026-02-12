"""번역 품질 검사 노드 모듈."""

from __future__ import annotations

import re
from typing import Any, Literal, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from firstsession.core.translate.state.translation_state import TranslationState


YesNo = Literal["YES", "NO"]


class QualityCheckNode:
    """번역 품질 검사를 담당하는 노드.

    - 전략 패턴: (LLM 기반 QC) 전략을 메서드로 캡슐화
    - 파이프라인 노드: state를 읽고 state에만 기록
    """

    _MODEL_ID = "gemini-3-flash-preview"
    _ALLOWED: set[str] = {"YES", "NO"}

    def __init__(self) -> None:
        self._llm = ChatGoogleGenerativeAI(model=self._MODEL_ID, temperature=0)

    def run(self, state: TranslationState) -> TranslationState:
        print("[NODE] QualityCheckNode")
        print("[DBG] service_state_id:", id(state))

        src_text = self._get(state, "text", default="") or ""
        tgt_text = self._get(state, "translated_text", default="") or ""
        src_lang = self._get(state, "source_language") or "auto"
        tgt_lang = self._get(state, "target_language") or "en"

        # 입력이 비어있으면 통과로 간주(정책에 맞게 변경 가능)
        if not src_text.strip():
            self._set(state, "qc_passed", "YES")
            return state

        # 번역문이 비어있으면 실패
        if not tgt_text.strip():
            self._set(state, "qc_passed", "NO")
            self._set(state, "qc_reason", "empty_translation")
            return state

        raw = self._judge_yes_no(
            source_text=src_text,
            translated_text=tgt_text,
            source_language=src_lang,
            target_language=tgt_lang,
        )
        yn = self._normalize_yes_no(raw, fallback="NO")
        self._set(state, "qc_passed", yn)

        # 선택: QC 이유를 state에 남겨두면 디버깅이 쉬움 (루프 설계에도 도움)
        # (여기서는 최소 정보만)
        self._set(state, "qc_reason", None if yn == "YES" else "qc_failed")
        print(state)
        return state
    def _extract_text(self, resp) -> str:
        content = getattr(resp, "content", resp)

        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text" and isinstance(item.get("text"), str):
                        parts.append(item["text"])
                elif isinstance(item, str):
                    parts.append(item)
            return "\n".join(parts).strip()

        if isinstance(content, dict) and isinstance(content.get("text"), str):
            return content["text"].strip()

        return str(content).strip()
    def _judge_yes_no(self, source_text: str, translated_text: str, source_language: str, target_language: str) -> str:
        # yes/no 파서 패턴: “정답은 YES 또는 NO만”
        prompt = f"""
You are a translation quality gate for a production system.

Task:
Given SOURCE and TRANSLATION, decide if the translation is acceptable.

Return EXACTLY one token:
YES or NO

Criteria (fail with NO if any are true):
- Translation meaning is significantly wrong or missing key info.
- Adds harmful/extra content not present in source.
- Not in the requested target language ({target_language}).
- Garbage output, placeholders, or clearly incomplete.
- Format is severely broken (minor differences OK).

Notes:
- Minor paraphrasing is OK if meaning is preserved.
- Keep it strict: if unsure, answer NO.

SOURCE ({source_language}):
{source_text}

TRANSLATION ({target_language}):
{translated_text}
""".strip()

        resp = self._llm.invoke([HumanMessage(content=prompt)])
        return self._extract_text(resp).strip()

    def _normalize_yes_no(self, raw: str, fallback: YesNo = "NO") -> YesNo:
        s = (raw or "").strip().upper()

        # 흔한 형태 방어: "YES.", "Answer: YES", 코드블럭 등
        s = s.replace("```", "").strip()
        s = re.sub(r"^(ANSWER|LABEL)\s*:\s*", "", s).strip()

        token = s.split()[0] if s else ""
        if token in self._ALLOWED:
            return token  # type: ignore[return-value]
        return fallback

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
