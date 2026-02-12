"""안전 분류 실패 응답 노드 모듈."""

from __future__ import annotations

from typing import Any

from firstsession.core.translate.state.translation_state import TranslationState

# 프로젝트에 존재한다고 가정 (참조에 명시됨)
from firstsession.core.translate.const.safeguard_messages import SafeguardMessage


class SafeguardFailResponseNode:
    """안전 분류 실패 응답을 담당하는 노드."""

    def run(self, state: TranslationState) -> TranslationState:
        print("[NODE] SafeguardFailNode")
        print("[DBG] service_state_id:", id(state))

        """차단 응답을 구성한다."""
        label = (self._get(state, "safeguard_label") or "UNKNOWN").strip()
        passed = bool(self._get(state, "safeguard_passed", default=False))

        # PASS면 여기까지 올 일이 없지만 방어
        if passed or label == "PASS":
            return state

        # SafeguardDecisionNode에서 세팅된 error_message 우선 사용
        msg = self._get(state, "error_message")
        if not msg:
            msg = self._fallback_message(label)

        # 표준 응답 필드에 기록 (프로젝트 스키마 차이를 대비해 여러 키를 순차 세팅)
        self._set_if_exists(state, ["translated_text", "result_text", "output_text", "response_text"], msg)
        self._set_if_exists(state, ["status", "result_status"], "BLOCKED")
        self._set_if_exists(state, ["blocked_reason", "safeguard_reason"], label)

        # 로깅 규칙(간단 버전): state에 남겨서 상위에서 로거가 수집하도록
        # (민감 데이터 유출 방지: 원문 전체를 남기지 말고 일부만/길이만)
        text = self._get(state, "text", default="") or ""
        self._set_if_exists(
            state,
            ["audit_log", "logs", "safeguard_log"],
            {
                "event": "SAFEGUARD_BLOCK",
                "label": label,
                "text_len": len(text),
            },
        )

        return state

    def _fallback_message(self, label: str) -> str:
        mapping = {
            "PII": SafeguardMessage.PII.value,
            "HARMFUL": SafeguardMessage.HARMFUL.value,
            "PROMPT_INJECTION": SafeguardMessage.PROMPT_INJECTION.value,
        }
        return mapping.get(label, SafeguardMessage.GENERAL.value)

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

    def _set_if_exists(self, state: Any, keys: list[str], value: Any) -> None:
        """state가 dict면 첫 키에 저장, 객체면 존재하는 첫 속성에 저장."""
        if isinstance(state, dict):
            state[keys[0]] = value
            return

        for k in keys:
            if hasattr(state, k):
                setattr(state, k, value)
                return

        # 아무 필드도 없으면 fallback으로 첫 키에 강제로 set (디버깅 용이)
        setattr(state, keys[0], value)
