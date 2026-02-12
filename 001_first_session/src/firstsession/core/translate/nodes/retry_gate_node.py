"""재번역 게이트 노드 모듈."""

from __future__ import annotations

from typing import Any

from firstsession.core.translate.state.translation_state import TranslationState


class RetryGateNode:
    """재번역 가능 여부를 판단하는 노드.

    이 노드는 "분기 결정을 위한 상태 기록"만 담당한다.
    (실제 분기는 TranslateGraph의 add_conditional_edges에서 수행)
    """

    def run(self, state: TranslationState) -> TranslationState:
        print("[NODE] RetryNode")
        print("[DBG] service_state_id:", id(state))

        qc_passed = (self._get(state, "qc_passed") or "NO").strip().upper()

        retry_count = int(self._get(state, "retry_count", default=0) or 0)
        max_retry_count = int(self._get(state, "max_retry_count", default=0) or 0)

        # 기본값(설정이 비어있으면 1회 재시도 허용)
        if max_retry_count <= 0:
            max_retry_count = 1
            self._set(state, "max_retry_count", max_retry_count)

        # QC 통과면 재시도 불필요
        if qc_passed == "YES":
            self._set(state, "retry_allowed", False)
            self._set(state, "retry_reason", None)
            return state

        # QC 실패면 재시도 가능 여부 판단
        if retry_count < max_retry_count:
            self._set(state, "retry_allowed", True)
            self._set(state, "retry_reason", "qc_failed_retry_available")
        else:
            self._set(state, "retry_allowed", False)
            self._set(state, "retry_reason", "max_retry_reached")

        return state

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
