"""안전 분류 결정 노드 모듈."""

from __future__ import annotations

from typing import Any

from firstsession.core.translate.state.translation_state import TranslationState

# 프로젝트에 존재한다고 가정 (참조에 명시됨)
from firstsession.core.translate.const.safeguard_messages import SafeguardMessage


class SafeguardDecisionNode:
    """안전 분류 결정을 담당하는 노드."""

    def run(self, state: TranslationState) -> TranslationState:
        print("[NODE] SafeguardDecisionNode")
        print("[DBG] service_state_id:", id(state))

        """PASS 여부와 오류 메시지를 기록한다."""
        label = (self._get(state, "safeguard_label") or "PASS").strip()

        if label == "PASS":
            self._set(state, "safeguard_passed", True)
            # 이전에 남아있을 수 있는 에러 메시지 제거(선택)
            self._set(state, "error_message", None)
            return state

        # 차단
        self._set(state, "safeguard_passed", False)

        # 라벨 -> 메시지 매핑
        message = self._map_label_to_message(label)
        self._set(state, "error_message", message)

        return state

    def _map_label_to_message(self, label: str) -> str:
        """SafeguardLabel -> SafeguardMessage 매핑."""
        mapping = {
            "PII": SafeguardMessage.PII.value,
            "HARMFUL": SafeguardMessage.HARMFUL.value,
            "PROMPT_INJECTION": SafeguardMessage.PROMPT_INJECTION.value,
        }

        # 예상 못한 값은 일반 차단 메시지(또는 PASS로 처리)로 방어
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
