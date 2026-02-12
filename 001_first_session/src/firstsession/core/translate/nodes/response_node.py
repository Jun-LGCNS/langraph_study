"""응답 구성 노드 모듈."""

from __future__ import annotations

from typing import Any

from firstsession.core.translate.state.translation_state import TranslationState


class ResponseNode:
    """응답 구성을 담당하는 노드."""

    def run(self, state: TranslationState) -> TranslationState:
        print("[NODE] ResponseNode")
        print("[DBG] service_state_id:", id(state))

        """
        우선순위:
        1) error_message가 있으면 차단/에러 응답으로 정리
        2) 아니면 translated_text를 성공 응답으로 정리
        """
        error_message = self._get(state, "error_message")
        translated_text = self._get(state, "translated_text", default="")

        if error_message:
            # 차단/에러 응답
            self._set(state, "translated_text", str(error_message))
            self._set(state, "status", "ERROR")
            self._set(state, "success", False)
            return state

        # 성공 응답
        self._set(state, "translated_text", str(translated_text or ""))
        self._set(state, "status", "OK")
        self._set(state, "success", True)

        # 불필요한 필드 정리(선택): dict일 때만 prune
        # (TranslationState가 dataclass/pydantic이면 굳이 삭제하지 않는 편이 안전)
        if isinstance(state, dict):
            keep = {
                "translated_text",
                "status",
                "success",
                # 디버깅/추적용으로 남기고 싶으면 추가
                "safeguard_label",
                "qc_passed",
                "retry_count",
            }
            for k in list(state.keys()):
                if k not in keep:
                    state.pop(k, None)

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
