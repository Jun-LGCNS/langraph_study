# 목적: 번역 서비스 계층을 제공한다.
# 설명: 요청 모델을 번역하고 응답 모델로 변환한다.
# 디자인 패턴: 서비스 레이어 패턴
# 참조: firstsession/api/translate/router/translate_router.py

"""번역 서비스 모듈."""

from firstsession.api.translate.model.translation_request import TranslationRequest
from firstsession.api.translate.model.translation_response import TranslationResponse
from firstsession.core.translate.graphs.translate_graph import TranslateGraph
from firstsession.core.translate.state.translation_state import TranslationState

class TranslationService:
    """번역 요청을 처리하는 서비스."""

    def __init__(self, graph: TranslateGraph) -> None:
        """서비스 의존성을 초기화한다.

        Args:
            graph: 번역 그래프 실행기.
        """
        self.graph = graph

    def translate(self, request: TranslationRequest) -> TranslationResponse:
        state = {
            "source_language": request.source_language,
            "target_language": request.target_language,
            "text": request.text,
            "retry_count": 0,
            "max_retry_count": 1,
        }

        result = self.graph.run(state)

        if isinstance(result, dict):
            translated = result.get("translated_text", "")
            src = result.get("source_language") or request.source_language
            tgt = result.get("target_language") or request.target_language
        else:
            translated = getattr(result, "translated_text", "")
            src = getattr(result, "source_language", None) or request.source_language
            tgt = getattr(result, "target_language", None) or request.target_language
            
        print("[DBG] service_state_id:", id(state))

        return TranslationResponse(
            source_language=src,
            target_language=tgt,
            translated_text=translated,
        )
    
        raise NotImplementedError("번역 서비스 처리 로직을 구현해야 합니다.")
