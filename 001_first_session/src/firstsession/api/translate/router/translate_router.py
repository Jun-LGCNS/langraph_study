# 목적: 번역 API 라우터를 제공한다.
# 설명: /api/v1/translate 경로에 번역 엔드포인트를 등록한다.
# 디자인 패턴: 라우터 팩토리 패턴
# 참조: firstsession/api/translate/const/api.py

"""번역 API 라우터 모듈."""

from fastapi import APIRouter

from firstsession.api.translate.const.api import (
    API_V1_PREFIX,
    TRANSLATE_PREFIX,
    TRANSLATE_TAG,
)
from firstsession.api.translate.model.translation_request import TranslationRequest
from firstsession.api.translate.model.translation_response import TranslationResponse
from firstsession.api.translate.service.translation_service import TranslationService


class TranslateRouter:
    """번역 API 라우터를 구성한다."""

    def __init__(self, service: TranslationService) -> None:
        """라우터와 의존성을 초기화한다.

        Args:
            service: 번역 서비스.
        """

        self.router = APIRouter(
        prefix = f"{API_V1_PREFIX}{TRANSLATE_PREFIX}",
        tags = [f"{TRANSLATE_TAG}"]
        )
        self.service = service
        self.router.add_api_route(
            path="",
            endpoint=self.translate,
            methods=["POST"],
            response_model=TranslationResponse,
        )


    def translate(self, request: TranslationRequest) -> TranslationResponse:
        """번역 요청을 처리한다.

        Args:
            request: 번역 요청 데이터.

        Returns:
            TranslationResponse: 번역 결과.
        """
        return self.service.translate(request)

