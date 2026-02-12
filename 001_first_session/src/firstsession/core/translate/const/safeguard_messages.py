# 목적: 안전 분류에 대한 오류 메시지 Enum을 제공한다.
# 설명: PII/HARMFUL/PROMPT_INJECTION 대응 메시지를 표준화한다.
# 디자인 패턴: Enum
# 참조: docs/04_string_tricks/02_single_choice_파서.md

"""안전 분류 오류 메시지 Enum 모듈."""

from enum import Enum


class SafeguardMessage(Enum):
    """안전 분류 오류 메시지."""

    PII = "개인정보가 포함된 요청으로 번역을 진행할 수 없습니다."
    HARMFUL = "유해하거나 위험한 요청으로 번역을 진행할 수 없습니다."
    PROMPT_INJECTION = "보안 정책 위반 가능성이 있어 번역을 진행할 수 없습니다."
    GENERAL = "요청을 처리할 수 없습니다."
