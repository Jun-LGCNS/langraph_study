# 04. 스키마 설계와 힌트 작성

## 이 챕터에서 배우는 것

- 입력/출력 스키마를 **인터페이스 기준**으로 설계하기
- LLM이 오용하지 않도록 **힌트와 제약**을 넣기
- 에러 스키마까지 포함한 **실무형 문서화**
- 스키마 변경/버전 관리로 **호환성 유지**하기

이 문서는 **초보자부터 고급자까지** 이어지는 튜토리얼입니다.  
초급자는 “스키마가 왜 필요한지”를 이해하고, 중급 이상은 “운영 실패를 줄이는 설계 기준”을 확인합니다.

스키마와 힌트는 단순 문서가 아니라 **실행 안전장치**입니다.  
입력이 흐트러지면 Tool은 예상치 못한 동작을 하고, 그 결과는 Agent의 판단을 왜곡합니다.

특히 LLM 기반 Tool 호출에서는 **힌트의 품질이 곧 입력 품질**을 결정합니다.  
따라서 “어떤 값을 넣어야 하는지”를 사람보다 더 명확하게 써야 합니다.

---

## 1. 스키마 설계는 “인터페이스 설계”다

스키마는 Tool의 계약이 아니라 **인터페이스**입니다. 한 번 정하면 쉽게 바꾸지 않습니다.

### 1-1) 설계 진행 순서

1. **업무 목표와 경계를 정의한다**
2. **입력 필드를 나열하고 필수/선택을 구분한다**
3. **단위/범위/허용 값을 명시한다**
4. **에러 코드와 실패 조건을 정한다**
5. **출력 구조를 고정한다**
6. **버전 정책을 추가한다**

### 1-2) 실무 설계 기준

- **모호한 타입 금지**: `string`으로 모든 것을 받지 않는다.
- **최소 필드 원칙**: 꼭 필요한 필드만 노출한다.
- **범위/제한 강제**: 서버에서 반드시 검증한다.
- **읽기/쓰기 구분**: 읽기 전용인지, 상태 변경인지 명확히 한다.
- **에러 구조화**: 예외가 아니라 표준 구조로 반환한다.

---

## 2. 입력 스키마 예시

```python
"""
목적: 입력 스키마와 힌트 예시를 제공한다.
설명: 범위/허용 값/단위를 명확히 한다.
디자인 패턴: 인터페이스 기반 설계
"""

from enum import Enum
from pydantic import BaseModel, Field


class Currency(str, Enum):
    """허용 통화 코드."""

    KRW = "KRW"
    USD = "USD"


class BillingQueryArgs(BaseModel):
    """미결제 내역 조회 입력 스키마."""

    customer_id: str = Field(min_length=1, max_length=50, description="고객 ID")
    date_from: str = Field(description="시작일(YYYY-MM-DD)")
    date_to: str = Field(description="종료일(YYYY-MM-DD)")
    currency: Currency = Field(description="통화 코드")
    top_k: int = Field(default=50, ge=1, le=200, description="최대 조회 건수")
    cursor: str | None = Field(default=None, description="페이지 커서")
```

실무 포인트

- `top_k`는 **범위를 명시**하고 서버에서 강제한다.
- 날짜 형식은 **정규식 또는 파서**로 검증한다.
- `cursor`를 두면 **대용량 조회**에 안전하다.

---

## 3. 출력/에러 스키마 예시

```python
"""
목적: 출력과 에러 스키마 예시를 제공한다.
설명: 성공/실패의 구조를 고정한다.
디자인 패턴: 인터페이스 기반 설계
"""

from pydantic import BaseModel, Field


class ErrorInfo(BaseModel):
    """표준 에러 정보."""

    code: str = Field(description="에러 코드")
    message: str = Field(description="사용자에게 보여줄 메시지")
    retryable: bool = Field(description="재시도 가능 여부")


class BillingQueryResult(BaseModel):
    """미결제 내역 조회 결과."""

    items: list[dict]
    count: int
    next_cursor: str | None = None
    data_as_of: str | None = Field(default=None, description="데이터 기준 시각")
    trace_id: str | None = Field(default=None, description="추적 ID")
    error: ErrorInfo | None = None
```

실무 포인트

- 에러는 **예외가 아니라 구조**로 반환한다.
- `retryable`은 재시도 정책의 핵심 신호다.
- `trace_id`는 관측/감사 연결을 위한 필수 단서다.

---

## 4. 힌트 작성 가이드

힌트는 LLM이 “잘못된 입력”을 만들지 않도록 돕는 장치입니다.

- **입력 목적을 1문장으로 설명**한다.
- **허용 값/범위/단위**를 반드시 포함한다.
- **금지 규칙**을 명시한다(예: 음수 불가, 날짜 역전 금지).
- **권한 조건**이 있다면 힌트에 포함한다.
- **예시를 1개** 이상 제공한다.
- LangChain의 `@tool`은 **docstring을 설명으로 사용**하므로, 설명 문장을 명확히 쓴다.

좋은 힌트 vs 나쁜 힌트

| 구분 | 예시 | 문제/효과 |
|---|---|---|
| 나쁜 힌트 | “필요한 값을 입력하세요” | 모호해서 오류 유발 |
| 나쁜 힌트 | “아무 값이나 가능” | 운영 장애 가능 |
| 좋은 힌트 | “YYYY-MM-DD 형식, 과거 날짜만 허용” | 제약이 명확 |
| 좋은 힌트 | “top_k는 1~200 사이 정수” | 범위가 명확 |

---

## 5. Tool 호출 입력 규칙(실무 관점)

Tool을 호출할 때는 **스키마 기준을 지키는 입력**이 반드시 포함되어야 합니다.

- **필수 필드 누락 금지**
- **허용 값 위반 금지**
- **읽기/쓰기 구분 준수**
- **쓰기 Tool은 멱등 키 포함**
- **대용량 조회는 cursor 또는 top_k 포함**

예시: Tool 호출 입력(개념)

```json
{
  "tool_name": "erp_billing_query",
  "arguments": {
    "customer_id": "C-1001",
    "date_from": "2025-01-01",
    "date_to": "2025-01-31",
    "currency": "KRW",
    "top_k": 50
  }
}
```

참고: Tool 등록 메타데이터와 MCP 노출 방식은 **MCP 챕터에서 별도 설명**한다.

---

## 6. 데코레이터 기반 Tool 구현 예시

실무에서는 **데코레이터로 Tool 메타데이터를 정리**해 일관성을 높입니다.  
아래 예시는 Tool 구현과 메타데이터 정의를 함께 두는 단순한 예입니다.

```python
"""
목적: 데코레이터 기반 Tool 구현 예시를 제공한다.
설명: 메타데이터/힌트를 구조적으로 등록한다.
디자인 패턴: 데코레이터
"""

from typing import Callable
from pydantic import BaseModel, Field

TOOL_REGISTRY: list[dict] = []


def tool(
    *,
    name: str,
    version: str,
    description: str,
    read_only: bool,
    idempotency_required: bool,
    input_schema: type,
    output_schema: type,
    error_schema: type,
    hints: dict,
    example_request: dict,
    example_response: dict,
) -> Callable:
    """Tool 메타데이터를 등록하는 데코레이터."""

    def decorator(func: Callable) -> Callable:
        TOOL_REGISTRY.append(
            {
                "name": name,
                "version": version,
                "description": description,
                "read_only": read_only,
                "idempotency_required": idempotency_required,
                "input_schema": input_schema.__name__,
                "output_schema": output_schema.__name__,
                "error_schema": error_schema.__name__,
                "hints": hints,
                "example_request": example_request,
                "example_response": example_response,
            }
        )
        return func

    return decorator


class ErrorInfo(BaseModel):
    """표준 에러 정보."""

    code: str = Field(description="에러 코드")
    message: str = Field(description="사용자 메시지")
    retryable: bool = Field(description="재시도 가능 여부")


class BillingQueryArgs(BaseModel):
    """미결제 내역 조회 입력."""

    customer_id: str = Field(description="고객 ID")
    date_from: str = Field(description="시작일(YYYY-MM-DD)")
    date_to: str = Field(description="종료일(YYYY-MM-DD)")
    top_k: int = Field(default=50, description="최대 조회 건수")


class BillingQueryResult(BaseModel):
    """미결제 내역 조회 결과."""

    items: list[dict]
    count: int
    error: ErrorInfo | None = None


@tool(
    name="erp_billing_query",
    version="v1",
    description="ERP 미결제 내역을 조회한다",
    read_only=True,
    idempotency_required=False,
    input_schema=BillingQueryArgs,
    output_schema=BillingQueryResult,
    error_schema=ErrorInfo,
    hints={
        "customer_id": "고객 ID, 최대 50자",
        "date_from": "YYYY-MM-DD, 과거 날짜만 허용",
        "date_to": "YYYY-MM-DD, date_from 이후",
        "top_k": "1~200 사이 정수",
    },
    example_request={
        "customer_id": "C-1001",
        "date_from": "2025-01-01",
        "date_to": "2025-01-31",
        "top_k": 50,
    },
    example_response={
        "items": [{"invoice_id": "INV-001", "amount": 120000, "currency": "KRW"}],
        "count": 1,
        "error": None,
    },
)
def erp_billing_query_tool(args: BillingQueryArgs) -> dict:
    """ERP 미결제 내역을 조회한다고 가정한다."""

    return {"items": [], "count": 0, "error": None}
```

실무 포인트

- 데코레이터는 **등록 누락**을 막고 표준 메타데이터를 강제한다.
- description은 **요약**, 상세 제약은 **hints**로 분리한다.
- 도구 카탈로그가 필요하면 `TOOL_REGISTRY`를 활용한다.

---

## 7. 스키마와 운영 정책의 연결

스키마는 단순한 형식 정의가 아니라 **운영 정책의 기준**입니다.

- **검증 실패는 즉시 차단**한다.
- **검증 실패 로그**를 남겨 품질 이슈를 추적한다.
- **에러 코드**로 재시도/폴백 여부를 결정한다.
- **읽기/쓰기 구분**은 권한/감사 정책과 연결된다.

---

## 8. 스키마 변경 절차(실무)

1. **추가 변경은 옵션 필드로만** 진행한다.
2. **삭제/변경은 새 버전**으로 승격한다.
3. 문서/예시/테스트를 함께 갱신한다.
4. 구버전은 **유예 기간** 후 폐기한다.

---

## 9. 체크리스트

- 입력/출력/에러 스키마가 모두 고정되어 있는가?
- 허용 값, 단위, 범위가 명확한가?
- 힌트가 모호하지 않고 실행 기준을 제공하는가?
- 스키마 변경 절차와 유예 정책이 정의되어 있는가?
