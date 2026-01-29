# 02. Tool 예시 모음

## 이 챕터에서 배우는 것

- 조회/쓰기/비동기 Tool의 **실무형 예시** 이해
- Tool과 MCP의 역할을 **예시로 구분**하기
- 입력/출력/에러 스키마를 **일관되게 설계**하기

이 문서는 **초보자부터 고급자까지** 이어지는 튜토리얼입니다.  
초급자는 예제의 입력/출력 구조를 익히고, 고급자는 운영 관점(멱등/감사/정책)을 확인하세요.

---

## 1. 조회 Tool 예시

조회 Tool은 **데이터를 읽기만 하는 실행**입니다.  
하지만 실무에서는 “어디까지 읽을지”와 “얼마나 정확한지”를 명확히 해야 합니다.

### 1-1) 조회 Tool 설계 포인트

- **조회 범위 제한**: 날짜 범위, 고객 범위, 최대 건수를 고정한다.
- **정합성 수준**: 실시간/지연/캐시 여부를 문서화한다.
- **필드 최소화**: 필요한 컬럼만 반환해 민감 정보 노출을 줄인다.
- **페이지네이션**: 결과가 클 수 있으므로 page/cursor를 고려한다.
- **권한 연계**: 조회 가능 범위는 MCP/정책 엔진에서 제한한다.

```python
"""
목적: ERP 미결제 내역 조회 Tool 예시를 제공한다.
설명: 읽기 전용이며 입력/출력 스키마가 고정되어 있다.
디자인 패턴: 커맨드
"""

from pydantic import BaseModel, Field


class ErpBillingQueryArgs(BaseModel):
    """ERP 미결제 내역 조회 입력."""

    customer_id: str = Field(description="고객 ID")
    as_of_date: str = Field(description="기준일(YYYY-MM-DD)")
    top_k: int = Field(default=50, description="최대 조회 건수")


def erp_billing_query_tool(args: ErpBillingQueryArgs) -> dict:
    """ERP 미결제 내역을 조회했다고 가정한다."""

    return {
        "items": [
            {"invoice_id": "INV-001", "amount": 120000, "currency": "KRW"},
            {"invoice_id": "INV-002", "amount": 54000, "currency": "KRW"},
        ],
        "count": 2,
    }
```

실무 포인트

- **읽기 전용**임을 문서에 명시한다.
- 결과 수가 많을 수 있으므로 **페이지/건수 제한**을 둔다.
- 캐시 사용 시 **데이터 시점**을 응답에 포함한다.
- 조회 실패는 **표준 에러 코드**로 반환한다.

---

## 2. 쓰기 Tool 예시

쓰기 Tool은 **시스템 상태를 변경**하므로 설계 기준이 더 엄격합니다.  
실무에서는 “중복 실행”, “승인 절차”, “감사 추적”을 반드시 다룹니다.

### 2-1) 쓰기 Tool 설계 포인트

- **멱등 키 필수**: 재시도/중복 호출을 방지한다.
- **승인/워크플로우**: 고위험 작업은 승인 절차를 분리한다.
- **부분 성공 처리**: 여러 건 처리 시 성공/실패를 분리해 반환한다.
- **트랜잭션 경계**: 한 번에 묶을 범위를 명확히 정의한다.
- **감사 로그**: 누가 무엇을 변경했는지 기록한다.

#### 멱등 키를 이해하기

멱등 키(idempotency key)는 **“같은 요청을 여러 번 보내도 결과가 한 번만 적용되게 만드는 식별자”**입니다.  
네트워크 재시도나 중복 클릭으로 요청이 여러 번 들어와도 **실제 변경은 1회만** 일어나게 합니다.

- **무엇을 막는가**: 중복 생성, 이중 결제, 중복 승인 같은 사고
- **어떻게 동작하는가**:  
  1) 클라이언트/Agent가 **고유한 키**를 생성한다.  
  2) 서버는 키를 저장하고, **같은 키 요청은 이전 결과를 재사용**한다.  
  3) 결과가 이미 확정되었으면 **재실행 없이 응답만 반환**한다.
- **어디에 저장하는가**: DB/캐시에 `idempotency_key`와 결과를 함께 저장한다.
- **얼마나 유지하는가**: 정책에 따라 **TTL(유효 기간)** 을 둔다.
- **주의점**: 같은 키에 **다른 입력**이 들어오면 거부하거나 오류로 처리한다.

즉, 멱등 키는 **“중복 실행 방지용 안전장치”**이며, 쓰기 Tool에는 필수입니다.

#### 부분 처리(부분 성공)를 이해하기

부분 처리는 **“여러 건을 한 번에 처리할 때, 일부만 성공하고 일부는 실패하는 상황”**을 의미합니다.  
예를 들어 10건을 처리했는데 7건 성공, 3건 실패가 발생할 수 있습니다.

- **왜 필요한가**:  
  - 한 건의 실패 때문에 **전체를 실패로 만들면 운영 부담**이 커진다.  
  - 성공한 결과를 **버리면 비용**이 낭비된다.
- **어떻게 표현하는가**:  
  - 응답에 `success_items`와 `failed_items`를 분리해 반환한다.  
  - 실패 항목에는 **실패 사유/재시도 가능 여부**를 포함한다.
- **운영 기준**:  
  - 실패 항목만 재시도할지, 전체를 롤백할지 정책을 정한다.  
  - 부분 성공을 허용하는 경우 **사용자에게 명확히 고지**한다.

부분 처리는 **“실무에서 흔한 정상 상황”**이므로, 결과 구조에 반드시 반영해야 합니다.

부분 처리 응답 예시

```json
{
  "success_items": [
    {"draft_id": "APP-1001", "status": "created"},
    {"draft_id": "APP-1002", "status": "created"}
  ],
  "failed_items": [
    {"draft_id": "APP-1003", "error_code": "PERMISSION_DENIED", "retryable": false},
    {"draft_id": "APP-1004", "error_code": "TIMEOUT", "retryable": true}
  ],
  "summary": {
    "success_count": 2,
    "failure_count": 2
  }
}
```

참고: 부분 처리 시에는 **폴백 또는 재시도 전략이 있어야 한다**는 점을 문서에 명기한다.

```python
"""
목적: 전자결재 초안 생성 Tool 예시를 제공한다.
설명: 멱등 키로 중복 생성을 방지한다.
디자인 패턴: 커맨드
"""

from pydantic import BaseModel, Field


class ApprovalDraftArgs(BaseModel):
    """전자결재 초안 생성 입력."""

    actor_id: str = Field(description="요청자 ID")
    title: str = Field(description="결재 제목")
    body: str = Field(description="결재 본문")
    idempotency_key: str = Field(description="중복 생성 방지 키")


def approval_draft_tool(args: ApprovalDraftArgs) -> dict:
    """결재 초안을 생성했다고 가정한다."""

    return {"draft_id": "APP-1001", "status": "created"}
```

실무 포인트

- 쓰기 Tool은 **멱등 키**가 필수다.
- **권한 집행**은 Tool이 아니라 MCP/정책 엔진에서 강제한다.
- 재시도 가능 여부는 **에러 코드**로 명확히 구분한다.
- 변경 결과는 **감사 로그**와 연결되어야 한다.

---

## 3. 비동기 Tool 예시

잡 기반 Tool은 **즉시 결과를 반환하기 어려운 작업**을 처리할 때 사용합니다.  
예: 대용량 리포트 생성, 대규모 데이터 정합성 검사, 배치 계산, 대량 파일 변환

### 3-1) 어떻게 동작하는가

1. Tool이 **작업 요청**을 받아 `job_id`를 발급한다.  
2. 실제 처리는 **워커/배치**가 백그라운드에서 수행한다.  
3. Agent는 `job_id`로 **상태 조회 Tool**을 호출해 진행 상황을 확인한다.  
4. 완료되면 결과를 저장소에서 가져오거나, 완료 알림을 받는다.

```python
"""
목적: 대용량 리포트 생성 Tool 예시를 제공한다.
설명: 작업 요청을 생성하고 job_id를 반환한다.
디자인 패턴: 비동기 커맨드
"""

from pydantic import BaseModel, Field


class ReportJobArgs(BaseModel):
    """리포트 생성 요청 입력."""

    report_type: str = Field(description="리포트 유형")
    date_from: str = Field(description="시작일(YYYY-MM-DD)")
    date_to: str = Field(description="종료일(YYYY-MM-DD)")


def report_create_tool(args: ReportJobArgs) -> dict:
    """비동기 작업을 생성했다고 가정한다."""

    return {"job_id": "JOB-9001", "status": "queued"}
```

비동기 상태 조회 Tool 예시

```python
"""
목적: 비동기 작업 상태 조회 Tool 예시를 제공한다.
설명: job_id로 진행 상태를 조회한다.
디자인 패턴: 조회 커맨드
"""

from pydantic import BaseModel, Field


class JobStatusArgs(BaseModel):
    """작업 상태 조회 입력."""

    job_id: str = Field(description="작업 ID")


def job_status_tool(args: JobStatusArgs) -> dict:
    """작업 상태를 조회했다고 가정한다."""

    return {"job_id": args.job_id, "status": "running", "progress": 60}
```

실무 포인트

- Agent는 **job_id로 상태를 추적**해야 한다.
- MCP는 **동시성/레이트 리밋**을 통해 폭주를 막는다.
- 잡 기반 Tool은 **타임아웃/취소 정책**을 함께 정의한다.
- 결과는 **저장소에 기록**하고 상태 조회는 그 저장소를 바라보게 만든다.

---

## 4. MCP가 해야 할 일

MCP는 Tool의 내부 구현이 아니라 **발견/통제/관측**을 담당한다.

예: MCP의 도구 목록 응답(개념 예시)

```json
{
  "tools": [
    {
      "name": "erp_billing_query",
      "version": "v1",
      "input_schema": "ErpBillingQueryArgs",
      "read_only": true,
      "rate_limit_per_min": 60
    },
    {
      "name": "approval_draft",
      "version": "v1",
      "input_schema": "ApprovalDraftArgs",
      "read_only": false,
      "requires_idempotency_key": true
    }
  ]
}
```

구분 포인트

- Tool은 **실제 실행**을 담당한다.
- MCP는 **등록/권한/제한/감사**를 담당한다.
- 비동기 Tool은 **상태 조회 Tool**을 함께 등록한다.

---

## 5. SI 기업 관점 예시 케이스

### 5-1) “고객 미결제 내역 + 결재 초안” 시나리오

1. Agent가 “미결제 내역 조회”와 “결재 초안 생성”을 분리한다.
2. MCP가 권한을 확인한 뒤 Tool 목록을 노출한다.
3. 조회 Tool이 ERP에서 데이터를 가져온다.
4. 쓰기 Tool이 결재 초안을 생성한다.
5. MCP가 감사 로그를 남기고 Agent가 요약 응답을 만든다.

실무 포인트

- 조회/쓰기 Tool을 **명확히 분리**해야 승인 흐름이 안전해진다.
- MCP가 **권한과 감사 필드**를 강제해야 누락을 막을 수 있다.

---

## 6. 체크리스트

- 조회/쓰기/비동기 Tool의 입력/출력 스키마가 분리되어 있는가?
- Tool과 MCP의 역할이 예시에서 명확히 구분되어 있는가?
- 쓰기 Tool에 멱등 키가 포함되어 있는가?
- 비동기 작업의 상태 조회 Tool이 정의되어 있는가?

---

## 7. LangGraph/LangChain 연동 미니 예시

아래는 **Tool 등록 → 에이전트 연결 → 실행** 흐름을 한 번에 보는 예시입니다.

```python
"""
목적: LangChain Tool을 등록하고 LangGraph 기반 에이전트를 실행한다.
설명: @tool 데코레이터와 create_agent를 사용한다.
디자인 패턴: 커맨드
참조: LangChain Tool, LangGraph create_agent
"""

from langchain.tools import tool
from langchain.agents import create_agent


@tool
def quick_calc(a: float, b: float) -> float:
    """두 수를 더한다."""

    return a + b


agent = create_agent(model="provider:model-name", tools=[quick_calc])
result = agent.invoke(
    {"messages": [{"role": "user", "content": "3.5와 2.5를 더해줘"}]}
)
```

실무 포인트

- Tool 등록은 **함수 시그니처/힌트**가 명확해야 한다.
- Agent 호출 입력은 **messages 구조**를 따른다.
- 모델은 사용하는 제공자에 맞게 설정한다.
