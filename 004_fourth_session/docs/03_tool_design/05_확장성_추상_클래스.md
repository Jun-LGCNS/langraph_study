# 05. 확장성을 위한 추상 클래스

## 이 챕터에서 배우는 것

- 공통 인터페이스(BaseTool)로 **확장 가능한 구조** 만들기
- Tool 추가 절차를 **팀 표준**으로 고정하기
- 공통 로깅/검증을 통해 **운영 일관성** 확보하기
- ToolSpec 기반으로 **자동 문서 생성**까지 연결하기

이 문서는 **초보자부터 고급자까지** 이어지는 튜토리얼입니다.  
초급자는 “왜 추상 클래스가 필요한지”를 이해하고, 고급자는 운영/확장 관점의 이점을 확인합니다.

추상 클래스는 “추상적이라 어려운 개념”이 아니라, **도구가 많아질 때 혼란을 막는 규칙**입니다.  
입력/출력/메타데이터를 통일해 두면, Agent/테스트/문서가 모두 안정됩니다.

또한 이 구조는 신규 Tool을 추가할 때 **검증/로깅/보안 규칙을 자동으로 적용**할 수 있게 합니다.  
즉, 개발자는 `run()`만 구현해도 운영 기준이 유지됩니다.

---

## 1. 왜 추상 클래스가 필요한가

Tool이 많아지면 인터페이스가 흔들립니다. 추상 클래스는 이를 막는 **안전장치**입니다.

- 입력/출력 구조가 도구마다 달라지는 문제 방지
- 이름/설명/버전 같은 메타데이터 불일치 방지
- 실행 시그니처가 통일되어 **Agent/테스트/문서**가 단순해짐

이 문서는 아래 순서로 차근차근 확장합니다.

1) 메타데이터 표준 정의 → 2) 공통 인터페이스 설계 → 3) 읽기 Tool 구현 → 4) 쓰기 Tool 구현 → 5) 공통 로깅/검증 추가 → 6) 자동 문서 생성

---

## 2. 설계 진행 순서

1. **메타데이터 표준을 정의한다**
   - name, description, version, read_only, idempotency_required

2. **입력/출력/에러 스키마를 고정한다**
   - Tool마다 다른 것은 입력값뿐이라는 전제를 만든다.

3. **실행 시그니처를 통일한다**
   - `run(args) -> dict` 형태로 고정한다.

4. **공통 검증/변환을 넣는다**
   - 입력 검증, 에러 변환, 로깅 훅을 BaseTool에 둔다.

5. **등록/조회 경로를 마련한다**
   - 팀 공용 Tool 카탈로그나 레지스트리에 연결한다.

---

## 3. BaseTool 설계 예시

```python
"""
목적: Tool 공통 인터페이스와 메타데이터 구조를 정의한다.
설명: 입력/출력/에러 스키마를 고정하고 실행 시그니처를 통일한다.
디자인 패턴: 템플릿 메서드
참조: Tool 카탈로그(레지스트리) 구조
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
from pydantic import BaseModel, Field


@dataclass(frozen=True)
class ToolSpec:
    """Tool 메타데이터 표준."""

    name: str
    description: str
    version: str
    read_only: bool
    idempotency_required: bool


class ErrorInfo(BaseModel):
    """표준 에러 정보."""

    code: str = Field(description="에러 코드")
    message: str = Field(description="사용자 메시지")
    retryable: bool = Field(description="재시도 가능 여부")


class BaseTool(ABC):
    """모든 Tool이 따라야 하는 기본 인터페이스."""

    spec: ToolSpec

    def validate(self, args: Any) -> None:
        """입력 검증 훅. 필요하면 오버라이드한다."""

    @abstractmethod
    def run(self, args: Any) -> dict:
        """Tool 실행 메서드."""
        raise NotImplementedError
```

---

## 4. 읽기 Tool 구현 예시

```python
"""
목적: BaseTool을 상속한 조회 Tool 구현 예시를 제공한다.
설명: 입력 스키마와 실행 로직을 분리한다.
디자인 패턴: 전략
참조: ToolSpec, ErrorInfo
"""

from pydantic import BaseModel, Field


class CustomerProfileArgs(BaseModel):
    """고객 프로필 조회 입력."""

    customer_id: str = Field(description="고객 ID")


class CustomerProfileTool(BaseTool):
    spec = ToolSpec(
        name="crm_customer_profile",
        description="CRM에서 고객 프로필을 조회한다",
        version="v1",
        read_only=True,
        idempotency_required=False,
    )

    def run(self, args: CustomerProfileArgs) -> dict:
        return {"customer_id": args.customer_id, "segment": "enterprise"}
```

---

## 5. 쓰기 Tool 구현 예시(멱등 키 포함)

```python
"""
목적: BaseTool을 상속한 쓰기 Tool 예시를 제공한다.
설명: 멱등 키로 중복 실행을 방지한다.
디자인 패턴: 커맨드
참조: ToolSpec, ErrorInfo
"""

from pydantic import BaseModel, Field


class ApprovalDraftArgs(BaseModel):
    """전자결재 초안 생성 입력."""

    title: str = Field(description="결재 제목")
    body: str = Field(description="결재 본문")
    idempotency_key: str = Field(description="중복 실행 방지 키")


class ApprovalDraftTool(BaseTool):
    spec = ToolSpec(
        name="approval_draft",
        description="전자결재 초안을 생성한다",
        version="v1",
        read_only=False,
        idempotency_required=True,
    )

    def run(self, args: ApprovalDraftArgs) -> dict:
        return {"draft_id": "APP-1001", "status": "created"}
```

---

## 6. BaseTool에 공통 로깅/검증 훅 추가 예시

공통 훅을 넣으면 모든 Tool에 **일관된 운영 규칙**을 적용할 수 있습니다.

```python
"""
목적: 공통 로깅/검증 훅 예시를 제공한다.
설명: validate와 run을 감싸 공통 로깅을 수행한다.
디자인 패턴: 템플릿 메서드
"""

from datetime import datetime


class LoggingBaseTool(BaseTool):
    """로깅과 검증 훅이 포함된 BaseTool."""

    def validate(self, args: Any) -> None:
        if args is None:
            raise ValueError("입력은 필수입니다")

    def execute(self, args: Any) -> dict:
        """공통 로깅 + 실행 래퍼."""

        self.validate(args)
        started_at = datetime.utcnow().isoformat()
        result = self.run(args)
        return {"result": result, "started_at": started_at}
```

실무 포인트

- `execute` 같은 공통 진입점을 두면 **로깅/메트릭**을 일관되게 수집할 수 있다.
- 각 Tool은 `run`만 구현해도 운영 기준이 적용된다.

사용 흐름 예시

```python
tool = CustomerProfileTool()
result = tool.execute(CustomerProfileArgs(customer_id="C-1001"))
```

---

## 7. ToolSpec 기반 자동 문서 생성 예시

ToolSpec과 스키마를 이용하면 **문서를 자동 생성**할 수 있습니다.

```python
"""
목적: ToolSpec 기반 문서 생성 예시를 제공한다.
설명: ToolSpec과 입력 스키마 정보를 문서 문자열로 변환한다.
디자인 패턴: 빌더
"""

from typing import Type
from pydantic import BaseModel


def build_tool_doc(tool: BaseTool, args_schema: Type[BaseModel]) -> str:
    """Tool 문서 문자열을 생성한다."""

    fields = []
    for name, field in args_schema.model_fields.items():
        fields.append(f"- {name}: {field.annotation}")

    return "\n".join(
        [
            f"이름: {tool.spec.name}",
            f"설명: {tool.spec.description}",
            f"버전: {tool.spec.version}",
            "입력 필드:",
            *fields,
        ]
    )
```

실무 포인트

- 문서와 코드가 **자동 동기화**되어 누락이 줄어든다.
- Tool이 늘어날수록 자동 문서의 효과가 커진다.

---

## 8. BaseTool을 LangChain Tool로 변환 예시

BaseTool로 만든 Tool을 LangChain Tool로 감싸면 **LangGraph 에이전트에 바로 연결**할 수 있습니다.

```python
"""
목적: BaseTool을 LangChain Tool로 변환하는 예시를 제공한다.
설명: ToolSpec과 입력 스키마를 이용해 래핑한다.
디자인 패턴: 어댑터
참조: BaseTool, ToolSpec
"""

from typing import Type
from pydantic import BaseModel
from langchain.tools import tool


def to_langchain_tool(tool_instance: BaseTool, args_schema: Type[BaseModel]):
    """BaseTool을 LangChain Tool로 변환한다."""

    @tool(tool_instance.spec.name)
    def _wrapper(**kwargs):
        """ToolSpec의 description을 설명으로 사용한다."""

        args = args_schema(**kwargs)
        return tool_instance.run(args)

    _wrapper.__doc__ = tool_instance.spec.description
    return _wrapper
```

실무 포인트

- BaseTool 구조를 유지하면서 **LangChain 생태계와 호환**된다.
- ToolSpec이 있으면 **설명/버전 정보**를 일관되게 전달할 수 있다.

---

## 9. Tool 추가 절차(팀 표준)

1. **입력/출력/에러 스키마 설계**
2. **ToolSpec 작성**
3. **BaseTool 구현**
4. **검증/에러 변환 정책 적용**
5. **문서/예시 업데이트**

---

## 10. 확장 시 자주 놓치는 부분

- Tool 이름 충돌(중복 name)
- 버전 정책 누락(v1/v2 구분 없음)
- read_only/idempotency_required 누락
- 입력 스키마 없이 문자열 기반 구현
- 에러 구조 불일치

---

## 11. 체크리스트

- BaseTool 인터페이스가 모든 Tool에 적용되는가?
- ToolSpec이 일관된 메타데이터를 제공하는가?
- 입력/출력/에러 스키마가 고정되어 있는가?
- 공통 로깅/검증 훅이 적용되는가?
- ToolSpec 기반 문서 생성이 가능한가?
- 신규 Tool 추가 절차가 문서화되어 있는가?
