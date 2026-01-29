# 03. Planning 개념 정리

## 이 챕터에서 배우는 것

- Agent가 만든 Plan을 **절차적으로 실행**하기 위한 설계 기준
- 계획을 **실행 인터페이스**로 만드는 스키마/의존성/정책 설계
- Plan 검증(스키마/도구/권한/비용)과 실행 매핑 방법
- 실무형 예제(절차 실행 기반)와 체크리스트

---

## 1. Planning은 “절차 실행을 위한 인터페이스”다

Plan은 단순한 아이디어가 아니라 **실행 가능한 인터페이스**입니다.  
특히 절차적 실행을 전제로 할 때, 다음 조건을 반드시 만족해야 합니다.
이 관점이 없으면 “좋은 계획처럼 보이는 문장”이 실제 실행에서 실패합니다.
즉, 계획은 실행 엔진이 **기계적으로 해석**할 수 있어야 합니다.
계획이 구조화될수록 실패 원인도 명확해집니다.

- **순서가 명확**해야 한다
- **의존성이 명시**되어야 한다
- **입력/출력 연결**이 정의되어야 한다
- **정책/권한/비용 제한**이 Plan에 포함되어야 한다

이 조건이 없으면 실행 단계에서 계획을 해석할 수 없고, 실패 대응도 불가능합니다.

---

## 2. 절차 실행 Plan 스키마

아래는 절차 실행을 전제로 한 Plan 예시입니다.  
핵심은 `depends_on`, `output_key`, `policy` 같은 메타정보입니다.
이 메타정보는 실행 로직이 “무엇을 언제 실행해야 하는지”를 이해하기 위한 최소 단서입니다.
또한 이 메타정보는 디버깅과 재시도를 가능하게 합니다.
따라서 Plan 스키마를 정의할 때 가장 먼저 결정해야 합니다.

```json
{
  "strategy": "sequential",
  "policy": {
    "merge": "strict",
    "timeout_ms": 3000,
    "retry": 2,
    "budget": 0.2
  },
  "steps": [
    {
      "id": 1,
      "action": "tool_call",
      "tool": "read_file",
      "input": {"path": "/data/input.txt"},
      "output_key": "file_content"
    },
    {
      "id": 2,
      "action": "tool_call",
      "tool": "summarize_text",
      "input": {"text_ref": "${file_content}"},
      "depends_on": [1],
      "output_key": "summary"
    },
    {
      "id": 3,
      "action": "tool_call",
      "tool": "write_file",
      "input": {"path": "/data/summary.txt", "content_ref": "${summary}"},
      "depends_on": [2]
    }
  ]
}
```

---

## 3. LLM으로 절차 실행 Plan 생성하기

절차 실행 Plan은 LLM이 생성하되, **의존성과 출력 키를 강제**해야 합니다.
강제하지 않으면 LLM이 자연어 설명만 생성하고, 실행 엔진은 이를 해석할 수 없습니다.
즉, LLM은 “계획을 작성하는 도구”이지 “규칙을 보장하는 도구”가 아닙니다.
따라서 프롬프트와 출력 파서가 함께 설계되어야 합니다.
이 조합이 있어야 계획이 실행과 연결됩니다.

```python
"""
목적: 절차 실행 Plan을 LLM으로 생성한다.
설명: depends_on/output_key를 포함하도록 강제한다.
디자인 패턴: Planner
"""

from textwrap import dedent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


_PROMPT = dedent(
    """
    너는 절차 실행 Plan을 작성한다.
    출력은 JSON만 허용한다.
    규칙:
    - steps는 순서가 있는 배열이어야 한다.
    - 의존성이 있는 단계는 depends_on을 포함한다.
    - 다음 단계가 이전 결과를 쓰면 output_key와 *_ref를 사용한다.
    질문: {{question}}
    Tool 목록: {{tool_list}}
    """
).strip()
_SEQUENTIAL_PLAN_PROMPT = PromptTemplate.from_template(_PROMPT)


def build_sequential_plan(question: str, tool_list: str) -> str:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return (_SEQUENTIAL_PLAN_PROMPT | llm | StrOutputParser()).invoke(
        {"question": question, "tool_list": tool_list}
    )
```

---

## 4. Plan 설계 핵심 요소

이 섹션은 Plan을 “실행 가능한 구조”로 만드는 핵심 요소를 분해합니다.  
각 요소는 실행 엔진의 안정성과 직결됩니다.
의존성, 입력/출력 연결, 정책은 서로 독립이 아닙니다.
어느 하나가 빠지면 나머지도 의미가 약해집니다.
따라서 세 요소를 동시에 설계해야 합니다.

### 3-1) 의존성(Dependency)

- `depends_on`으로 실행 순서를 강제한다.
- 의존성이 있는 단계는 **병렬 실행 불가**.

### 3-2) 입력/출력 연결

- `output_key`는 후속 step에서 참조할 수 있는 **공식 출력 변수**다.
- `*_ref` 형식(`text_ref`, `content_ref`)으로 이전 결과를 명시한다.

### 3-3) 정책(Policy)

- `timeout_ms`, `retry`, `budget`, `merge`는 **실행 정책**이다.
- 절차 실행에서는 `merge`가 거의 의미 없지만, 실패 처리 기준으로 사용된다.

---

## 5. Plan 검증: 실행 전 필수 관문

Plan이 실행 가능하려면 아래 검증이 필요합니다.
검증이 없으면 실패가 실행 단계에서 발생해 비용과 시간을 낭비하게 됩니다.
특히 비용이 큰 Tool이나 외부 시스템 호출에서는 검증이 필수입니다.
검증 결과는 재계획의 근거가 됩니다.
결국 검증은 “실행 전에 하는 투자”입니다.

- **스키마 검증**: 필수 필드 존재 여부
- **도구 검증**: Tool 등록 여부, 입력 스키마 호환성
- **정책 검증**: 권한/비용/보안 정책 준수
- **의존성 검증**: cycle(순환) 여부, 존재하지 않는 step 참조 여부

### 4-1) 검증 예시

```python
"""
목적: 절차 실행 Plan의 기본 검증을 수행한다.
설명: 의존성/도구/입력 참조를 검사한다.
디자인 패턴: Validator
"""

def validate_sequential_plan(plan: dict, tool_registry: dict) -> bool:
    steps = plan.get("steps", [])
    if not steps:
        return False

    step_ids = {s.get("id") for s in steps}
    for step in steps:
        # 1) 도구 존재 여부
        tool = step.get("tool")
        if tool and tool not in tool_registry:
            return False

        # 2) 의존성 유효성
        depends_on = step.get("depends_on", [])
        if any(dep not in step_ids for dep in depends_on):
            return False

        # 3) 입력 참조 형식 검사
        for v in step.get("input", {}).values():
            if isinstance(v, str) and v.startswith("${") and not v.endswith("}"):
                return False

    return True
```

---

## 6. Plan → 실행 매핑

절차 실행은 **Step 순서대로 실행**하고, 각 결과를 State에 저장합니다.
이 매핑이 명확해야 재시도와 디버깅이 가능합니다.
매핑 규칙이 없으면 결과가 흩어져 추적이 어렵습니다.
따라서 output_key와 state 저장 규칙을 먼저 정해야 합니다.
이 원칙이 지켜지면 재현성이 높아집니다.

```mermaid
flowchart LR
    S1[Step 1 실행] --> S2[Step 2 실행]
    S2 --> S3[Step 3 실행]
    S3 --> DONE[완료]
```

이 흐름에서 중요한 점:

- 실행 중 Plan을 바꾸지 않는다.
- 각 Step 결과는 `output_key`로 저장한다.
- 실패 시 정책(retry/timeout)에 따라 중단 또는 폴백.

---

## 7. 절차 실행 Plan의 실무 포인트

아래 포인트는 실제 현장에서 자주 겪는 실패를 줄이기 위한 기준입니다.
이 기준은 팀 내 리뷰 체크리스트로도 활용할 수 있습니다.
Plan이 복잡해질수록 기준의 중요성은 더 커집니다.
따라서 계획 단계에서 이 포인트를 점검해야 합니다.
이 습관이 운영 장애를 줄이는 데 도움 됩니다.

- **작업 순서가 곧 책임 경계**가 된다.
- `output_key`가 명확해야 디버깅이 가능하다.
- 의존성 구조는 **DAG처럼 해석**할 수 있어야 한다.

---

## 8. 실무 체크리스트

- Plan에 `depends_on`과 `output_key`가 있는가?
- 입력 참조가 표준 형식으로 정의되어 있는가?
- 도구/정책/권한 검증을 통과하는가?
- 절차 실행 시 실패 처리 규칙이 있는가?
- 실행 로그에 Step ID와 output_key가 기록되는가?
