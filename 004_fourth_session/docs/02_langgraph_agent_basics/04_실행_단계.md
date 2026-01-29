# 04. Execution 단계 구성

## 이 챕터에서 배우는 것

- Plan 인터페이스를 실행하기 위한 **Tool Registry 설계**
- Registry가 **어떻게 제어/검증되는지**
- Node마다 다른 Tool을 등록하는 **멀티 레지스트리 구성**
- 병렬 Tool 호출과 **부분 실패 처리**
- Router로 Registry를 선택하는 **LangGraph 연결 방식**
- 각 전략을 **언제 써야 하는지**에 대한 기준

---

## 1. 실행 단계는 무엇을 보장해야 하는가

실행 단계는 Plan을 **있는 그대로 수행**하는 단계입니다.
이 단계의 목표는 “실행 일관성”이며, 계획을 임의로 바꾸지 않는 것이 핵심입니다.
실행 단계가 흔들리면 계획의 의미가 사라집니다.
따라서 실행은 **규칙에 따라 자동화**되는 것이 이상적입니다.
이 원칙이 유지되면 장애 원인 추적도 쉬워집니다.

- Plan 인터페이스를 **해석하고 실행**한다.
- Tool Registry를 통해 **허용된 Tool만 호출**한다.
- 실패는 정책(retry/timeout)에 따라 처리한다.

---

## 2. Tool Registry 설계 (핵심 구성 요소)

Tool Registry는 “실행 가능한 Tool 목록과 규칙”을 관리하는 구성 요소입니다.
Registry가 없으면 Tool 호출이 분산되고, 권한/정책을 일관되게 적용하기 어렵습니다.
즉, Registry는 **통제 지점**입니다.
도구가 많아질수록 Registry의 가치가 커집니다.
실무에서는 Registry가 도구 운영의 기준이 됩니다.

필수 구성:

- **등록 목록**: tool_name → callable
- **스키마 메타**: 입력/출력 규칙
- **정책 메타**: 권한/비용/타임아웃

### 2-1) Tool Registry 예시

```python
"""
목적: Tool Registry의 최소 형태를 구현한다.
설명: 등록/조회/정책 확인 기능을 포함한다.
디자인 패턴: Registry
"""

from dataclasses import dataclass
from typing import Callable


@dataclass
class ToolMeta:
    name: str
    handler: Callable
    timeout_ms: int = 3000
    role_required: str | None = None
    input_schema: type | None = None


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolMeta] = {}

    def register(self, meta: ToolMeta) -> None:
        self._tools[meta.name] = meta

    def get(self, name: str) -> ToolMeta:
        if name not in self._tools:
            raise KeyError(f"Tool not found: {name}")
        return self._tools[name]

    def allowed(self, name: str, role: str) -> bool:
        meta = self.get(name)
        if meta.role_required is None:
            return True
        return meta.role_required == role

    def render_catalog(self) -> str:
        """LLM 프롬프트에 넣기 위한 Tool 카탈로그 문자열."""
        lines = []
        for meta in self._tools.values():
            line = f"- {meta.name}: timeout={meta.timeout_ms}ms"
            if meta.role_required:
                line += f", role={meta.role_required}"
            lines.append(line)
        return "\n".join(lines)
```

---

## 3. Tool Registry 제어(검증 흐름)

Plan 인터페이스와 Registry를 연결할 때는 **검증 단계**가 필요합니다.
검증은 실행 이전에 실패를 걸러내는 가장 저렴한 안전장치입니다.
검증이 없으면 오류가 실행 단계에서 폭발합니다.
검증 단계는 비용을 줄이고 장애를 예방합니다.
따라서 실행 전에 반드시 거쳐야 합니다.

```text
Plan Step -> Tool Registry 조회 -> 스키마/권한 검증 -> 실행 허용/차단
```

### 언제 필요한가

- Tool이 많아질수록 **오용 위험**이 커진다.
- 권한/비용이 있는 Tool은 **반드시 검증**이 필요하다.
- 입력 스키마가 깨지면 실행 단계에서 오류가 폭발한다.

---

## 4. 스키마 검증 포함 실행

입력 스키마를 강제하면 실행 실패를 줄일 수 있습니다.
특히 외부 시스템 호출에서는 스키마 오류가 곧 장애로 이어지므로 필수입니다.
스키마 검증은 입력 품질을 보장합니다.
검증이 없으면 잘못된 값이 그대로 전달됩니다.
따라서 스키마 검증은 안전장치의 핵심입니다.

```python
"""
목적: 입력 스키마를 강제해 Tool 호출을 안정화한다.
설명: schema가 있으면 입력을 검증한다.
디자인 패턴: Guard
"""

def execute_with_schema(state: dict, registry: ToolRegistry) -> dict:
    step = state["plan"]["steps"][0]
    meta = registry.get(step["tool"])
    payload = step.get("input", {})

    if meta.input_schema:
        payload = meta.input_schema(**payload).model_dump()

    result = meta.handler(payload)
    return {**state, "tool_results": [result]}
```

### 언제 필요한가

- 외부 API/DB를 호출하는 Tool
- 비용이 큰 Tool
- 잘못된 입력이 서비스 장애로 이어지는 Tool

---

## 5. Node마다 다른 Tool 등록 (멀티 레지스트리)

실무에서는 Tool을 목적별로 분리합니다.  
예: 파일 Tool, 비즈니스 Tool, 모니터링 Tool.
이 분리는 “LLM 혼동 감소”와 “권한 경계 분리”를 동시에 해결합니다.
또한 도메인별 정책을 분리하는 데도 도움이 됩니다.
결과적으로 운영 부담이 줄어듭니다.

```python
"""
목적: Node별로 다른 Tool Registry를 사용한다.
설명: 카테고리별 Tool 분리를 구현한다.
디자인 패턴: Registry + Router
"""

file_registry = ToolRegistry()
file_registry.register(ToolMeta(name="read_file", handler=read_file))
file_registry.register(ToolMeta(name="write_file", handler=write_file, role_required="admin"))

biz_registry = ToolRegistry()
biz_registry.register(ToolMeta(name="crm_get_customer", handler=crm_get_customer))

monitor_registry = ToolRegistry()
monitor_registry.register(ToolMeta(name="get_metrics", handler=get_metrics))
```

### 언제 필요한가

- Tool 수가 많아져 LLM 혼동이 증가할 때
- 권한 레벨이 다른 Tool을 분리해야 할 때
- 운영 정책(타임아웃/재시도)을 도메인별로 다르게 적용할 때

---

### 5-1) Registry를 사용하는 실행 노드 예시

```python
"""
목적: Registry를 통해 Tool 실행을 제어한다.
설명: 권한 검증 후 Tool을 호출하고 결과를 저장한다.
디자인 패턴: Guard
"""

def node_execute_with_registry(state: dict, registry: ToolRegistry) -> dict:
    step = state["plan"]["steps"][0]
    tool_name = step["tool"]

    if not registry.allowed(tool_name, state.get("role", "user")):
        return {**state, "errors": ["permission denied"], "halted": True}

    meta = registry.get(tool_name)
    result = meta.handler(step.get("input", {}))
    return {**state, "tool_results": [result]}
```

---


카테고리별 Registry를 실제 그래프에 연결하려면 Router가 필요합니다.

```python
"""
목적: 요청에 따라 Registry를 선택한다.
설명: 라우터에서 노드 분기 후 해당 Registry로 실행한다.
디자인 패턴: Router
"""

from langgraph.graph import StateGraph


def route_registry(state: dict) -> str:
    query = state.get("query", "")
    if "파일" in query:
        return "file_exec"
    if "고객" in query:
        return "biz_exec"
    return "monitor_exec"


builder = StateGraph(dict)
builder.add_node("router", lambda s: s)
builder.add_node("file_exec", lambda s: node_execute_with_registry(s, file_registry))
builder.add_node("biz_exec", lambda s: node_execute_with_registry(s, biz_registry))
builder.add_node("monitor_exec", lambda s: node_execute_with_registry(s, monitor_registry))

builder.add_conditional_edges(
    "router",
    route_registry,
    {
        "file_exec": "file_exec",
        "biz_exec": "biz_exec",
        "monitor_exec": "monitor_exec",
    },
)
```

### 언제 필요한가

- Tool 카테고리를 분리했을 때
- 도메인별 실행 정책이 다를 때
- 권한 기반 흐름을 분기해야 할 때

---

## 7. 병렬 Tool 호출

Plan 인터페이스가 `strategy: parallel`일 경우, Tool을 병렬로 호출합니다.
병렬은 지연을 줄이는 대신, 실패 처리와 결과 합성 규칙이 더 중요해집니다.
병렬은 SLA가 중요한 요청에서 특히 유용합니다.
하지만 실패가 섞이면 합성 정책이 없으면 혼란이 생깁니다.
따라서 병렬은 정책과 함께 설계해야 합니다.

```python
"""
목적: 병렬 Tool 호출의 개념을 보여준다.
설명: 실제 구현에서는 asyncio.gather 등을 사용한다.
디자인 패턴: 병렬 실행
"""

import asyncio


async def run_parallel_tools(steps: list[dict], registry: ToolRegistry) -> list[dict]:
    async def _run(step: dict) -> dict:
        meta = registry.get(step["tool"])
        return meta.handler(step["input"])

    tasks = [_run(step) for step in steps]
    return await asyncio.gather(*tasks)
```

### 언제 필요한가

- 서로 독립적인 조회/수집 작업일 때
- SLA가 중요하고 지연을 줄여야 할 때
- 순서 보장이 필요 없을 때

---

## 8. 부분 실패 처리(Partial OK)

병렬 실행에서 일부 Tool이 실패하는 것은 흔합니다. 이 경우 **partial_ok 정책**이 필요합니다.
실무에서는 “전부 실패보다 일부 성공이 더 낫다”는 상황이 많기 때문에 이 정책이 자주 사용됩니다.
partial_ok는 실패를 숨기는 것이 아니라 **구조화해서 노출**하는 정책입니다.
이를 통해 사용자에게 부분 결과를 전달할 수 있습니다.
결과적으로 응답 품질이 개선됩니다.

```python
"""
목적: 병렬 호출에서 부분 실패를 허용한다.
설명: 실패한 결과는 error로 기록하고 성공 결과만 합성한다.
디자인 패턴: Partial OK
"""

def merge_partial_ok(results: list[dict]) -> dict:
    success = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]
    return {"success": success, "failed": failed}
```

### 언제 필요한가

- 일부 데이터만 있어도 답변이 가능한 경우
- 실패한 Tool이 전체 실패를 유발하면 안 되는 경우
- 응답 지연보다 부분 결과가 더 중요한 경우

---

## 9. 결과 합성과 Plan 인터페이스 연결

Plan에서 정의한 `output_key`를 기준으로 결과를 저장합니다.
이 연결이 없으면 후속 단계가 이전 결과를 참조할 수 없어 실행이 끊깁니다.
따라서 output_key는 계획 단계에서 미리 정의되어야 합니다.
저장 규칙이 명확하면 재시도도 쉬워집니다.
이는 실행 안정성의 핵심 요소입니다.

```python
"""
목적: Plan output_key를 기준으로 결과를 저장한다.
설명: 후속 step의 입력 참조를 가능하게 한다.
디자인 패턴: Mapper
"""

def map_results_to_state(steps: list[dict], results: list[dict]) -> dict:
    state_outputs = {}
    for step, result in zip(steps, results):
        output_key = step.get("output_key")
        if output_key:
            state_outputs[output_key] = result
    return state_outputs
```

---

## 10. plan.strategy 기반 자동 실행 선택

Plan에 `strategy`가 명시되어 있으면, 실행 경로를 자동으로 선택할 수 있습니다.
이 구조는 실행 엔진이 “전략 선택”을 명시적으로 수행하도록 돕습니다.
전략 선택이 명확하면 실행 로직이 단순해집니다.
또한 운영자는 어떤 전략이 쓰였는지 추적할 수 있습니다.
결과적으로 실행 품질이 안정됩니다.

```python
"""
목적: plan.strategy에 따라 병렬/절차 실행을 자동 선택한다.
설명: strategy 값이 없으면 기본값을 사용한다.
디자인 패턴: Strategy
"""

async def execute_by_strategy(plan: dict, registry: ToolRegistry) -> list[dict]:
    strategy = plan.get("strategy", "sequential")
    steps = plan["steps"]
    if strategy == "parallel":
        return await run_parallel_tools(steps, registry)
    results: list[dict] = []
    for step in steps:
        out = await run_parallel_tools([step], registry)
        results.extend(out)
    return results
```

이 방식은 **우선순위 Tool을 의미하지 않습니다.**  
`strategy`는 “동시에 실행할지, 순서대로 실행할지”를 정하는 기준입니다.

우선순위가 필요하다면 다음 중 하나를 추가해야 합니다.

- `priority` 필드: 낮은 숫자가 먼저 실행
- `depends_on` 필드: 의존성으로 순서를 강제

---

## 11. priority/depends_on 기반 실행 순서 예시

이 섹션은 “실행 순서를 자동으로 정하는 규칙”을 최소 형태로 보여줍니다.  
고급 환경에서는 이 로직을 DAG 위상 정렬로 확장하는 것이 일반적입니다.
핵심은 의존성이 있는 작업을 먼저 구분하는 것입니다.
이 과정이 없으면 순서 오류가 발생합니다.
따라서 우선순위 규칙은 실무에서 필수입니다.

```python
"""
목적: priority와 depends_on을 기준으로 실행 순서를 결정한다.
설명: 의존성이 있으면 우선 적용하고, 나머지는 priority로 정렬한다.
디자인 패턴: Scheduler
"""

def order_steps(steps: list[dict]) -> list[dict]:
    # 1) depends_on이 있는 step은 뒤로 보낸다
    no_dep = [s for s in steps if not s.get("depends_on")]
    has_dep = [s for s in steps if s.get("depends_on")]

    # 2) priority로 정렬(없으면 기본 100)
    def _prio(step: dict) -> int:
        return step.get("priority", 100)

    no_dep_sorted = sorted(no_dep, key=_prio)
    has_dep_sorted = sorted(has_dep, key=_prio)

    return no_dep_sorted + has_dep_sorted
```

이 예시는 **간단한 규칙**이며, 실제로는 DAG 위상 정렬로 확장합니다.

---

## 12. depends_on 기반 병렬 그룹 분리 예시

depends_on을 기준으로 병렬 그룹을 나누면 “동시 실행 가능한 묶음”을 만들 수 있습니다.  
이 방식은 병렬성과 안정성을 동시에 확보하는 대표적인 패턴입니다.
먼저 실행 가능한 그룹을 분리하면 리소스를 효율적으로 쓸 수 있습니다.
또한 실패가 발생해도 다음 그룹에 영향을 최소화할 수 있습니다.
따라서 그룹 분리는 병렬 실행의 핵심입니다.

```python
"""
목적: depends_on을 기준으로 병렬 실행 가능한 그룹을 나눈다.
설명: 의존성이 없는 단계부터 순차적으로 그룹을 만든다.
디자인 패턴: DAG 레이어링
"""

def build_parallel_groups(steps: list[dict]) -> list[list[dict]]:
    remaining = {s["id"]: s for s in steps}
    done: set[int] = set()
    groups: list[list[dict]] = []

    while remaining:
        ready = []
        for step_id, step in list(remaining.items()):
            deps = set(step.get("depends_on", []))
            if deps.issubset(done):
                ready.append(step)
                del remaining[step_id]

        if not ready:
            raise ValueError("순환 의존성 감지")

        groups.append(ready)
        done.update(s["id"] for s in ready)

    return groups
```

이 결과로 나온 각 그룹은 **동시에 실행 가능한 단계 묶음**입니다.

### 12-1) 그룹을 asyncio.gather로 실행하는 예시

```python
"""
목적: 병렬 그룹을 asyncio.gather로 실행한다.
설명: 그룹 내는 병렬, 그룹 간은 순차 실행한다.
디자인 패턴: 병렬 + 순차 혼합 실행
"""

import asyncio


async def run_step(step: dict, registry: ToolRegistry) -> dict:
    meta = registry.get(step["tool"])
    return meta.handler(step.get("input", {}))


async def run_groups_in_order(groups: list[list[dict]], registry: ToolRegistry) -> list[dict]:
    all_results: list[dict] = []
    for group in groups:
        tasks = [run_step(step, registry) for step in group]
        results = await asyncio.gather(*tasks)
        all_results.extend(results)
    return all_results
```

---

## 13. 실무 체크리스트

- Tool Registry에 등록되지 않은 Tool이 실행되지 않는가?
- Node마다 다른 Registry를 사용하고 있는가?
- 스키마 검증을 통해 입력 오류를 차단하는가?
- 병렬 실행 시 타임아웃/재시도 기준이 있는가?
- 부분 실패 정책(partial_ok)이 정의되어 있는가?
- Plan의 output_key가 결과 저장에 반영되는가?
- Router가 권한/도메인 기준을 반영하는가?
