# 02. LangGraph 체크포인터 원리

## 이 챕터에서 배우는 것

- LangGraph에서 체크포인터가 연결되는 위치
- 저장/복구의 기본 흐름과 개념
- 체크포인트 데이터 구성 요소
- 실전 운영에서의 설계 포인트
- LangGraph API 기반 저장/복구 예시
- 저장 시점 메타데이터 설계 예시
- 특정 checkpoint_id로 복구하는 방법
- 동기 vs 비동기 체크포인터 운영 관점

---

## 1. LangGraph에서 체크포인터의 역할

LangGraph는 **그래프 실행 중 상태를 갱신**합니다.
체크포인터는 이 상태를 **지속적으로 저장**하고, 필요하면 복구할 수 있게 합니다.

일반적으로 다음 구조로 연결됩니다.

- 그래프 실행 중 상태 변경 발생
- 체크포인터가 상태 스냅샷 저장
- 재실행 시 마지막 스냅샷 로드

---

## 2. 체크포인트 데이터 구성(개념)

체크포인트는 보통 다음 정보로 구성됩니다.

- **state**: 그래프 상태 스냅샷
- **metadata**: 저장 시점/노드/라우팅 정보
- **checkpoint_id**: 버전 식별자
- **thread_id**: 세션 식별자

```python
"""
목적: 체크포인트 데이터 구조를 예시로 보여준다.
설명: 필드 구성을 이해하기 위한 딕셔너리 예시다.
디자인 패턴: Memento
"""

checkpoint = {
    "thread_id": "thread-123",
    "checkpoint_id": "ckpt-0003",
    "state": {"query": "환불 정책", "route": "generate"},
    "metadata": {"node": "generate", "timestamp": "2026-01-27T10:00:00Z"},
}
```

---

## 3. 저장 시점(실전 기준)

체크포인트 저장 시점은 **운영 정책**에 따라 달라집니다.
보통은 다음 지점을 기준으로 저장합니다.

- 노드 실행 후 상태가 변경된 시점
- 라우팅 결정 직후
- 폴백 경로 진입 시점

저장 빈도가 높을수록 복구는 쉬워지지만 **저장 비용이 증가**합니다.

---

## 4. LangGraph API 예시(저장/복구)

```python
"""
목적: LangGraph 체크포인터를 실제로 연결한다.
설명: thread_id를 기준으로 저장/복구가 자동으로 동작한다.
디자인 패턴: Memento
"""

from typing import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver


class SimpleState(TypedDict):
    query: str
    answer: str


def generate_node(state: SimpleState) -> dict:
    return {"answer": f"응답: {state['query']}"}


graph = StateGraph(SimpleState)
graph.add_node("generate", generate_node)
graph.set_entry_point("generate")
graph.add_edge("generate", END)

checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

# 저장: invoke 실행 시점에 체크포인트가 자동 저장됨
config = {"configurable": {"thread_id": "thread-123"}}
app.invoke({"query": "환불 정책 알려줘"}, config=config)

# 복구: 같은 thread_id로 다시 invoke하면 마지막 체크포인트에서 이어짐
app.invoke({"query": "추가 질문"}, config=config)
```

---

## 5. 저장 시점 메타데이터 예시(노드/라우팅/폴백)

체크포인트에 **노드/라우팅/폴백 정보를 담고 싶다면**
상태에 메타데이터를 직접 포함시키는 방식이 가장 단순합니다.

```python
"""
목적: 체크포인트에 메타데이터를 포함시키는 흐름을 보여준다.
설명: 노드 실행 시점에 메타데이터를 상태에 추가한다.
디자인 패턴: Memento
"""

from typing import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver


class MetaState(TypedDict):
    query: str
    route: str
    error_code: str | None
    checkpoint_meta: dict


def route_node(state: MetaState) -> dict:
    route = "fallback" if "환불" in state["query"] else "generate"
    return {
        "route": route,
        "checkpoint_meta": {"node": "route", "route": route},
    }


def generate_node(state: MetaState) -> dict:
    return {
        "checkpoint_meta": {"node": "generate", "route": state["route"]},
    }


def fallback_node(state: MetaState) -> dict:
    return {
        "error_code": "low_quality",
        "checkpoint_meta": {"node": "fallback", "error_code": "low_quality"},
    }


graph = StateGraph(MetaState)
graph.add_node("route", route_node)
graph.add_node("generate", generate_node)
graph.add_node("fallback", fallback_node)
graph.set_entry_point("route")

graph.add_conditional_edges("route", lambda s: s["route"], {
    "generate": "generate",
    "fallback": "fallback",
})
graph.add_edge("generate", END)
graph.add_edge("fallback", END)

checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)
app.invoke({"query": "환불 정책 알려줘"}, config={"configurable": {"thread_id": "thread-123"}})
```

이 방식은 **체크포인트에 필요한 메타데이터를 명시적으로 포함**할 수 있어
운영 로그/분석에 유리합니다.

---

## 6. 특정 checkpoint_id로 복구하는 시나리오

특정 시점으로 되돌아가려면 **checkpoint_id를 지정**해 복구합니다.
아래는 실전 흐름을 이해하기 위한 예시입니다.

```python
"""
목적: 특정 checkpoint_id로 그래프를 복구한다.
설명: checkpoint_id를 지정해 원하는 시점으로 되돌린다.
디자인 패턴: Memento
"""

# 1) 먼저 여러 번 실행해 체크포인트를 누적
config = {"configurable": {"thread_id": "thread-123"}}
app.invoke({"query": "1차 질문"}, config=config)
app.invoke({"query": "2차 질문"}, config=config)

# 2) 특정 checkpoint_id로 복구 (버전별 키 이름은 다를 수 있음)
rollback_config = {
    "configurable": {
        "thread_id": "thread-123",
        "checkpoint_id": "ckpt-0001",
    }
}
app.invoke({"query": "되돌린 상태에서 이어서 질문"}, config=rollback_config)
```

`checkpoint_id` 키의 정확한 이름은 **버전에 따라 다를 수 있으므로**
운영 환경에서 사용하는 LangGraph 버전 문서에 맞춰 확인해야 합니다.

---

## 7. 동기 vs 비동기 체크포인터 운영 관점

동기/비동기 체크포인터는 **정합성보다 성능/지연**에 더 큰 차이를 만듭니다.

- **동기**: 저장 시점에 I/O 대기가 발생하지만 흐름이 단순함
- **비동기**: 저장 I/O가 분리되어 처리량이 높지만 구현 복잡도가 증가함

운영에서는 **thread_id 충돌 방지**와 **저장 시점 일관성**이 핵심입니다.

---

## 8. 실전 설계 포인트

- **thread_id 일관성**: 동일 세션은 동일 thread_id 유지
- **버전 관리**: checkpoint_id로 이력 추적
- **정리 정책**: 최신 N개만 유지, TTL 적용
- **관측성 확보**: metadata에 노드/라우팅 정보 기록

---

## 9. 구현 체크리스트

- state와 metadata가 분리되어 있는가?
- 저장 시점이 명확히 정의되어 있는가?
- thread_id와 checkpoint_id가 일관되게 관리되는가?
- 복구 시나리오가 테스트 가능한가?
- 정리 정책이 문서화되어 있는가?

---

## 10. 코드 연결(프로젝트 기준)

- 체크포인터 팩토리: `src/secondsession/core/common/checkpointer/redis_checkpointer.py`
- 비동기 체크포인터 구현: `src/secondsession/core/common/checkpointer/redis_async_checkpointer.py`
- 그래프 컴파일: `src/secondsession/core/chat/graphs/chat_graph.py`
- 워커 실행(config 전달): `src/secondsession/core/worker/chat_worker.py`

---

## 11. 실습 최소 순서(권장)

1) `ChatWorker`에서 `configurable.thread_id`를 반드시 전달한다.  
2) 필요 시 `checkpoint_id`로 되돌리는 복구 옵션을 추가한다.  
3) 체크포인트 메타데이터에 `node/route/error_code`를 기록한다.  
4) 동기/비동기 체크포인터 전환을 쉽게 하도록 팩토리 함수를 사용한다.
