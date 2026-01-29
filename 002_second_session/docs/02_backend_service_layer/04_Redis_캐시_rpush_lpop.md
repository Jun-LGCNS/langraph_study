# 04. Redis 캐시 스트리밍(rpush/lpop)

## 이 챕터에서 배우는 것

- Redis 리스트(rpush/lpop) 기반 스트리밍 구조
- 생산자/소비자 분리와 메시지 순서 보장
- 폴링, 재시도, 종료 신호 처리
- 실전 운영에서의 안정성 고려사항
- FastAPI 스트리밍 엔드포인트 예시
- 워커/큐 분리 운영 전략

---

## 1. 왜 Redis 리스트인가?

Redis 리스트는 **가벼운 메시지 큐**로 활용하기 좋습니다.
특히 스트리밍 데이터처럼 **순서가 중요한 이벤트**에 적합합니다.

- `RPUSH`: 생산자가 이벤트를 뒤에 추가
- `LPOP`: 소비자가 앞에서 순서대로 소비

---

## 2. 기본 구조(Producer-Consumer)

- **Producer(그래프)**: 이벤트를 `rpush`로 적재
- **Consumer(API)**: `lpop`으로 이벤트를 읽고 SSE로 전달

이 구조는 **요청 처리와 전달 채널을 분리**하는 핵심입니다.

---

## 3. 워커/큐 분리 전략

비동기 실행에서는 **Producer가 워커**가 됩니다.

- **Producer(워커)**: LangGraph 실행 → 이벤트를 `rpush`
- **Consumer(API)**: 이벤트를 `lpop` → SSE 전송

이 구조를 사용하면 **API 서버가 과부하로 중단되더라도**
워커는 독립적으로 실행을 유지할 수 있습니다.

---

## 4. 메시지 구조 설계

```python
"""
목적: Redis에 저장할 스트리밍 메시지 구조를 예시로 보여준다.
설명: 메시지는 딕셔너리 형태로 저장한다.
디자인 패턴: Value Object
"""

redis_message = {
    "type": "token",
    "payload": {"content": "안녕하세요"},
    "trace_id": "t-123",
    "seq": 2,
}
```

---

## 5. 생산자/소비자 예시

```python
"""
목적: 생산자/소비자 흐름을 분리한다.
설명: 워커(그래프 실행)는 rpush, API는 lpop을 사용한다.
디자인 패턴: Producer-Consumer
"""

import json


def push_event(redis_client, key: str, event: dict) -> None:
    """Redis 리스트에 이벤트를 적재한다."""
    redis_client.rpush(key, json.dumps(event, ensure_ascii=False))


def pop_event(redis_client, key: str) -> dict | None:
    """Redis 리스트에서 이벤트를 소비한다."""
    raw = redis_client.lpop(key)
    if not raw:
        return None
    data = json.loads(raw)
    return data
```

---

## 6. FastAPI 스트리밍 엔드포인트 예시(동기)

```python
"""
목적: Redis 리스트를 SSE로 전달한다.
설명: lpop으로 이벤트를 소비하고 SSE 라인으로 변환한다.
디자인 패턴: Producer-Consumer
"""

import time
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()


def stream_from_redis(redis_client, key: str):
    """Redis 리스트를 폴링하며 SSE를 전송한다."""
    while True:
        event = pop_event(redis_client, key)
        if event is None:
            time.sleep(0.1)
            continue
        yield to_sse_line(event)
        if event.type == "done":
            break


@app.get("/stream/{job_id}")
def stream_job(job_id: str):
    return StreamingResponse(
        stream_from_redis(redis_client, job_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )
```

---

## 7. 폴링과 종료 신호 처리

`lpop`은 블로킹이 아니므로 **폴링/백오프**가 필요합니다.
또한 `done` 이벤트로 스트리밍 종료를 명확히 해야 합니다.

```python
"""
목적: lpop 기반 폴링 루프를 구성한다.
설명: 이벤트가 없으면 짧게 대기한다.
디자인 패턴: Polling
"""

import time


def stream_loop(redis_client, key: str) -> None:
    """Redis 리스트를 폴링하며 스트리밍한다."""
    while True:
        event = pop_event(redis_client, key)
        if event is None:
            time.sleep(0.1)
            continue
        send_sse(event)
        if event.type == "done":
            break
```

---

## 8. 운영 안정성 포인트

- **TTL 설정**: 작업 완료 후 키를 만료 처리
- **seq 관리**: 순서 보장 및 재조립 가능
- **오버플로 대응**: 리스트 길이 제한
- **재시도 정책**: 일정 시간 동안 이벤트가 없으면 종료

`lpop` 기반 구조는 **단순하지만 내구성이 약할 수 있습니다**.
필요 시 **Redis Streams**나 **RPOPLPUSH**로 확장하는 전략을 고려합니다.

---

## 9. 구현 체크리스트

- `rpush/lpop` 흐름이 명확한가?
- `done` 이벤트로 종료를 보장하는가?
- 폴링/백오프 정책이 있는가?
- TTL/정리 정책이 있는가?
- 순서 보장(`seq`)이 가능한가?

---

## 10. 코드 연결(프로젝트 기준)

- 작업 큐: `src/secondsession/core/common/queue/chat_job_queue.py`
- 이벤트 큐: `src/secondsession/core/common/queue/chat_stream_event_queue.py`
- 워커 실행: `src/secondsession/core/worker/chat_worker.py`
- 스트리밍 소비자: `src/secondsession/api/chat/service/chat_service.py`

---

## 11. 실습 최소 순서(권장)

1) `ChatJobQueue.enqueue/dequeue`에서 JSON 직렬화 규칙을 확정한다.  
2) `ChatStreamEventQueue.push_event/pop_event`에 job_id 키 규칙을 적용한다.  
3) 이벤트가 없을 때 **짧은 폴링**을 적용한다.  
4) `done` 이벤트 수신 시 스트리밍을 종료한다.
