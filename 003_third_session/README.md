# 세번째 세션

이 프로젝트는 LangChain/LangGraph 입문자를 대상으로 한 **실전형 교육 코스**입니다.  
"RAG는 검색과 생성의 조합"이라는 관점을 중심으로, 검색 전략부터 그래프 기반 파이프라인까지 단계적으로 학습합니다.

## 교육 방향성

- **한 작업-한 책임** 원칙으로 단계를 분리한다.
- 검색 결과는 **정규화 → 병합 → 후처리**로 안정화한다.
- LangGraph를 사용해 **태스크 분리와 흐름 제어**를 구현한다.
- 실패 응답은 **폴백/재시도 전략**으로 복구한다.

## 학습 콘텐츠

- 교재 문서: `docs/`
- 코드 템플릿: `src/thirdsession/`
- RAG 엔드포인트: `/api/v1/rag`

## 현재 구현 상태

- 주요 기능은 **TODO/`NotImplementedError`** 상태이며, 학습자가 직접 구현하도록 설계되어 있습니다.
- 스트리밍은 **답변 → 근거(sources) → done** 순서를 반드시 유지하도록 구현해야 합니다.
- 잡 기반 API는 스캐폴딩만 제공되며, 큐/워커/상태 저장 로직은 비어 있습니다.

## 사전 준비

- Postgres 설치 및 구동
- PGVector 확장 설치 및 활성화
  - DB에서 `CREATE EXTENSION vector;` 실행

### Ubuntu 기준 설치 가이드

아래 명령은 **Ubuntu 환경 기준**입니다. 로컬 환경/버전에 맞게 조정하세요.

```bash
# 1) PostgreSQL 설치 및 서비스 시작
sudo apt update
sudo apt install -y postgresql postgresql-contrib
sudo systemctl enable --now postgresql

# 2) PostgreSQL 버전 확인
psql --version

# 3) PGVector 설치(버전에 맞는 패키지 선택)
# 예: PostgreSQL 16 → postgresql-16-pgvector
sudo apt install -y postgresql-16-pgvector

# 4) DB 접속 후 확장 활성화
sudo -u postgres psql
CREATE EXTENSION vector;
```

## 구현해야 하는 과제

아래 항목은 코드의 `TODO`/`NotImplementedError`를 기준으로 정리했습니다.  
학습자는 각 항목을 직접 구현해 전체 RAG 파이프라인을 완성합니다.

### 1) API/서비스 레이어 구현

- `src/thirdsession/main.py`
  - 앱 생성 시 의존성 주입 구성
  - 라우터 등록 방식 확정
- `src/thirdsession/api/rag/router/rag_router.py`
  - `/api/v1/rag` 요청 처리 로직 구현
  - `/api/v1/rag/stream` 스트리밍 응답 구현
  - 스트리밍 순서: **답변 → 근거(sources) → done**
- `src/thirdsession/api/rag/service/rag_service.py`
  - 요청 → 유스케이스 호출 → 응답 변환
  - 예외 처리/로깅/메트릭 수집
  - 스트리밍 순서: **답변 → 근거(sources) → done**
- `src/thirdsession/api/rag/model/request.py`
  - 입력 필드 확장 및 검증 규칙 정의
- `src/thirdsession/api/rag/model/response.py`
  - 응답 포맷(요약/상세/메타) 확장
- `src/thirdsession/api/rag/router/rag_job_router.py`
  - 잡 생성 라우터 구현
- `src/thirdsession/api/rag/router/rag_stream_router.py`
  - 잡 스트리밍 라우터 구현
- `src/thirdsession/api/rag/router/rag_status_router.py`
  - 잡 상태 조회 라우터 구현
- `src/thirdsession/api/rag/router/rag_cancel_router.py`
  - 잡 취소 라우터 구현
- `src/thirdsession/api/rag/service/rag_job_service.py`
  - 잡 생성/스트리밍/상태/취소 서비스 구현
- `src/thirdsession/api/rag/model/job_stream_response.py`
  - 스트리밍 이벤트 타입별 필드 규칙 확정

### 1-1) 잡/스트리밍 아키텍처 확장

- `nextStep.md`에 정의된 002 스타일 **잡/스트리밍 구조**를 003에 이식
- 큐/워커/체크포인터/상태 복구/이벤트 규격 구현
- `src/thirdsession/core/common/queue/chat_job_queue.py`
  - 작업 큐 적재/소비 로직 구현
- `src/thirdsession/core/common/queue/chat_stream_event_queue.py`
  - 스트리밍 이벤트 큐 적재/소비 로직 구현
- `src/thirdsession/core/common/worker/worker_base.py`
  - 워커 실행 루프/작업 조회/처리 로직 구현
- `src/thirdsession/core/common/worker/async_worker_base.py`
  - 비동기 워커 실행 루프/작업 조회/처리 로직 구현

### 2) 유스케이스/파이프라인 구성

- `src/thirdsession/core/rag/graphs/rag_pipeline_graph.py`
  - RAG 파이프라인 구성 및 실행
  - 리트리버/벡터 스토어 주입 구조 확장
  - 프롬프트/후처리/생성 단계 분리

### 3) 검색/벡터 검색 로직

- 검색 관련 로직은 `nodes` 내부에서 구현한다.
- 대상 노드: `async_search_node.py`, `merge_node.py`, `hyde_node.py` (5번 참고)

### 4) 검색 전략 그래프 구성

- `src/thirdsession/core/rag/graphs/query_decompose_graph.py`
  - 쿼리 분해 → 병렬 검색 → 병합 그래프 구성
- `src/thirdsession/core/rag/graphs/adaptive_hyde_graph.py`
  - 적응형 HyDE 그래프 구성
- `src/thirdsession/core/rag/graphs/rag_pipeline_graph.py`
  - 서비스 수준 전체 파이프라인 그래프 구성

### 5) LangGraph 노드 구현

- `src/thirdsession/core/rag/nodes/query_decompose_node.py`
  - 질문 분해 LLM 호출 및 파싱
- `src/thirdsession/core/rag/nodes/async_search_node.py`
  - 비동기 병렬 검색 및 동시성 제한
- `src/thirdsession/core/rag/nodes/hyde_node.py`
  - HyDE 생성 및 재검색 로직
- `src/thirdsession/core/rag/nodes/merge_node.py`
  - 병합/정규화/중복 제거 규칙
- `src/thirdsession/core/rag/nodes/postprocess_node.py`
  - 후처리 로직 구현
- `src/thirdsession/core/rag/nodes/stream_answer_node.py`
  - 답변 생성 및 스트리밍 규칙
- `src/thirdsession/core/rag/nodes/stream_sources_node.py`
  - 근거 스트리밍 규칙
- `src/thirdsession/core/rag/nodes/safeguard_node.py`
  - 입력 안전 분류 로직
- `src/thirdsession/core/rag/nodes/summary_node.py`
  - 대화 요약 생성 로직

### 6) 프롬프트 구성

- `src/thirdsession/core/rag/prompts/query_decompose_prompt.py`
  - 쿼리 분해 프롬프트 정의
- `src/thirdsession/core/rag/prompts/hyde_prompt.py`
  - HyDE 프롬프트 정의
- `src/thirdsession/core/rag/prompts/answer_prompt.py`
  - 답변 생성 프롬프트 정의

### 7) 후처리 구성

- 후처리는 `postprocess_node.py` 내부에서 구현한다.

### 8) 에러/안전 정책 상수 정비

- `src/thirdsession/core/rag/const/error_code.py`
  - 도메인별 에러 코드 및 매핑 규칙 확장
- `src/thirdsession/core/rag/const/safeguard_label.py`
  - 라벨별 정책(차단/완화/리다이렉트) 문서화/정의

---

## 목표 플로우

1) 쿼리 분해 및 서브쿼리 검색  
2) 결과 없을 때 HyDE 재검색  
3) 점수 정규화 및 결과 병합  
4) 후처리(중복 제거/다양성/재정렬/압축)  
5) 근거 기반 답변 생성  
6) 스트리밍 응답(답변 → 근거 → done)  

## 프로그램 동작

아래 다이어그램은 **비동기 잡/스트리밍 구조**를 기준으로 전체 동작 흐름을 요약합니다.

### 플로우차트

```mermaid
flowchart TD
    A[클라이언트 요청] -->|POST /api/v1/chat/jobs| B[잡 생성 라우터]
    B --> C[잡 서비스]
    C --> D[잡 저장/상태 초기화]
    C --> E[잡 큐 적재]
    D --> F[잡 ID 응답]
    E --> G[비동기 워커 루프]

    G --> H[잡 소비]
    H --> I[그래프 실행]
    I --> J[쿼리 분해]
    J --> K[서브쿼리 비동기 검색]
    K --> L{검색 결과 존재?}
    L -->|예| O[점수 정규화/병합]
    L -->|아니오| M[HyDE 생성/재검색]
    M --> O
    O --> P[후처리 파이프라인]
    P --> Q[답변 생성]
    Q --> R[스트리밍 이벤트 큐 적재]

    S[클라이언트 스트리밍 연결] -->|GET /api/v1/chat/stream/:job_id| T[스트리밍 라우터]
    T --> U[스트리밍 서비스]
    U --> V[이벤트 큐 소비]
    V --> W[답변 이벤트]
    V --> X[근거 이벤트]
    V --> Y[done 이벤트]
    W --> Z[클라이언트 수신]
    X --> Z
    Y --> Z
```

### 시퀀스 다이어그램

```mermaid
sequenceDiagram
    autonumber
    participant C as 클라이언트
    participant R as 잡 라우터
    participant S as 잡 서비스
    participant Q as 잡 큐
    participant W as 비동기 워커
    participant G as RAG 그래프
    participant E as 이벤트 큐
    participant SR as 스트리밍 라우터

    C->>R: POST /api/v1/chat/jobs (요청)
    R->>S: 잡 생성
    S->>S: 잡 상태 초기화/저장
    S->>Q: 잡 적재
    S-->>C: job_id 반환

    W->>Q: 잡 소비
    W->>G: 그래프 실행(비동기 노드 포함)
    G->>G: 쿼리 분해
    G->>G: 서브쿼리 비동기 검색
    alt 검색 결과 존재
        G->>G: 정규화/병합/후처리
    else 검색 결과 없음
        G->>G: HyDE 생성/재검색
        G->>G: 정규화/병합/후처리
    end
    G->>G: 답변 생성
    G->>E: 답변 이벤트 적재
    G->>E: 근거 이벤트 적재
    G->>E: done 이벤트 적재

    C->>SR: GET /api/v1/chat/stream/{job_id}
    SR->>E: 이벤트 소비
    E-->>C: 답변 → 근거 → done (스트리밍)
```

## 실행 방법

```bash
uv run uvicorn thirdsession.main:app --host 0.0.0.0 --port 8000 --reload
```

또는

```bash
uv run python -m thirdsession.main
```

## 기본 엔드포인트

- 헬스 체크: `GET /health`
- RAG 요청: `POST /api/v1/rag`
- RAG 스트리밍: `POST /api/v1/rag/stream`
- 잡 생성: `POST /api/v1/chat/jobs`
- 스트리밍: `GET /api/v1/chat/stream/{job_id}`
- 상태 조회: `GET /api/v1/chat/status/{job_id}`
- 취소: `POST /api/v1/chat/cancel/{job_id}`

> 주의: 현재 엔드포인트는 학습용 스캐폴딩이며, 실제 동작 로직은 구현되어 있지 않습니다.

## 잡 기반 엔드포인트(스캐폴딩 제공)

`nextStep.md`의 잡 기반 비동기 구조를 반영하는 목표 경로이며, 현재는 스캐폴딩만 제공됩니다.

- 잡 생성: `POST /api/v1/chat/jobs`
- 스트리밍: `GET /api/v1/chat/stream/{job_id}`
- 상태 조회: `GET /api/v1/chat/status/{job_id}`
- 취소: `POST /api/v1/chat/cancel/{job_id}`

> 주의: 위 경로는 스캐폴딩만 제공되며, 큐/워커/상태 저장 로직은 직접 구현해야 합니다.

## 주요 위치

- 애플리케이션 진입점: `src/thirdsession/main.py`
- API 영역: `src/thirdsession/api`
- RAG API(라우터 포함): `src/thirdsession/api/rag`
- 잡 서비스/모델: `src/thirdsession/api/rag`
- Core 영역: `src/thirdsession/core`
- RAG Core: `src/thirdsession/core/rag`
- 공통 큐/워커: `src/thirdsession/core/common/queue`, `src/thirdsession/core/common/worker`

## 환경 설정

프로젝트 루트에서 아래 순서로 실행하세요.

```txt
uv venv .venv
uv sync
```

### 환경 변수(.env)

기본값이 없으므로 **모든 필수 항목을 반드시 설정**해야 합니다.  
`003_third_session/.env.sample`을 복사해 `.env`를 만들고 값을 채워주세요.

필수

- `GOOGLE_API_KEY`: Google API 키(Gemini)
- `LLM_MODEL`: LLM 모델명 (예: `gemini-1.5-flash`)
- `LLM_TEMPERATURE`: LLM 온도 (예: `0.0`)

선택

- `OPENAI_API_KEY`: OpenAI API 키(향후 확장 대비)
- `REDIS_URL`: Redis 연결 문자열 (예: `redis://localhost:6379/0`)
