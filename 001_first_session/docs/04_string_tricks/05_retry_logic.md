# 05. 재시도 로직 설계 (불량 응답 복구 전략)

## 이 문서의 목표

- LLM이 **형식을 지키지 않았을 때** 안전하게 복구하는 방법을 이해한다.
- "문자열 트릭" 대신 **재시도 전략**으로 품질을 회복하는 흐름을 설계한다.
- 기존 응답을 입력으로 주고 **기대 형식으로 다시 출력**하게 만드는 방법을 배운다.
- YES/NO, 단일 선택, 복수 선택, JSON 등 **유형별 재시도 전략**을 정리한다.
- json-repair를 포함한 **복구 파이프라인**을 설계한다.

이 문서는 `docs/04_string_tricks/01~04`의 실패 대응을 **통합**해서 다룹니다.

---

## 1) 왜 재시도 전략이 필요한가?

LLM은 종종 다음과 같은 실패를 냅니다.

- 출력 형식이 깨짐(예: JSON 대신 설명문 출력)
- 금지된 문장을 포함함(예: 사과/서론/결론)
- 라벨이 아닌 문장을 반환함(예: "정답은 A입니다")

이때 단순 문자열 파싱으로 대응하면 **오류를 숨기거나 왜곡**할 위험이 큽니다.
따라서 실패를 감지한 뒤 **LLM에게 스스로 수정하도록** 재요청하는 방식이 더 안전합니다.

---

## 2) 재시도 설계 원칙

재시도 프롬프트는 기존 프롬프트와 목적이 다릅니다.
**목표는 "새로운 답"이 아니라 "형식 복구"**입니다.

### 핵심 원칙

- 기존 응답을 그대로 전달하고, **형식만 맞추도록** 요청한다.
- 새로운 추측/추가 정보를 금지한다.
- 출력 형식을 **다시 명시**하고, 규칙 위반 시 재시도 횟수를 제한한다.
- 검증 실패 이유를 함께 전달하면 수정 성공률이 높아진다.

즉, 실패 응답을 **초안(draft)**으로 보고 의미를 유지한 채
기대 형식으로 **재작성(elaborate)**하도록 지시하는 것이 핵심입니다.

---

## 3) 재시도 프롬프트 템플릿 (모듈 싱글턴)

아래는 "불량 응답을 기대 형식으로 변환"하는 재시도 프롬프트 예시입니다.

```python
"""
목적: 이전 응답을 기대 형식으로 재작성하도록 지시한다.
설명: 새로운 정보 추가를 금지하고 형식만 복구한다.
디자인 패턴: 모듈 싱글턴
"""

from langchain_core.prompts import PromptTemplate

_prompt = """너는 출력 형식 복구기다.

[규칙]
- 아래의 기존 응답을 참고하되, 새로운 정보는 추가하지 않는다.
- 출력 형식만 정확히 맞춘다.
- 설명, 머리말/맺음말, 코드 펜스는 금지한다.

[검증 실패 이유]
{error_reason}

[기대 출력 형식]
{expected_format}

[기존 응답]
{failed_output}

[출력]
기대 형식에 맞는 결과만 출력하라.
"""
prompt = PromptTemplate(
    template=_prompt,
    input_variables=["error_reason", "expected_format", "failed_output"],
)
```

---

## 4) 재시도 흐름 예시

### 흐름 개요

1. 원래 프롬프트로 응답 생성
2. 파싱/검증 실패 시 재시도 프롬프트 호출
3. 재시도 결과를 다시 파싱/검증
4. 실패 시 보수적 기본값 또는 휴먼 리뷰로 전환

### 예시 코드 (개념)

```python
"""
목적: 파싱 실패 시 재시도 프롬프트를 적용한다.
설명: 실패 응답을 기대 형식으로 재작성하도록 유도한다.
디자인 패턴: Strategy
참조: docs/04_string_tricks/05_retry_logic.md
"""

from dataclasses import dataclass
from src.examples.retry_prompt_singleton import prompt as retry_prompt_template


@dataclass(frozen=True)
class RetryFlow:
    """재시도 기반 응답 복구 흐름."""

    expected_format: str

    def run(self, first_output: str) -> str:
        """불량 응답을 재시도로 복구한다."""
        if self._is_valid(first_output):
            return first_output

        retry_prompt = retry_prompt_template.format(
            expected_format=self.expected_format,
            failed_output=first_output,
            error_reason="출력 형식 위반",
        )

        # 실제 사용 시에는 retry_prompt를 LLM에 전달한다.
        # 여기서는 흐름만 설명한다.
        retry_output = "<LLM 재응답>"

        if self._is_valid(retry_output):
            return retry_output
        return "<보수적 기본값>"

    def _is_valid(self, output: str) -> bool:
        """출력 형식 검증 (프로젝트 규칙에 맞게 구현)."""
        return bool(output.strip())
```

---

## 5) 유형별 재시도 전략

재시도는 출력 유형에 따라 프롬프트를 조금씩 다르게 구성해야 합니다.
아래는 실무에서 자주 쓰는 유형별 전략입니다.

### 5-1. YES/NO 재시도

**목표:** 기존 응답을 참고해 YES/NO만 반환.
**전략:** 실패 응답과 질문을 함께 제공하고, "YES 또는 NO만"을 강하게 고정합니다.

```text
너는 이진 응답 복구기다.
규칙: YES 또는 NO 한 단어만 출력. 설명 금지.
질문: {question}
기존 응답: {failed_output}
출력: YES 또는 NO
```

### 5-2. 단일 선택(PASS/PII/HARMFUL/PROMPT_INJECTION) 재시도

**목표:** 기존 응답을 표준 라벨로 정규화.
**전략:** 라벨 정의와 우선순위를 다시 제공하고, 기존 응답을 참고해 한 라벨만 출력.

```text
너는 라벨 복구기다.
라벨: PASS, PII, HARMFUL, PROMPT_INJECTION
우선순위: PROMPT_INJECTION > HARMFUL > PII > PASS
규칙: 위 라벨 중 하나만 출력. 설명 금지.
입력: {user_input}
기존 응답: {failed_output}
출력: PASS|PII|HARMFUL|PROMPT_INJECTION
```

### 5-3. 복수 선택(A,B,...) 재시도

**목표:** 여러 라벨을 쉼표 구분 형태로 복구.
**전략:** 정렬 규칙과 NONE 처리 규칙을 함께 제공한다.

```text
너는 복수 선택 복구기다.
선택지: A=가격, B=품질, C=배송, D=서비스
규칙: 쉼표로만 구분, 정렬은 A,B,C,D, 없으면 NONE.
입력: {text}
기존 응답: {failed_output}
출력: A,B 또는 B,D 또는 NONE
```

### 5-4. JSON 재시도

**목표:** 깨진 JSON을 스키마에 맞춰 재작성.
**전략:** 스키마와 타입 규칙을 다시 제공하고, 기존 응답을 참고해 JSON만 출력하게 한다.

```text
너는 JSON 복구기다.
규칙: JSON 외 텍스트 금지, 스키마와 타입 엄수.
스키마: {schema}
입력: {text}
기존 응답: {failed_output}
출력: JSON만 반환
```

### 5-5. Pydantic 검증 실패 재시도

**목표:** 파싱은 성공했지만 **스키마 검증에 실패한 JSON**을 исправ하게 한다.\n
**전략:** Pydantic 오류 이유를 함께 제공하고, 필수 필드/타입 규칙을 다시 강조한다.

```text
너는 JSON 교정기다.
규칙: JSON 외 텍스트 금지, 스키마와 타입 엄수.
Pydantic 오류: {validation_error}
스키마: {schema}
입력: {text}
기존 응답: {failed_output}
출력: JSON만 반환
```

이렇게 유형별로 재시도 프롬프트를 조정하면 복구 성공률이 크게 올라갑니다.

---

## 6) Pydantic + json-repair 포함 복구 파이프라인

JSON은 \"살짝 깨진\" 상태로 많이 오기 때문에, LLM 재시도 전에 **Pydantic 검증**과 `json-repair`를 함께 두는 것이 효율적입니다.

### 권장 순서

1. `json.loads`로 엄격 파싱
2. 파싱 성공 시 **Pydantic 스키마 검증**
3. 검증 실패 또는 파싱 실패 시 `json-repair`로 복구 시도
4. 그래도 실패하면 **LLM 재시도 프롬프트** 호출
5. 최종 실패 시 보수적 기본값 또는 휴먼 리뷰 전환

### json-repair 예시 코드

```python
"""
목적: json-repair로 JSON 복구 후 파싱한다.
설명: 1차 파싱 실패 시 복구 절차를 적용한다.
디자인 패턴: Strategy
"""

import json
from dataclasses import dataclass
from json_repair import repair_json


@dataclass(frozen=True)
class RepairJsonParser:
    """json-repair 기반 JSON 파서."""

    required_keys: set[str]

    def parse(self, raw_text: str) -> dict:
        """원본 텍스트를 JSON으로 변환한다."""
        cleaned = raw_text.replace("```json", "").replace("```", "").strip()
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            repaired = repair_json(cleaned)
            data = json.loads(repaired)
        self._validate_keys(data)
        return data

    def _validate_keys(self, data: dict) -> None:
        """필수 키가 존재하는지 검증한다."""
        missing = self.required_keys - data.keys()
        if missing:
            raise ValueError(f"필수 키 누락: {sorted(missing)}")
```

설치 예시:

```bash
uv add json-repair
```

**주의:** `json-repair`는 보조 도구입니다. 복구된 JSON이라도 의미가 왜곡될 수 있으므로
필수 키/타입 검증을 반드시 유지하세요.

---

## 7) 재시도 실패 시 대처 방법

재시도에도 실패하는 경우는 반드시 발생합니다. 다음 순서를 권장합니다.

1. **재시도 횟수 제한**: 보통 1~2회로 제한
2. **보수적 기본값**: UNKNOWN 또는 안전한 기본 경로
3. **휴먼 리뷰 전환**: 중요한 작업은 수동 확인
4. **실패 패턴 기록**: 프롬프트 개선에 활용

재시도 전략은 "만능"이 아니라 **품질을 회복하기 위한 안전장치**입니다.

---

## 8) 구현 체크리스트

- 실패 조건(파싱/검증)이 코드에 명확히 정의되어 있는가?
- 재시도 프롬프트가 "형식 복구"만 요청하도록 구성되었는가?
- 재시도 횟수와 종료 조건이 문서화되어 있는가?
- 실패 시 기본값/휴먼 리뷰 경로가 준비되어 있는가?
- JSON의 경우 json-repair 적용 순서가 문서화되어 있는가?
- JSON의 경우 Pydantic 검증 위치가 문서화되어 있는가?
