# 03. 적응형 HyDE 패턴

## 이 챕터에서 배우는 것

- HyDE의 개념과 적용 이점
- 단점과 비용/품질 리스크
- 적응형 HyDE 설계 시 고려 사항
- LangChain/LangGraph 기반 실무 예시

---

## 1. 개념 설명과 이점

HyDE(Hypothetical Document Embeddings)는 **가상의 답변 문서**를 생성하고,
그 문서를 검색 쿼리로 사용해 검색 품질을 높이는 전략입니다.

**이점**

- **질문 재표현 효과**: 모호한 질문을 더 구체화된 문서 형태로 변환
- **검색 품질 개선**: 기본 검색이 약한 도메인에서 성능 향상
- **메타데이터 결합 용이**: HyDE 생성 시 필터 힌트를 포함 가능

---

## 2. 단점과 리스크

- **비용 증가**: HyDE 생성이 추가 LLM 호출을 유발
- **노이즈 위험**: 잘못된 가설 문서가 검색 품질을 악화
- **지연 시간 증가**: 재검색 단계가 추가됨
- **과적용 위험**: 모든 쿼리에 적용하면 효율이 떨어짐

---

## 3. 실제 구현 시 고려 사항

### 1) 적응형 조건

- **검색 결과가 적절한지 LLM으로 검증** 후 HyDE 실행
- 예: “답변 가능” 판정 실패 시 HyDE로 전환

### 2) HyDE 문서 길이

- 너무 길면 임베딩 비용 증가
- 200~500 토큰 수준 권장(도메인에 맞게 조정)

### 3) 메타데이터 필터

- HyDE 생성 프롬프트에 **필터 힌트** 포함
- 재검색 시 동일 필터 적용

### 4) 결과 병합

- 기본 검색 + HyDE 검색 결과를 병합
- 중복 제거 및 점수 정규화 필요

### 5) 문서 형태 반영

- HyDE는 **초기 검색 문서의 형식/톤**을 반영할수록 품질이 높아짐
- 예: “정책 문서형/FAQ형/가이드형” 등 문서 스타일을 힌트로 제공

### 6) 권장 흐름(요약)

1) 기본 검색 수행  
2) LLM으로 “답변 가능성” 검증  
3) 불충분하면 **기존 문서 형태 힌트**를 포함해 HyDE 생성  
4) HyDE 문서로 재검색  
5) 결과 병합 후 반환  

---

## 4. 예시

```python
"""
목적: 검색 결과를 LLM으로 검증 후 HyDE를 조건부로 수행한다.
설명: 기존 문서 형태를 참고해 HyDE 문서를 생성하고 재검색한다.
디자인 패턴: State Machine
"""

from typing import Any
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from textwarp import dedent

HYDE_PROMPT = PromptTemplate(
    dedent(
    f"""
    You are generating a hypothetical document to improve retrieval.
    Write a concise document that would directly answer the user's question.
    Use the style hint to match tone/format (e.g., policy/FAQ/guide).
    Rules:
    - Do not invent citations or sources.
    - Keep it focused; avoid tangents.
    - Preserve the user's language.
    Output plain text only.
    Question: {{question}}
    Style hint: {{style_hint}}
    """.strip()
)

JUDGE_PROMPT = PromptTemplate(
    dedent(
        f"""
        You are a sufficiency judge for retrieval results.
        Decide if the search results contain enough information to answer the question.
        Criteria:
        - PASS if a direct answer can be produced without guessing.
        - FAIL if key details are missing or the results are too generic.
        Output exactly one token: PASS or FAIL.
        Question: {{question}}
        Search summary: {{summary}}
        """
    ).strip()
)


def node_search(state: dict, store: Any) -> dict:
    """기본 검색을 수행한다."""
    question = state.get("question")
    docs = store.similarity_search(question, k=3)
    use_hyde = len(docs) < 3
    return {
        "question": question,
        "docs": docs,
        "use_hyde": use_hyde,
    }


def node_judge(state: dict) -> dict:
    """검색 결과가 충분한지 LLM으로 판정한다."""
    question = state.get("question")
    docs = state.get("docs", [])
    summary = " / ".join([str(d) for d in docs])[:1000]
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    result = (JUDGE_PROMPT | llm | StrOutputParser()).invoke(
        {"question": question, "summary": summary}
    )
    use_hyde = not result.strip().upper().startswith("PASS")
    return {"question": question, "docs": docs, "use_hyde": use_hyde}


def node_hyde(state: dict, store: Any) -> dict:
    """LangChain LLM으로 HyDE 문서를 생성 후 재검색한다."""
    question = state.get("question")
    docs = state.get("docs", [])
    style_hint = str(docs[0])[:300] if docs else "일반 가이드 문서 형식"
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    hyde_doc = (HYDE_PROMPT | llm).invoke(
        {"question": question, "style_hint": style_hint}
    ).content
    emb = OpenAIEmbeddings(model="text-embedding-3-small")
    vec = emb.embed_query(hyde_doc)
    hyde_docs = store.similarity_search_by_vector(vec, k=3)
    docs = docs + hyde_docs
    return {"question": question, "docs": docs, "use_hyde": False}


def node_merge(state: dict) -> list[Any]:
    """결과를 반환한다."""
    return state.get("docs", [])


class AdaptiveHyDEGraph:
    """Adaptive HyDE 그래프 구성 클래스."""

    def __init__(self, store: Any) -> None:
        self._store = store
        self._graph = StateGraph(dict)
        self._graph.add_node("search", lambda s: node_search(s, self._store))
        self._graph.add_node("judge", node_judge)
        self._graph.add_node("hyde", lambda s: node_hyde(s, self._store))
        self._graph.add_node("merge", node_merge)
        self._graph.set_entry_point("search")
        self._graph.add_edge("search", "judge")
        self._graph.add_conditional_edges(
            "judge",
            lambda s: s.get("use_hyde", False),
            {True: "hyde", False: "merge"},
        )
        self._graph.add_edge("hyde", "merge")
        self._graph.add_edge("merge", END)

    def build(self):
        """컴파일된 그래프를 반환한다."""
        return self._graph.compile()
```

---

## 5. 체크리스트

- HyDE 적용 기준이 명확한가?
- HyDE 문서 길이가 과도하지 않은가?
- 기본 검색과 HyDE 결과 병합 규칙이 있는가?
- 비용/지연 시간 상한이 정의되어 있는가?
