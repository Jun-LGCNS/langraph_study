# 목적: 번역 처리를 LangGraph로 구성한다.
# 설명: 입력 → 안전 분류 → 번역 → QC → 재번역 → 응답 흐름을 연결한다.
# 디자인 패턴: 파이프라인 + 빌더
# 참조: docs/04_string_tricks/01_yes_no_파서.md, docs/04_string_tricks/02_single_choice_파서.md

"""번역 그래프 구성 모듈."""

from langgraph.graph import START, END, StateGraph

from firstsession.core.translate.state.translation_state import TranslationState
from firstsession.core.translate.nodes.normalize_input_node import NormalizeInputNode
from firstsession.core.translate.nodes.safeguard_classify_node import SafeguardClassifyNode
from firstsession.core.translate.nodes.safeguard_decision_node import SafeguardDecisionNode
from firstsession.core.translate.nodes.safeguard_fail_response_node import SafeguardFailResponseNode
from firstsession.core.translate.nodes.translate_node import TranslateNode
from firstsession.core.translate.nodes.quality_check_node import QualityCheckNode
from firstsession.core.translate.nodes.retry_gate_node import RetryGateNode
from firstsession.core.translate.nodes.retry_translate_node import RetryTranslateNode
from firstsession.core.translate.nodes.response_node import ResponseNode

class TranslateGraph:
    """번역 그래프 실행기."""

    def __init__(self) -> None:
        """그래프를 초기화한다."""
        graph = self._build_graph()
        self._app = graph.compile()

    def run(self, state: TranslationState) -> TranslationState:
        """번역 그래프를 실행한다.

        Args:
            state: 번역 입력 상태.

        Returns:
            TranslationState: 번역 결과 상태.
        """
        return self._app.invoke(state)
        raise NotImplementedError("번역 그래프 실행 로직을 구현해야 합니다.")

    def _build_graph(self) -> StateGraph:
        """번역 그래프를 구성한다.

        Returns:
            StateGraph: 구성된 그래프.
        """
        graph: StateGraph = StateGraph(TranslationState)

        # --- 노드 인스턴스(무상태) ---
        normalize = NormalizeInputNode()
        safeguard_classify = SafeguardClassifyNode()
        safeguard_decision = SafeguardDecisionNode()
        safeguard_fail = SafeguardFailResponseNode()
        translate = TranslateNode()
        quality_check = QualityCheckNode()
        retry_gate = RetryGateNode()
        retry_translate = RetryTranslateNode()
        response = ResponseNode()

        graph.add_node("normalize", normalize.run)
        graph.add_node("safeguard_classify", safeguard_classify.run)
        graph.add_node("safeguard_decision", safeguard_decision.run)
        graph.add_node("safeguard_fail", safeguard_fail.run)
        graph.add_node("translate", translate.run)
        graph.add_node("quality_check", quality_check.run)
        graph.add_node("retry_gate", retry_gate.run)
        graph.add_node("retry_translate", retry_translate.run)
        graph.add_node("response", response.run)

        graph.add_edge(START, "normalize")
        graph.add_edge("normalize", "safeguard_classify")
        graph.add_edge("safeguard_classify", "safeguard_decision")

        def _get(state, key, default=None):
            if isinstance(state, dict):
                return state.get(key, default)
            return getattr(state, key, default)

        def route_after_safeguard(state: TranslationState) -> str:
            label = (_get(state, "safeguard_label") or "PASS").strip()
            return "translate" if label == "PASS" else "safeguard_fail"

        def route_after_retry_gate(state: TranslationState) -> str:
            qc = (_get(state, "qc_passed") or "NO").strip().upper()

            if qc == "YES":
                return "response"

            retry_count = int(_get(state, "retry_count", 0) or 0)
            max_retry_count = int(_get(state, "max_retry_count", 0) or 0)

            # 기본값 방어(0이면 RetryGateNode에서 1로 넣지만, 라우팅 함수도 안전하게)
            if max_retry_count <= 0:
                max_retry_count = 1

            if retry_count < max_retry_count:
                return "retry_translate"
            return "response"


        graph.add_conditional_edges(
            "safeguard_decision",
            route_after_safeguard,
            {
                "translate": "translate",
                "safeguard_fail": "safeguard_fail",
            },
        )

        graph.add_edge("safeguard_fail", "response")
        graph.add_edge("response", END)

        graph.add_edge("translate", "quality_check")
        graph.add_edge("quality_check", "retry_gate")

        graph.add_conditional_edges(
            "retry_gate",
            route_after_retry_gate,
            {
                "retry_translate": "retry_translate",
                "response": "response",
            },
        )

        graph.add_edge("retry_translate", "quality_check")

        return graph



        # TODO: START 노드에서 시작하는 흐름을 명시한다.
        # - START -> NormalizeInputNode
        # TODO: 노드 등록 방식은 두 가지 모두 가능하다.
        # - 함수형: graph.add_node("normalize", normalize_input)
        # - 클래스형: graph.add_node("normalize", self.normalize_input_node.run)
        #   - 클래스형은 무상태로 설계하고, 공유 데이터는 state에만 기록한다.
        # TODO: 다음 노드들을 추가하고 엣지를 연결한다.
        # - NormalizeInputNode: 입력 정규화
        # - SafeguardClassifyNode: PASS/PII/HARMFUL/PROMPT_INJECTION 판정
        # - SafeguardDecisionNode: PASS 여부 기록 및 오류 메시지 세팅
        # - SafeguardFailResponseNode: 차단 응답 구성
        # - TranslateNode: 번역 수행
        # - QualityCheckNode: 번역 품질 YES/NO 판정
        # - RetryGateNode: 재번역 가능 여부 판단
        # - RetryTranslateNode: 재번역 수행
        # - ResponseNode: 최종 응답 구성

        # TODO: 조건부 엣지 설계(구체 경로 예시)
        # - NormalizeInputNode -> SafeguardClassifyNode -> SafeguardDecisionNode
        # - SafeguardDecisionNode에서 PASS가 아니면 SafeguardFailResponseNode -> ResponseNode -> END
        #   - safeguard_label: PASS/PII/HARMFUL/PROMPT_INJECTION (안전 분류 결과)
        #   - error_message: 차단 시 사용자에게 전달할 메시지
        # - PASS면 TranslateNode -> QualityCheckNode -> RetryGateNode
        # - RetryGateNode에서 qc_passed가 YES이면 ResponseNode -> END
        #   - qc_passed: YES/NO (번역 품질 검사 결과)
        # - RetryGateNode에서 qc_passed가 NO이고 재시도 가능하면 RetryTranslateNode -> QualityCheckNode로 루프
        #   - retry_count: 재시도 횟수
        #   - max_retry_count: 최대 재시도 횟수
        # - RetryGateNode에서 qc_passed가 NO이고 재시도 불가이면 ResponseNode -> END
        
        raise NotImplementedError("번역 그래프 구성 로직을 구현해야 합니다.")
