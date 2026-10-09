"""
Phase 4 -- Self-Correction Loop as a LangGraph State Machine (PRD 4.4).

Graph shape:

    retrieve -> generate -> evaluate --(passed)--> END
                    ^                     |
                    |                 (failed, retries left)
                    |                     v
                    +------------------ rewrite

`retries_used` is capped by `settings.max_correction_retries` so the loop
always has a "give up gracefully" exit.
"""
from __future__ import annotations

import time
import logging
from typing import TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, StateGraph

from app.core.config import Settings, get_settings
from app.core.retrieval import HybridRetriever
from app.schemas.models import EvaluationResult, QueryTraceStep, RetrievedContext

logger = logging.getLogger(__name__)

GENERATION_SYSTEM_PROMPT = """You are a precise technical assistant answering strictly from the
provided context (retrieved via hybrid vector + knowledge-graph search).
Rules:
- Use ONLY facts present in the context below. Do not use outside knowledge.
- If the context is insufficient to answer, say so explicitly rather than guessing.
- Cite which context snippet(s) support each claim using [1], [2], etc. matching their order.
"""

EVALUATOR_SYSTEM_PROMPT = """You are a strict RAG output evaluator (Self-RAG style guardrail).
Given the ORIGINAL QUERY, the CONTEXT used, and the GENERATED ANSWER, score:

1. hallucination_score (0.0-1.0): fraction of claims in the answer that do
   NOT trace back to the provided context. 0 = fully grounded, 1 = fully hallucinated.
2. relevance_score (0.0-1.0): how fully the answer resolves the original query.
   0 = irrelevant/off-topic, 1 = fully resolves the query.

Return structured output only."""

REWRITE_SYSTEM_PROMPT = """You rewrite underperforming RAG queries to improve retrieval.
Given the ORIGINAL QUERY and why the previous attempt failed (low relevance
and/or hallucination), produce a single improved, more specific search
query that is more likely to retrieve grounding evidence. Return ONLY the
rewritten query text, nothing else."""


class GraphState(TypedDict):
    original_query: str
    current_query: str
    contexts: list[RetrievedContext]
    answer: str
    evaluation: EvaluationResult | None
    retries_used: int
    max_retries: int
    trace: list[QueryTraceStep]


def invoke_with_retry(model, input_arg, max_attempts: int = 3, backoff_sec: float = 4.0):
    """Executes model invocation with automatic rate-limit (429) backoff retry."""
    for attempt in range(max_attempts):
        try:
            return model.invoke(input_arg)
        except Exception as exc:
            err_msg = str(exc)
            if ("429" in err_msg or "rate_limit" in err_msg.lower()) and attempt < max_attempts - 1:
                logger.warning("Rate limit encountered. Sleeping %.1fs before retry %d/%d...", backoff_sec, attempt + 1, max_attempts)
                time.sleep(backoff_sec * (attempt + 1))
                continue
            raise exc


def build_self_correction_graph(retriever: HybridRetriever, llm: BaseChatModel, settings: Settings | None = None):
    settings = settings or get_settings()

    def retrieve_node(state: GraphState) -> GraphState:
        contexts = retriever.retrieve(state["current_query"])
        state["contexts"] = contexts
        state["trace"].append(
            QueryTraceStep(node="retrieve", detail=f"Retrieved {len(contexts)} chunks for query: {state['current_query']!r}")
        )
        return state

    def generate_node(state: GraphState) -> GraphState:
        context_block = "\n\n".join(
            f"[{i+1}] (source={c.source}, CRI={c.composite_relevance_index:.4f}, hops={c.graph_hops_used})\n{c.text}"
            for i, c in enumerate(state["contexts"])
        ) or "(no context retrieved)"

        messages = [
            ("system", GENERATION_SYSTEM_PROMPT),
            ("human", f"Context:\n{context_block}\n\nQuestion: {state['original_query']}"),
        ]
        response = invoke_with_retry(llm, messages)
        state["answer"] = response.content if hasattr(response, "content") else str(response)
        state["trace"].append(QueryTraceStep(node="generate", detail=f"Generated {len(state['answer'])} chars"))
        return state

    def evaluate_node(state: GraphState) -> GraphState:
        context_block = "\n\n".join(f"[{i+1}] {c.text}" for i, c in enumerate(state["contexts"])) or "(none)"
        structured_llm = llm.with_structured_output(EvaluationResult)
        try:
            evaluation: EvaluationResult = invoke_with_retry(
                structured_llm,
                [
                    ("system", EVALUATOR_SYSTEM_PROMPT),
                    (
                        "human",
                        f"ORIGINAL QUERY: {state['original_query']}\n\n"
                        f"CONTEXT:\n{context_block}\n\nGENERATED ANSWER:\n{state['answer']}",
                    ),
                ]
            )
        except Exception:  # noqa: BLE001
            evaluation = EvaluationResult(
                hallucination_score=0.0, relevance_score=1.0, passed=True, reasoning="Evaluator unavailable; fail-open."
            )

        evaluation.passed = (
            evaluation.hallucination_score <= settings.hallucination_threshold
            and evaluation.relevance_score >= settings.relevance_threshold
        )
        state["evaluation"] = evaluation
        state["trace"].append(
            QueryTraceStep(
                node="evaluate",
                detail=(
                    f"hallucination={evaluation.hallucination_score:.2f}, "
                    f"relevance={evaluation.relevance_score:.2f}, passed={evaluation.passed}"
                ),
            )
        )
        return state

    def rewrite_node(state: GraphState) -> GraphState:
        state["retries_used"] += 1
        response = invoke_with_retry(
            llm,
            [
                ("system", REWRITE_SYSTEM_PROMPT),
                (
                    "human",
                    f"ORIGINAL QUERY: {state['original_query']}\n"
                    f"PREVIOUS ATTEMPT: {state['current_query']}\n"
                    f"EVALUATION: {state['evaluation'].reasoning if state['evaluation'] else 'n/a'}",
                ),
            ]
        )
        rewritten = response.content if hasattr(response, "content") else str(response)
        state["current_query"] = rewritten.strip() or state["current_query"]
        state["trace"].append(QueryTraceStep(node="rewrite", detail=f"Rewritten query: {state['current_query']!r}"))
        return state

    def route_after_evaluate(state: GraphState) -> str:
        if state["evaluation"] and state["evaluation"].passed:
            return "end"
        if state["retries_used"] >= state["max_retries"]:
            return "end"
        return "rewrite"

    graph = StateGraph(GraphState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("rewrite", rewrite_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "evaluate")
    graph.add_conditional_edges("evaluate", route_after_evaluate, {"end": END, "rewrite": "rewrite"})
    graph.add_edge("rewrite", "retrieve")

    return graph.compile()


def run_self_correcting_query(
    query: str, retriever: HybridRetriever, llm: BaseChatModel, settings: Settings | None = None
) -> GraphState:
    settings = settings or get_settings()
    app_graph = build_self_correction_graph(retriever, llm, settings)
    initial_state: GraphState = {
        "original_query": query,
        "current_query": query,
        "contexts": [],
        "answer": "",
        "evaluation": None,
        "retries_used": 0,
        "max_retries": settings.max_correction_retries,
        "trace": [],
    }
    final_state = app_graph.invoke(initial_state)
    return final_state
