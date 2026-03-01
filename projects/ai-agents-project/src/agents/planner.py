# ============================================
# Planner Agent - Query Decomposition
# ============================================
# Decomposes a research query into 1-3 focused sub-topics.
# Teaches: with_structured_output() for guaranteed JSON parsing.

import time
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from src.models.state import ResearchState, PlannerOutput
from src.config import get_model_name, get_pipeline_config
from src.guardrails import check_budget

logger = logging.getLogger(__name__)


def planner(state: ResearchState) -> dict:
    """Decompose a research query into focused sub-topics.

    Uses with_structured_output() to get guaranteed PlannerOutput schema.
    Returns {sub_topics, research_plan, token_count, pipeline_trace}.
    """
    start = time.time()

    if not check_budget(state.get("token_count", 0)):
        return {
            "errors": ["Budget exceeded before planning"],
            "sub_topics": [state["query"]],
            "research_plan": "Direct research (budget constraint)",
            "pipeline_trace": [{"agent": "planner", "status": "skipped", "reason": "budget"}],
        }

    try:
        cfg = get_pipeline_config()
        max_topics = cfg.get("max_sub_topics", 3)

        llm = ChatGoogleGenerativeAI(model=get_model_name(), temperature=0)
        structured_llm = llm.with_structured_output(PlannerOutput)

        result = structured_llm.invoke(
            f"You are a research planner. Break this research query into "
            f"1-{max_topics} focused sub-topics that can be researched independently. "
            f"Each sub-topic should be a specific, searchable question.\n\n"
            f"Query: {state['query']}\n\n"
            f"Return exactly 1-{max_topics} sub-topics and a brief research strategy."
        )

        # Enforce max_topics limit
        sub_topics = result.sub_topics[:max_topics]

        duration = int((time.time() - start) * 1000)
        logger.info(f"Planner: {len(sub_topics)} sub-topics in {duration}ms")

        return {
            "sub_topics": sub_topics,
            "research_plan": result.research_plan,
            "token_count": 500,
            "pipeline_trace": [{
                "agent": "planner",
                "duration_ms": duration,
                "tokens": 500,
                "summary": f"Decomposed into {len(sub_topics)} sub-topics",
            }],
        }
    except Exception as e:
        logger.error(f"Planner error: {e}")
        # Fallback: use original query as single sub-topic
        return {
            "sub_topics": [state["query"]],
            "research_plan": f"Direct research (planner error: {e})",
            "token_count": 0,
            "errors": [f"Planner error: {e}"],
            "pipeline_trace": [{
                "agent": "planner",
                "duration_ms": int((time.time() - start) * 1000),
                "tokens": 0,
                "summary": "Fallback to direct query",
            }],
        }
