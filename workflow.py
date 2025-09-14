import aiosqlite
from pathlib import Path
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
import logging

from config import CHECKPOINT_DB
from models import AgentState
from agents import (
    question_rewriter, 
    question_classifier, 
    on_topic_router,
    retriever_node,
    batch_relevant_code_chunk,
    router,
    refine_question,
    agent_selection,
    code_analyst_router,
    security_router,
    performance_router,
    present_teller_code_analyst,
    present_teller_security,
    present_teller_performance,
    present_teller_quality,
    code_analyst_agent,
    security_agent,
    performance_agent,
    quality_agent,
    cannot_answer,
    off_topic
)


async def create_workflow():
    """Create and configure the LangGraph workflow"""
    workflow = StateGraph(AgentState)
    
    # Add all nodes
    workflow.add_node("question_rewriter", question_rewriter)
    workflow.add_node("question_classifier", question_classifier)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("batch_relevant_code_chunk", batch_relevant_code_chunk)
    workflow.add_node("refine_question", refine_question)
    workflow.add_node("agent_selection", agent_selection)
    workflow.add_node("present_teller_code_analyst", present_teller_code_analyst)
    workflow.add_node("present_teller_security", present_teller_security)
    workflow.add_node("present_teller_performance", present_teller_performance)
    workflow.add_node("present_teller_quality", present_teller_quality)
    workflow.add_node("code_analyst_agent", code_analyst_agent)
    workflow.add_node("security_agent", security_agent)
    workflow.add_node("performance_agent", performance_agent)
    workflow.add_node("quality_agent", quality_agent)
    workflow.add_node("cannot_answer", cannot_answer)
    workflow.add_node("off_topic", off_topic)
    
    # Add edges
    workflow.add_edge("question_rewriter", "question_classifier")
    workflow.add_conditional_edges(
        "question_classifier", 
        on_topic_router, 
        {"retriever": "retriever", "off_topic": "off_topic"}
    )
    workflow.add_edge("retriever", "batch_relevant_code_chunk")
    workflow.add_conditional_edges(
        "batch_relevant_code_chunk", 
        router, 
        {
            "agent_selection": "agent_selection", 
            "refine_question": "refine_question", 
            "cannot_answer": "cannot_answer"
        }
    )
    workflow.add_edge("refine_question", "retriever")
    workflow.add_edge("agent_selection", "present_teller_code_analyst")
    workflow.add_edge("present_teller_code_analyst", "code_analyst_agent")
    workflow.add_conditional_edges(
        "code_analyst_agent", 
        code_analyst_router,
        {
            "present_teller_security": "present_teller_security",
            "present_teller_performance": "present_teller_performance",
            "present_teller_quality": "present_teller_quality",
            END: END
        }
    )
    workflow.add_edge("present_teller_security", "security_agent")
    workflow.add_conditional_edges(
        "security_agent", 
        security_router,
        {
            "present_teller_performance": "present_teller_performance",
            "present_teller_quality": "present_teller_quality",
            END: END
        }
    )
    workflow.add_edge("present_teller_performance", "performance_agent")
    workflow.add_conditional_edges(
        "performance_agent", 
        performance_router,
        {
            "present_teller_quality": "present_teller_quality",
            END: END
        }
    )
    workflow.add_edge("present_teller_quality", "quality_agent")
    workflow.add_edge("quality_agent", END)
    workflow.add_edge("off_topic", END)
    workflow.add_edge("cannot_answer", END)
    workflow.set_entry_point("question_rewriter")
    
    try:
        checkpoint_db_path = Path(CHECKPOINT_DB)
        checkpoint_db_path.parent.mkdir(exist_ok=True)
        conn = await aiosqlite.connect(CHECKPOINT_DB)
        checkpointer = AsyncSqliteSaver(conn=conn)
        logging.info(f"Checkpointer at: {CHECKPOINT_DB}")
    except Exception as e:
        logging.info(f"Checkpointer error: {e}")
        raise e
    
    return workflow.compile(checkpointer=checkpointer)


workflow_app = None


async def get_workflow():
    """Get the workflow instance (singleton pattern)"""
    global workflow_app
    if workflow_app is None:
        workflow_app = await create_workflow()
    return workflow_app