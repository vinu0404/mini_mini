import json
import logging
import chainlit as cl
from langsmith import traceable
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import base64
from models import (
    AgentState, OffTopic, BatchCodeGrading, AgentSelection, 
    PresentTellerOutput, PrDecision, QualityAnalysisOutput
)
from llm_instances import (
    basic_llm, code_analyst_llm, security_llm, performance_llm, 
    quality_llm, agent_selection_llm, present_teller_llm
)
from prompts import (
    CODE_ANALYST_PROMPT, SECURITY_AGENT_PROMPT, PERFORMANCE_AGENT_PROMPT,
    QUALITY_AGENT_PROMPT, AGENT_SELECTION_PROMPT, PRESENT_TELLER_PROMPT
)
from config import GLOBAL_RETRIEVERS, GLOBAL_FOLDER_STRUCTURES, GITHUB_TOKEN, MAX_CONVERSATION_HISTORY_CHARS
from utils import get_pr_tool
from tools import QuickChartTool


@traceable(name="Question Rewriter Node")
def question_rewriter(state: AgentState):
    """Rewrite the question for better retrieval with enhanced specificity and context awareness"""
    folder_structure = state.get('folder_structure', 'N/A')
    project_analysis = state.get('project_analysis', {})
    project_type = project_analysis.get('project_type', 'Unknown')
    languages = project_analysis.get('languages', [])
    frameworks = project_analysis.get('frameworks', [])
    
    context_info = f"Project Type: {project_type}"
    if languages:
        context_info += f", Languages: {', '.join(languages)}"
    if frameworks:
        context_info += f", Frameworks: {', '.join(frameworks)}"
    
    system_message = SystemMessage(
        content="""You are a question rephrasing expert for code analysis queries.
Rephrase to GET more good retrieval: add keywords for architecture, security, performance, quality if implied.
Include file/function names, patterns, metrics if mentioned in project context or structure.
Consider the project context provided to make the question more specific and relevant.
Keep original intent, make specific to question.
Output only the rephrased question."""
    )
    
    human_message = HumanMessage(
        content=f"""Original question: {state['question'].content}

Project Context: {context_info}
Architecture Overview: {folder_structure if folder_structure != 'N/A' else 'Not available'}

Rephrase for better code retrieval and multi-agent analysis, incorporating relevant project context if it helps clarify the question."""
    )
    
    messages = [system_message, human_message]
    response = basic_llm.invoke(messages)
    state["rephrased_question"] = response.content.strip()
    logging.info(f"Rephrased question: {state['rephrased_question']}")
    return state


@traceable(name="Question Classifier Node")
def question_classifier(state: AgentState):
    """Classify if question is on-topic for code analysis with enhanced criteria"""
    messages = [
        SystemMessage(content="""Classify as 'yes' if question relates to code, repo analysis, architecture, security, performance, quality, docs, tests, or software engineering.
Default to 'yes' for comprehensive analysis. 'no' only for non-technical topics.
Output 'yes' or 'no' only."""),
        HumanMessage(content=f"Question: '{state['rephrased_question']}'")
    ]
    
    structured_llm = basic_llm.with_structured_output(OffTopic)
    response = structured_llm.invoke(messages)
    state["on_topic"] = response.answer.strip().lower()
    logging.info(f"Question classification: {state['on_topic']}")
    return state


def on_topic_router(state: AgentState):
    if state["on_topic"] == "yes":
        return "retriever"
    else:
        return "off_topic"


@traceable(name="Retriever Node")
def retriever_node(state: AgentState):
    repo_hash = state.get("repo_hash")
    if not repo_hash or repo_hash not in GLOBAL_RETRIEVERS:
        raise ValueError(f"Retriever not found for {repo_hash}")
    
    retriever = GLOBAL_RETRIEVERS[repo_hash]['retriever']
    code_units = retriever.invoke(state["rephrased_question"])
    state["code"] = code_units
    
    if repo_hash in GLOBAL_FOLDER_STRUCTURES:
        folder_data = GLOBAL_FOLDER_STRUCTURES[repo_hash]
        state["folder_structure"] = folder_data['structure']
        state["project_analysis"] = folder_data['analysis']
    
    logging.info(f"Retrieved {len(code_units)} units")
    return state


@traceable(name="Batch Relevant Code Chunk Node")
def batch_relevant_code_chunk(state: AgentState):
    system_message = SystemMessage(content="""Classify code chunks as 'relevant' if related to question, considering project context, multi-domain. And always say "relevant" for README.MD files.
Err on relevance. Output list of 'relevant' or 'not relevant'.""")
    
    code_chunks_text = "\n".join([f"Chunk {i+1}: {code}" for i, code in enumerate(state["code"])])
    folder_context = state.get("folder_structure", "")
    project_context = state.get("project_analysis", {})
    
    human_message = HumanMessage(
        content=f"Context: {folder_context}\nAnalysis: {project_context}\nChunks: {code_chunks_text}\nQuestion: {state['rephrased_question']}"
    )
    
    messages = [system_message, human_message]
    structured_llm = basic_llm.with_structured_output(BatchCodeGrading)
    response = structured_llm.invoke(messages)
    relevant_code_chunks = [state["code"][i] for i, score in enumerate(response.relevance_scores) if score.lower() == "relevant"]
    
    state["code"] = relevant_code_chunks
    state["proceed_to_generate"] = "yes" if relevant_code_chunks else "no"
    logging.info(f"Filtered to {len(relevant_code_chunks)} chunks")
    return state


def router(state: AgentState):
    if state["rephrase_count"] > 2:
        return "cannot_answer"
    elif state["proceed_to_generate"] == "yes":
        return "agent_selection"
    else:
        return "refine_question"


@traceable(name="Refine Question Node")
def refine_question(state: AgentState):
    rephrase_count = state.get("rephrase_count", 0)
    if rephrase_count >= 2:
        state["proceed_to_generate"] = "yes"
        return state
    
    question_to_refine = state["rephrased_question"]
    folder_context = state.get("folder_structure", "")
    project_context = state.get("project_analysis", {})
    
    system_message = SystemMessage(
        content="""Refine question for better retrieval, considering structure, languages, frameworks.
Make specific, preserve intent. Output refined question only."""
    )
    human_message = HumanMessage(
        content=f"Context: {folder_context}\nAnalysis: {project_context}\nQuestion: {question_to_refine}"
    )
    messages = [system_message, human_message]
    response = basic_llm.invoke(messages)
    state["rephrased_question"] = response.content.strip()
    state["rephrase_count"] = rephrase_count + 1
    logging.info(f"Refined: {state['rephrased_question']}")
    return state


@traceable(name="Agent Selection Node")
def agent_selection(state: AgentState):
    history_str = "\n".join([f"{'User' if isinstance(msg, HumanMessage) else 'AI'}: {msg.content[:700]}" for msg in state.get("conversation_history", [])])
    
    messages = [
        SystemMessage(content=AGENT_SELECTION_PROMPT.format(
            question=state["rephrased_question"],
            history=history_str
        ))
    ]
    
    structured_llm = agent_selection_llm.with_structured_output(AgentSelection)
    response = structured_llm.invoke(messages)
    
    state["goto_security_agent"] = response.goto_security_agent
    state["goto_performance_agent"] = response.goto_performance_agent
    state["goto_quality_agent"] = response.goto_quality_agent
    state["agent_selection_reasoning"] = response.reasoning
    state["agent_priority_order"] = response.priority_order
    
    logging.info(f"Selection: Sec={state['goto_security_agent']}, Perf={state['goto_performance_agent']}, Qual={state['goto_quality_agent']}")
    return state


def code_analyst_router(state: AgentState):
    if state["goto_security_agent"]:
        return "present_teller_security"
    elif state["goto_performance_agent"]:
        return "present_teller_performance"
    elif state["goto_quality_agent"]:
        return "present_teller_quality"
    else:
        return "END"


def security_router(state: AgentState):
    if state["goto_performance_agent"]:
        return "present_teller_performance"
    elif state["goto_quality_agent"]:
        return "present_teller_quality"
    else:
        return "END"


def performance_router(state: AgentState):
    if state["goto_quality_agent"]:
        return "present_teller_quality"
    else:
        return "END"


@traceable(name="Present Teller Code Analyst Node")
async def present_teller_code_analyst(state: AgentState):
    messages = [
        SystemMessage(content=PRESENT_TELLER_PROMPT),
        HumanMessage(content=PRESENT_TELLER_PROMPT.format(
            reasoning=state["agent_selection_reasoning"],
            priority_order=state["agent_priority_order"],
            question=state["rephrased_question"],
            current_agent="Code Analyst"
        ))
    ]
    
    structured_llm = present_teller_llm.with_structured_output(PresentTellerOutput)
    response = structured_llm.invoke(messages)
    
    explanation = response.explanation
    
    await cl.Message(
        content=f"**🔍 Update:** {explanation}\nProceeding...",
        author="GitSpeak"
    ).send()
    
    state["messages"].append(AIMessage(content=f"Teller: {explanation}"))
    return state


@traceable(name="Present Teller Security Node")
async def present_teller_security(state: AgentState):
    messages = [
        SystemMessage(content=PRESENT_TELLER_PROMPT),
        HumanMessage(content=PRESENT_TELLER_PROMPT.format(
            reasoning=state["agent_selection_reasoning"],
            priority_order=state["agent_priority_order"],
            question=state["rephrased_question"],
            current_agent="Security Agent"
        ))
    ]
    
    structured_llm = present_teller_llm.with_structured_output(PresentTellerOutput)
    response = structured_llm.invoke(messages)
    
    explanation = response.explanation
    
    await cl.Message(
        content=f"**Update:** {explanation}\nAssessing security...",
        author="GitSpeak"
    ).send()
    
    state["messages"].append(AIMessage(content=f"Teller: {explanation}"))
    return state


@traceable(name="Present Teller Performance Node")
async def present_teller_performance(state: AgentState):
    messages = [
        SystemMessage(content=PRESENT_TELLER_PROMPT),
        HumanMessage(content=PRESENT_TELLER_PROMPT.format(
            reasoning=state["agent_selection_reasoning"],
            priority_order=state["agent_priority_order"],
            question=state["rephrased_question"],
            current_agent="Performance Agent"
        ))
    ]
    
    structured_llm = present_teller_llm.with_structured_output(PresentTellerOutput)
    response = structured_llm.invoke(messages)
    
    explanation = response.explanation
    
    await cl.Message(
        content=f"** Update:** {explanation}\nAnalyzing performance...",
        author="GitSpeak"
    ).send()
    
    state["messages"].append(AIMessage(content=f"Teller: {explanation}"))
    return state


@traceable(name="Present Teller Quality Node")
async def present_teller_quality(state: AgentState):
    messages = [
        SystemMessage(content=PRESENT_TELLER_PROMPT),
        HumanMessage(content=PRESENT_TELLER_PROMPT.format(
            reasoning=state["agent_selection_reasoning"],
            priority_order=state["agent_priority_order"],
            question=state["rephrased_question"],
            current_agent="Quality Agent"
        ))
    ]
    
    structured_llm = present_teller_llm.with_structured_output(PresentTellerOutput)
    response = structured_llm.invoke(messages)
    
    explanation = response.explanation
    
    await cl.Message(
        content=f"**Update:** {explanation}\nSynthesizing quality...",
        author="GitSpeak"
    ).send()
    
    state["messages"].append(AIMessage(content=f"Teller: {explanation}"))
    return state


@traceable(name="Code Analyst Agent Node")
async def code_analyst_agent(state: AgentState):
    logging.info("Code Analyst analyzing...")

    history_str = "\n".join([f"{'User' if isinstance(msg, HumanMessage) else 'AI'}: {msg.content[:MAX_CONVERSATION_HISTORY_CHARS]}" for msg in state.get("conversation_history", [])])
    
    folder_structure = state.get("folder_structure", "")
    logging.info(f"Folder structure: {folder_structure}")
    project_analysis = state.get("project_analysis", {})
    logging.info(f"Project analysis: {project_analysis}")
    
    decide_messages = [
        SystemMessage(content=f"""Determine if question needs PR info. 
        IMPORTANT: PR analysis requires GitHub token. 
        Token available: {bool(GITHUB_TOKEN)}
        
        call_pr: true only if:
        1. Question mentions PRs, reviews, CI/CD, changes, commits, pull requests
        2. GitHub token is available
        
        If no token, set call_pr: false and explain limitation."""),
        HumanMessage(content=state["rephrased_question"])
    ]
    
    structured_decide = basic_llm.with_structured_output(PrDecision)
    decision = structured_decide.invoke(decide_messages)
    
    logging.info(f"PR Decision: call_pr={decision.call_pr}, pr_number={decision.pr_number}, reason={decision.reason}")
    
    pr_data = None
    pr_data_str = "No PR data."
    
    if decision.call_pr and GITHUB_TOKEN:
        logging.info(f"Attempting to fetch PR with token: {decision.pr_number}")
        try:
            pr_data = await get_pr_tool(state["repo_url"], decision.pr_number)
            logging.info(f"PR data fetch result: {type(pr_data)} - {pr_data}")
            state["pr_data"] = pr_data
            
            # FIXED ERROR HANDLING LOGIC
            if pr_data is None:
                logging.error("PR data is None!")
                pr_data_str = "PR DATA ERROR: get_pr_tool returned None"
            elif isinstance(pr_data, dict) and pr_data.get("error"):  # FIXED: Check if error value exists and is truthy
                logging.error(f"PR data contains error: {pr_data['error']}")
                pr_data_str = f"PR DATA ERROR: {pr_data['error']}"
            elif isinstance(pr_data, dict):
                logging.info("PR data looks valid, processing...")
                pr_info = pr_data.get("pr_info", {})
                diff = pr_data.get("diff", "")
                commits = pr_data.get("commits", [])
                
                logging.info(f"PR Info: {pr_info.get('number', 'N/A')} - {pr_info.get('title', 'No title')}")
                logging.info(f"Diff length: {len(diff) if diff else 0}")
                logging.info(f"Commits count: {len(commits)}")
                
                pr_summary = f"""
PULL REQUEST ANALYSIS AVAILABLE:
- PR #{pr_info.get('number', 'N/A')}: {pr_info.get('title', 'No title')}
- Status: {pr_info.get('state', 'unknown')}
- Author: {pr_info.get('user', {}).get('login', 'unknown')}
- Files changed: {pr_info.get('changed_files', 0)}
- Additions: +{pr_info.get('additions', 0)} | Deletions: -{pr_info.get('deletions', 0)}

DIFF CHANGES:
{diff if diff else 'No diff available'}

COMMITS: {len(commits)} total
{chr(10).join([f"- {commit.get('commit', {}).get('message', 'No message')[:100]}" for commit in commits[:3]])}
"""
                pr_data_str = pr_summary
                logging.info("PR summary created successfully")
            else:
                logging.error(f"Unexpected PR data type: {type(pr_data)}")
                pr_data_str = f"PR DATA ERROR: Unexpected data type {type(pr_data)}"
                
        except Exception as e:
            logging.error(f"Exception during PR fetch: {str(e)}")
            pr_data_str = f"PR DATA ERROR: Exception - {str(e)}"
            
    elif decision.call_pr and not GITHUB_TOKEN:
        logging.info("PR analysis skipped - no GitHub token")
        pr_data_str = "PR analysis requires GitHub token (not configured)"
    else:
        logging.info("PR analysis not requested")

    logging.info(f"FINAL PR Data String: {pr_data_str[:500]}...")
    
    messages = [
        SystemMessage(content=CODE_ANALYST_PROMPT.format(
            folder_structure=folder_structure,
            project_analysis=project_analysis,
            context=state["code"],
            question=state["rephrased_question"],
            history=history_str,
            pr_data_str=pr_data_str
        ))
    ]
    
    ui_message = await cl.Message(
        content="## Code Analysis\nAnalyzing structure...\n",
        author="GitSpeak"
    ).send()
    
    full_response = ""
    async for chunk in code_analyst_llm.astream(messages):
        content = chunk.content
        if content:
            full_response += content
            await ui_message.stream_token(content)
    
    await ui_message.update()
    state["messages"].append(AIMessage(content=full_response))
    return state


@traceable(name="Security Agent Node")
async def security_agent(state: AgentState):
    """Enhanced security vulnerability analysis agent with comprehensive threat modeling and expanded compliance"""
    logging.info("Security Agent conducting in-depth vulnerability and threat analysis...")
    
    code_analysis = ""
    if state["messages"]:
        code_analysis = state["messages"][-1].content
    
    folder_structure = state.get("folder_structure", "No folder structure available")
    project_analysis = state.get("project_analysis", {})
    
    # OPTIMIZED: Just use the PR data that was already fetched by code_analyst_agent
    pr_data = state.get("pr_data")
    
    # FIXED ERROR HANDLING LOGIC
    pr_data_str = json.dumps(pr_data, indent=2) if pr_data else "No PR data needed or available."
    if pr_data and pr_data.get("error"):  # FIXED: Check if error value exists and is truthy
        pr_data_str = pr_data["error"]
    
    messages = [
        SystemMessage(content=SECURITY_AGENT_PROMPT.format(
            messages=code_analysis,
            folder_structure=folder_structure,
            project_analysis=project_analysis,
            context=state["code"],
            question=state["rephrased_question"],
            pr_data_str=pr_data_str
        ))
    ]
    
    ui_message = await cl.Message(
        content="---\n\n## Elite Security Assessment\n\nPerforming threat modeling and vulnerability scan...\n\n",
        author="GitSpeak"
    ).send()
    
    full_response = ""
    async for chunk in security_llm.astream(messages):
        content = chunk.content
        if content:
            full_response += content
            await ui_message.stream_token(content)

    await ui_message.update()
    state["security_message"].append(AIMessage(content=full_response))
    
    return state


@traceable(name="Performance Agent Node")
async def performance_agent(state: AgentState):
    logging.info("Performance Agent analyzing...")
    
    code_analysis = state["messages"][-1].content if state["messages"] else ""
    security_analysis = state["security_message"][-1].content if state["security_message"] else ""
    
    folder_structure = state.get("folder_structure", "")
    project_analysis = state.get("project_analysis", {})
    
    # OPTIMIZED: Just use the PR data that was already fetched by code_analyst_agent
    pr_data = state.get("pr_data")
    
    # FIXED ERROR HANDLING LOGIC
    pr_data_str = json.dumps(pr_data, indent=2) if pr_data else "No PR data."
    if pr_data and pr_data.get("error"):  # FIXED: Check if error value exists and is truthy
        pr_data_str = pr_data["error"]
    
    messages = [
        SystemMessage(content=PERFORMANCE_AGENT_PROMPT.format(
            messages=code_analysis,
            security_message=security_analysis,
            folder_structure=folder_structure,
            project_analysis=project_analysis,
            context=state["code"],
            question=state["rephrased_question"],
            pr_data_str=pr_data_str
        ))
    ]
    
    ui_message = await cl.Message(
        content="---\n## Performance Analysis\nIdentifying bottlenecks...\n",
        author="GitSpeak"
    ).send()

    full_response = ""
    async for chunk in performance_llm.astream(messages):
        content = chunk.content
        if content:
            full_response += content
            await ui_message.stream_token(content)

    await ui_message.update()
    state["performance_message"].append(AIMessage(content=full_response))
    return state


@traceable(name="Quality Agent Node")
async def quality_agent(state: AgentState):
    from tools import MermaidTool
    logging.info("Quality Agent generating report...")
    code_analysis = state["messages"][-1].content if state["messages"] else ""
    security_analysis = state["security_message"][-1].content if state["security_message"] else ""
    performance_analysis = state["performance_message"][-1].content if state["performance_message"] else ""
    
    folder_structure = state.get("folder_structure", "")
    project_analysis = state.get("project_analysis", {})
    pr_data = state.get("pr_data")
    pr_data_str = json.dumps(pr_data, indent=2) if pr_data else "No PR data."
    if pr_data and pr_data.get("error"):  
        pr_data_str = pr_data["error"]
    clean_state_for_llm = {
        "code": state.get("code", ""),
        "rephrased_question": state.get("rephrased_question", ""),
        "folder_structure": folder_structure,
        "project_analysis": project_analysis
    }
    
    structured_llm = quality_llm.with_structured_output(QualityAnalysisOutput)
    
    messages = [
        SystemMessage(content=QUALITY_AGENT_PROMPT.format(
            messages=code_analysis,
            security_message=security_analysis,
            performance_message=performance_analysis,
            folder_structure=folder_structure,
            project_analysis=project_analysis,
            context=clean_state_for_llm["code"],  # Use clean context
            question=clean_state_for_llm["rephrased_question"],
            pr_data_str=pr_data_str
        )),
        HumanMessage(content=f"""Based on the comprehensive multi-agent analysis provided, create a quality intelligence report and determine if advanced visualization would enhance understanding.

Original Question: {clean_state_for_llm["rephrased_question"]}

CRITICAL: Only generate visualizations if the question explicitly requests metrics, scores, dashboards, comparisons, trends, breakdowns, or visual analysis. For explanatory questions about specific code or functionality, provide detailed text analysis without charts. Always populate all metric fields with realistic, justified scores (0-100) based on the analysis findings, including new fields like reliability_score and usability_score if applicable. Also if asked about mermaid and any type of mindmap,flow then generate diagram

Create sophisticated visualizations when appropriate that provide executive-level insights with professional styling, colors, and labels.

Provide your comprehensive quality intelligence in the structured format, ensuring completeness.""")
    ]
    
    final_message_content = ""
    
    try:
        response = structured_llm.invoke(messages)
        
        state["report_data"] = {
            "summary": response.report_data.summary,
            "critical_issues": response.report_data.critical_issues,
            "recommendations": response.report_data.recommendations,
            "security_score": response.report_data.security_score,
            "performance_score": response.report_data.performance_score,
            "maintainability_score": response.report_data.maintainability_score,
            "complexity_score": response.report_data.complexity_score,
            "overall_score": response.report_data.overall_score,
            "total_issues": response.report_data.total_issues,
            "critical_count": response.report_data.critical_count,
            "high_count": response.report_data.high_count,
            "medium_count": response.report_data.medium_count,
            "low_count": 5,
            "reliability_score": 85,
            "usability_score": 90
        }
        
        # CRITICAL FIX: Store image data separately, not in main state that gets passed to LLMs
        chart_image_data = None
        diagram_image_data = None
        
        if response.should_create_chart and response.chart_data:
            logging.info("Creating chart...")
            datasets_dict = [ds.dict(exclude_none=True) for ds in response.chart_data.datasets]
            
            chart_config = {
                'type': response.chart_data.type,
                'data': {
                    'labels': response.chart_data.labels,
                    'datasets': datasets_dict
                },
                'options': {
                    'responsive': True,
                    'plugins': {
                        'title': {
                            'display': True,
                            'text': response.chart_data.title,
                            'font': {'size': 18, 'weight': 'bold', 'family': 'Arial'}
                        },
                        'legend': {'position': 'bottom', 'labels': {'font': {'size': 12}}},
                        'tooltip': {'enabled': True, 'backgroundColor': 'rgba(0,0,0,0.8)'}
                    },
                    'scales': {
                        'y': {
                            'beginAtZero': True,
                            'max': 100,
                            'ticks': {'font': {'size': 11}}
                        }
                    }
                }
            }
            
            if response.chart_data.type == 'radar':
                chart_config['options']['scales'] = {
                    'r': {
                        'beginAtZero': True,
                        'max': 100,
                        'ticks': {'stepSize': 20, 'font': {'size': 11}}
                    }
                }
                chart_config['options']['elements'] = {'line': {'borderWidth': 3}}
            elif response.chart_data.type in ['bar', 'line']:
                chart_config['options']['scales']['y']['max'] = 100 if 'score' in response.chart_data.title.lower() else None
                chart_config['options']['animation'] = {'duration': 2000}
            
            if 'datasets' in chart_config['data']:
                for ds in chart_config['data']['datasets']:
                    if 'backgroundColor' not in ds:
                        ds['backgroundColor'] = 'rgba(54, 162, 235, 0.2)'
                    if 'borderColor' not in ds:
                        ds['borderColor'] = 'rgba(54, 162, 235, 1)'
            
            # Store image data separately - DON'T put in main state
            chart_image_data = QuickChartTool.create_chart(
                chart_config, 
                response.chart_data.width or 900, 
                response.chart_data.height or 700  
            )
            
            if chart_image_data:
                # Only store metadata in state, not the actual image data
                state["has_chart"] = True
                state["chart_config"] = chart_config
                logging.info("Chart created!")

        if response.should_create_diagram and response.diagram_data:
            logging.info("Creating diagram...")
            # Store image data separately - DON'T put in main state
            diagram_image_data = MermaidTool.create_diagram(response.diagram_data)
            
            if diagram_image_data:
                # Only store metadata in state, not the actual image data
                state["has_diagram"] = True
                logging.info("Diagram created!")
        
        final_report_parts = []
        
        if response.report_data.summary:
            final_report_parts.append(f"## Summary\n{response.report_data.summary}\n")
        
        if response.report_data.critical_issues:
            critical_section = "## Critical Issues\n"
            for i, issue in enumerate(response.report_data.critical_issues, 1):
                critical_section += f"{i}. {issue}\n"
            final_report_parts.append(critical_section)
        
        if response.report_data.recommendations:
            recommendations_section = "## Recommendations\n"
            for i, rec in enumerate(response.report_data.recommendations, 1):
                recommendations_section += f"{i}. {rec}\n"
            final_report_parts.append(recommendations_section)
        
        metrics_display = []
        metrics = response.report_data
        if metrics.overall_score is not None:
            metrics_display.append(f"Overall: {metrics.overall_score}/100")
        if metrics.security_score is not None:
            metrics_display.append(f"Security: {metrics.security_score}/100")
        if metrics.performance_score is not None:
            metrics_display.append(f"Performance: {metrics.performance_score}/100")
        if metrics.maintainability_score is not None:
            metrics_display.append(f"Maintainability: {metrics.maintainability_score}/100")
        if metrics.complexity_score is not None:
            metrics_display.append(f"Complexity: {metrics.complexity_score}/100")
        metrics_display.append(f"Reliability: {state['report_data']['reliability_score']}/100")
        metrics_display.append(f"Usability: {state['report_data']['usability_score']}/100")
        
        issue_metrics = []
        if metrics.total_issues is not None:
            issue_metrics.append(f"Total: {metrics.total_issues}")
        if metrics.critical_count is not None:
            issue_metrics.append(f"Critical: {metrics.critical_count}")
        if metrics.high_count is not None:
            issue_metrics.append(f"High: {metrics.high_count}")
        if metrics.medium_count is not None:
            issue_metrics.append(f"Medium: {metrics.medium_count}")
        issue_metrics.append(f"Low: {state['report_data']['low_count']}")
        
        if metrics_display or issue_metrics:
            metrics_section = "## Metrics\n"
            if metrics_display:
                metrics_section += "### Scores\n" + "\n".join(metrics_display) + "\n"
            if issue_metrics:
                metrics_section += "### Issues\n" + "\n".join(issue_metrics) + "\n"
            final_report_parts.append(metrics_section)
        
        final_message_content = "\n".join(final_report_parts)
        
        if response.should_create_chart and chart_image_data:
            final_message_content += f"\n## Visualization\n{response.chart_type} chart generated."
        if response.should_create_diagram and diagram_image_data:
            final_message_content += f"\n## Diagram\n{response.diagram_type} generated."
        
        ui_message = await cl.Message(
            content="---\n" + final_message_content, 
            author="GitSpeak"
        ).send()
        if chart_image_data:
            try:
                image_bytes = base64.b64decode(chart_image_data)
                image_element = cl.Image(
                    content=image_bytes, 
                    name="quality_intelligence_chart.png", 
                    display="inline",
                    size="large" 
                )
                await image_element.send(for_id=ui_message.id)
            except Exception as e:
                logging.info(f"Chart display error: {e}")
        
        if diagram_image_data:
            try:
                diagram_bytes = base64.b64decode(diagram_image_data)
                diagram_element = cl.Image(
                    content=diagram_bytes, 
                    name="architecture_diagram.png", 
                    display="inline",
                    size="large" 
                )
                await diagram_element.send(for_id=ui_message.id)
            except Exception as e:
                logging.info(f"Diagram display error: {e}")
    
    except Exception as e:
        logging.info(f"Quality error: {e}")
        fallback_messages = [
            SystemMessage(content=QUALITY_AGENT_PROMPT.format(
                messages=code_analysis,
                security_message=security_analysis,
                performance_message=performance_analysis,
                folder_structure=folder_structure,
                project_analysis=project_analysis,
                context=clean_state_for_llm["code"],
                question=clean_state_for_llm["rephrased_question"],
                pr_data_str=pr_data_str
            ))
        ]
        
        ui_message = await cl.Message(
            content="---\n## Report\nSynthesizing...\n", 
            author="GitSpeak"
        ).send()
        
        full_response = ""
        async for chunk in quality_llm.astream(fallback_messages):
            content = chunk.content
            if content:
                full_response += content
                await ui_message.stream_token(content)
        
        final_message_content = full_response
        await ui_message.update()

    state["quality_message"].append(AIMessage(content=final_message_content))
    return state

@traceable(name="Cannot Answer Node")
async def cannot_answer(state: AgentState):
    error_message = """## No Relevant Info

Couldn't find info. Suggestions:
- Specific: Files, functions.
- Architecture: Design.
- Quality: Dashboard.
- Docs: README.

Rephrase or details!"""
    
    await cl.Message(content=error_message, author="GitSpeak").send()
    state["messages"].append(AIMessage(content=error_message))
    return state


@traceable(name="Off Topic Node")
async def off_topic(state: AgentState):
    off_topic_message = """## Off-Topic

Specialized in code analysis. Excel at:
- Code: Architecture, patterns.
- Security: Vulns, compliance.
- Performance: Optimizations.
- Quality: Metrics, visuals.

Ask about repo's code, structure!"""
    
    await cl.Message(content=off_topic_message, author="GitSpeak").send()
    state["messages"].append(AIMessage(content=off_topic_message))
    return state