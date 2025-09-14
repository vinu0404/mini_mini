import base64
import logging
import aiosqlite
from datetime import datetime
import chainlit as cl
from langchain_core.messages import HumanMessage, AIMessage

from config import (
    CHECKPOINT_DB, 
    GLOBAL_FOLDER_STRUCTURES, 
    MAX_CONVERSATION_HISTORY_CHARS,
    USER_INPUT_TIMEOUT
)
from models import AgentState
from workflow import create_workflow
from repository_manager import setup_new_repository, setup_repository_session
from utils import find_existing_repositories, load_session_data, save_session_data, load_retriever_data
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


@cl.on_chat_start
async def start():
    """Initialize chat session with repository selection"""
    existing_repos = find_existing_repositories()
    
    if existing_repos:
        repo_options = []
        for i, repo in enumerate(existing_repos):
            project_type = repo.get('project_type', 'Unknown')
            languages = repo.get('languages', 'Unknown')
            repo_options.append(f"**{i+1}.** `{repo['repo_name']}` ({project_type} - {languages})\n   📍 {repo['repo_url']}\n   Created: {repo['created_at']}")
        
        repo_list = "\n".join(repo_options)
        welcome_msg = f"""# GitSpeak

Available Repos:

{repo_list}

Choose number or new URL."""
        await cl.Message(content=welcome_msg, author="GitSpeak").send()
        
        while True:
            user_input = await cl.AskUserMessage(
                content="**Choice:**", 
                author="GitSpeak", 
                timeout=USER_INPUT_TIMEOUT
            ).send()
            
            if not user_input: 
                continue
            choice = user_input.get("output", "").strip()
            
            if choice.isdigit():
                repo_index = int(choice) - 1
                if 0 <= repo_index < len(existing_repos):
                    selected_repo = existing_repos[repo_index]
                    repo_hash = selected_repo['repo_hash']
                    retriever, repo_name, repo_url, folder_structure, project_analysis = load_retriever_data(repo_hash)
                    if retriever:
                        GLOBAL_FOLDER_STRUCTURES[repo_hash] = {
                            'structure': folder_structure or "",
                            'analysis': project_analysis or {}
                        }
                        await setup_repository_session(retriever, repo_name, repo_url, repo_hash)
                        return
                else:
                    await cl.Message(
                        content=f"Invalid. Choose 1-{len(existing_repos)}.", 
                        author="GitSpeak"
                    ).send()
            elif "github.com" in choice or "gitlab.com" in choice:
                await setup_new_repository(choice)
                return
            else:
                await cl.Message(
                    content="Select Repo by number or give url :", 
                    author="GitSpeak"
                ).send()
    else:
        await cl.Message(
            content="""# GitSpeak

Paste GitHub URL to start.""", 
            author="GitSpeak"
        ).send()
        
        while True:
            user_input = await cl.AskUserMessage(
                content="**URL:**", 
                author="GitSpeak", 
                timeout=USER_INPUT_TIMEOUT
            ).send()
            
            if user_input and ("github.com" in user_input.get("output", "") or "gitlab.com" in user_input.get("output", "")):
                await setup_new_repository(user_input.get("output", "").strip())
                return
            else:
                await cl.Message(
                    content="Valid URL. Ex: https://github.com/user/repo", 
                    author="GitSpeak"
                ).send()


@cl.on_message
async def main(message: cl.Message):
    """Main message handler for processing user queries"""
    repo_url = cl.user_session.get("repo_url")
    repo_name = cl.user_session.get("repo_name")
    repo_hash = cl.user_session.get("repo_hash")
    
    if not repo_url or not repo_name or not repo_hash:
        await cl.Message(
            content="No repo selected.",
            author="GitSpeak"
        ).send()
        return
    
    conversation_history = cl.user_session.get("conversation_history", [])
    
    async with aiosqlite.connect(CHECKPOINT_DB) as conn:
        checkpointer = AsyncSqliteSaver(conn)
        workflow = await create_workflow()
        thread_config = {
            "configurable": {
                "thread_id": f"{repo_hash}_{datetime.now().isoformat()}"
            }, 
            "checkpointer": checkpointer
        }
        
        history_str = "\n".join([
            f"{'User' if isinstance(msg, HumanMessage) else 'AI'}: {msg.content[:500]}" 
            for msg in conversation_history
        ])
        
        initial_state = {
            "messages": [],
            "on_topic": "",
            "rephrased_question": "",
            "proceed_to_generate": "",
            "rephrase_count": 0,
            "question": HumanMessage(content=message.content),
            "code": [],
            "repo_hash": repo_hash,
            "repo_url": repo_url,
            "security_message": [],
            "performance_message": [],
            "quality_message": [],
            "chart_data": None,
            "report_data": None,
            "chart_image_base64": None,
            "diagram_data": None,
            "diagram_image_base64": None,
            "conversation_history": conversation_history,
            "folder_structure": GLOBAL_FOLDER_STRUCTURES.get(repo_hash, {}).get('structure', ''),
            "project_analysis": GLOBAL_FOLDER_STRUCTURES.get(repo_hash, {}).get('analysis', {}),
            "goto_security_agent": False,
            "goto_performance_agent": False,
            "goto_quality_agent": False,
            "agent_selection_reasoning": "",
            "agent_priority_order": [],
            "pr_data": None
        }
        
        async for step_result in workflow.astream(initial_state, config=thread_config):
            for key, value in step_result.items():
                if key in ["messages", "security_message", "performance_message", "quality_message"]:
                    for msg in value:
                        if isinstance(msg, AIMessage) and msg.content:
                            await cl.Message(content=msg.content, author="GitSpeak").send()
                if key == "chart_image_base64" and value:
                    await cl.Image(name="Chart", content=base64.b64decode(value), display="inline").send()
                if key == "diagram_image_base64" and value:
                    await cl.Image(name="Diagram", content=base64.b64decode(value), display="inline").send()
        
        final_state = await workflow.aget_state(thread_config)
        
        full_ai_content = ""
        
        for msg in final_state.values.get("messages", []):
            if isinstance(msg, AIMessage) and msg.content:
                full_ai_content += msg.content[:2000] + "\n\n"
        
        if final_state.values.get("goto_security_agent", False):
            for msg in final_state.values.get("security_message", []):
                if isinstance(msg, AIMessage) and msg.content:
                    full_ai_content += "---\n" + msg.content[:2000] + "\n\n"
        
        if final_state.values.get("goto_performance_agent", False):
            for msg in final_state.values.get("performance_message", []):
                if isinstance(msg, AIMessage) and msg.content:
                    full_ai_content += "---\n" + msg.content[:2000] + "\n\n"
        
        if final_state.values.get("goto_quality_agent", False):
            for msg in final_state.values.get("quality_message", []):
                if isinstance(msg, AIMessage) and msg.content:
                    full_ai_content += "---\n" + msg.content[:2000] + "\n\n"
        
        if not full_ai_content.strip():
            for msg in final_state.values.get("messages", []):
                if isinstance(msg, AIMessage) and msg.content:
                    full_ai_content += msg.content[:2000] + "\n\n"
        
        full_ai_content = full_ai_content.strip()
        conversation_history.extend([
            HumanMessage(content=message.content),
            AIMessage(content=full_ai_content)
        ])
        
        cl.user_session.set("conversation_history", conversation_history)
        
        if repo_hash:
            existing_session_data = load_session_data(repo_hash)
            analysis_count = existing_session_data.get("analysis_count", 0) + 1
            
            session_data = {
                "repo_name": repo_name,
                "repo_url": repo_url,
                "conversation_history": conversation_history,
                "last_workflow_state": {
                    "thread_id": thread_config["configurable"]["thread_id"], 
                    "timestamp": datetime.now().isoformat(),
                    "query": message.content[:100]
                },
                "analysis_count": analysis_count
            }
            save_session_data(repo_hash, session_data)
            logging.info(f"Saved for {repo_hash}, analyses: {analysis_count}")


if __name__ == "__main__":
    cl.run()