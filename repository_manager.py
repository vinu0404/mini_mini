import os
import git
import logging
from pathlib import Path
from typing import Tuple
import chainlit as cl
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from config import (
    OPENAI_API_KEY,
    GLOBAL_RETRIEVERS,
    GLOBAL_FOLDER_STRUCTURES,
    MAX_FILES_TO_PROCESS,
    GITHUB_TOKEN
)
from utils import (
    get_repo_hash, 
    clone_repo_via_api,
    generate_folder_structure,
    analyze_project_structure,
    save_retriever_data,
    load_retriever_data
)
from code_extractor import extract_code_units


async def build_retriever_from_repo(repo_url: str) -> tuple:
    """Build retriever with token-aware cloning strategy"""
    try:
        repo_name = repo_url.split("/")[-1].replace(".git", "")
        repo_hash = get_repo_hash(repo_url)
        
        # Check if already exists in global cache
        if repo_hash in GLOBAL_RETRIEVERS:
            cached_data = GLOBAL_RETRIEVERS[repo_hash]
            return cached_data['retriever'], cached_data['repo_name'], repo_hash
        
        # Try to load from disk
        existing_retriever, existing_name, existing_url, folder_structure, project_analysis = load_retriever_data(repo_hash)
        if existing_retriever:
            # Cache in memory including folder structure
            GLOBAL_RETRIEVERS[repo_hash] = {
                'retriever': existing_retriever,
                'repo_name': existing_name,
                'repo_url': existing_url
            }
            GLOBAL_FOLDER_STRUCTURES[repo_hash] = {
                'structure': folder_structure,
                'analysis': project_analysis
            }
            await cl.Message(
                content=f"**Found cached data for `{existing_name}`!**",
                author="GitSpeak"
            ).send()
            return existing_retriever, existing_name, repo_hash
        
        # Build new retriever
        await cl.Message(
            content=f"**Processing new repository: `{repo_name}`**",
            author="GitSpeak"
        ).send()
        
        # Token-aware cloning strategy
        if GITHUB_TOKEN:
            await cl.Message(
                content=f"**Using GitHub API (Private access enabled)**",
                author="GitSpeak"
            ).send()
            local_path = await clone_repo_via_api(repo_url, GITHUB_TOKEN)
        else:
            await cl.Message(
                content=f"**Using Git clone (Public repos only)**",
                author="GitSpeak"
            ).send()
            repos_dir = Path("repos")
            repos_dir.mkdir(exist_ok=True)
            local_path = repos_dir / repo_name
            if not local_path.exists():
                git.Repo.clone_from(repo_url, str(local_path))
            # Ensure local_path is a Path object
            local_path = Path(local_path)

        # Generate folder structure and project analysis
        await cl.Message(
            content="**Analyzing project structure and folders...**",
            author="GitSpeak"
        ).send()
        
        # Now local_path is guaranteed to be a Path object
        folder_structure = generate_folder_structure(local_path)
        project_analysis = analyze_project_structure(local_path)
        
        # Store in global variable
        GLOBAL_FOLDER_STRUCTURES[repo_hash] = {
            'structure': folder_structure,
            'analysis': project_analysis
        }
        
        # Extract code units
        await cl.Message(
            content="**Extracting and analyzing code structure...**",
            author="GitSpeak"
        ).send()
        
        code_units = []
        file_count = 0
        
        for root, _, files in os.walk(local_path):
            if file_count > MAX_FILES_TO_PROCESS: 
                break
            for file in files:
                if file_count > MAX_FILES_TO_PROCESS:
                    break
                file_path = os.path.join(root, file)
                units = extract_code_units(file_path)
                if units:
                    code_units.extend(units)
                    file_count += 1
        
        if not code_units:
            raise ValueError("No analyzable code units found in the repository")
        
        # Build vector store
        await cl.Message(
            content=f"**Building knowledge base from {len(code_units)} code units...**",
            author="GitSpeak"
        ).send()

        embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=OPENAI_API_KEY)

        chunks = [chunk for chunk, _ in code_units]
        metadata = [meta for _, meta in code_units]
        
        vector_store = FAISS.from_texts(texts=chunks, embedding=embeddings, metadatas=metadata)
        retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 10})
        save_retriever_data(repo_hash, vector_store, repo_name, repo_url, folder_structure, project_analysis)
        GLOBAL_RETRIEVERS[repo_hash] = {
            'retriever': retriever,
            'repo_name': repo_name,
            'repo_url': repo_url
        }
        
        return retriever, repo_name, repo_hash
        
    except Exception as e:
        raise Exception(f"Error building retriever: {str(e)}")


async def setup_new_repository(repo_url: str):
    """Setup a new repository for analysis"""
    try:
        if not ("github.com" in repo_url or "gitlab.com" in repo_url):
            raise ValueError("Use GitHub or GitLab.")
        
        retriever, repo_name, repo_hash = await build_retriever_from_repo(repo_url)
        await setup_repository_session(retriever, repo_name, repo_url, repo_hash)
    except git.GitCommandError as ge:
        await cl.Message(
            content=f"Clone Error: {str(ge)}\nTroubleshoot: Public? URL correct?", 
            author="GitSpeak"
        ).send()
    except Exception as e:
        await cl.Message(
            content=f"Setup Error: {str(e)}\nTroubleshoot: URL, network, size.", 
            author="GitSpeak"
        ).send()


async def setup_repository_session(retriever, repo_name: str, repo_url: str, repo_hash: str):
    """Setup the repository session with proper context"""
    from utils import load_session_data
    
    cl.user_session.set("repo_name", repo_name)
    cl.user_session.set("repo_url", repo_url)
    cl.user_session.set("repo_hash", repo_hash)
    
    session_data = load_session_data(repo_hash)
    conversation_history = session_data.get("conversation_history", [])
    cl.user_session.set("conversation_history", conversation_history)
    
    project_info = ""
    if repo_hash in GLOBAL_FOLDER_STRUCTURES:
        project_analysis = GLOBAL_FOLDER_STRUCTURES[repo_hash].get('analysis', {})
        project_type = project_analysis.get('project_type', 'Unknown')
        languages = project_analysis.get('languages', [])
        has_tests = project_analysis.get('has_tests', False)
        doc_coverage = project_analysis.get('doc_coverage', False)
        if project_type != 'Unknown':
            project_info = f" ({project_type})"
        if languages:
            project_info += f" - Languages: {', '.join(languages[:3])}"
        if has_tests:
            project_info += " - Tests"
        if doc_coverage:
            project_info += " - Docs"
    
    token_status = "Private access" if GITHUB_TOKEN else "🔓 Public only"
    pr_status = "PR analysis available" if GITHUB_TOKEN else "PR analysis unavailable"
    
    await cl.Message(
        content=f"""## {repo_name}{project_info} Ready!
URL: {repo_url}
Access: {token_status}
PR Tools: {pr_status}
Files: {project_analysis.get('total_files', 'N/A')} | Folders: {project_analysis.get('total_directories', 'N/A')}

Ask about code!""", 
        author="GitSpeak"
    ).send()