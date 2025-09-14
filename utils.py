import os
import json
import hashlib
import zipfile
import io
import aiohttp
from pathlib import Path
from typing import Dict, Any, List,Optional,Tuple
from datetime import datetime
from urllib.parse import urlparse

from config import (
    PERSISTENT_DIR,
    ANALYZABLE_EXTENSIONS,
    IMPORTANT_FILES,
    SKIP_DIRECTORIES,
    MAX_FOLDER_DEPTH,
    GITHUB_TOKEN,
    MAX_FILES_TO_PROCESS
)


def extract_code_units_from_directory(directory_path: Path) -> List[Tuple[str, Dict[str, Any]]]:
    """Extract code units from all files in a directory"""
    from code_extractor import extract_code_units
    
    all_units = []
    file_count = 0
    
    for root, _, files in os.walk(directory_path):
        if file_count > MAX_FILES_TO_PROCESS:
            break
            
        for file in files:
            if file_count > MAX_FILES_TO_PROCESS:
                break
                
            file_path = os.path.join(root, file)
            
            # Skip hidden files and directories
            if any(part.startswith('.') for part in Path(file_path).parts):
                continue
                
            # Skip files in excluded directories
            if any(skip_dir in file_path for skip_dir in SKIP_DIRECTORIES):
                continue
                
            try:
                units = extract_code_units(file_path)
                if units:
                    all_units.extend(units)
                    file_count += 1
            except Exception as e:
                import logging
                logging.warning(f"Error processing {file_path}: {e}")
                continue
    
    return all_units


def validate_repo_url(repo_url: str) -> bool:
    """Validate if the repository URL is supported"""
    supported_domains = ['github.com', 'gitlab.com']
    return any(domain in repo_url.lower() for domain in supported_domains)


def format_file_size(size_bytes: int) -> str:
    """Convert file size to human readable format"""
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f}{size_names[i]}"


def truncate_text(text: str, max_length: int = 1000) -> str:
    """Truncate text to specified length with ellipsis"""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def get_file_language(file_path: str) -> str:
    """Determine programming language from file extension"""
    extension_map = {
        '.py': 'Python',
        '.js': 'JavaScript',
        '.jsx': 'JavaScript (React)',
        '.ts': 'TypeScript',
        '.tsx': 'TypeScript (React)',
        '.java': 'Java',
        '.cpp': 'C++',
        '.c': 'C',
        '.go': 'Go',
        '.rs': 'Rust',
        '.php': 'PHP',
        '.rb': 'Ruby',
        '.swift': 'Swift',
        '.kt': 'Kotlin',
        '.scala': 'Scala',
        '.sh': 'Shell',
        '.bash': 'Bash',
        '.ps1': 'PowerShell',
        '.r': 'R',
        '.R': 'R',
        '.yaml': 'YAML',
        '.yml': 'YAML',
        '.json': 'JSON',
        '.xml': 'XML',
        '.html': 'HTML',
        '.css': 'CSS',
        '.scss': 'SCSS',
        '.sql': 'SQL',
        '.dockerfile': 'Dockerfile',
        '.md': 'Markdown'
    }
    
    ext = Path(file_path).suffix.lower()
    return extension_map.get(ext, 'Unknown')


def clean_repository_name(repo_url: str) -> str:
    """Extract clean repository name from URL"""
    # Remove .git suffix and extract repo name
    repo_name = repo_url.split("/")[-1]
    if repo_name.endswith('.git'):
        repo_name = repo_name[:-4]
    
    # Clean any special characters
    import re
    repo_name = re.sub(r'[^\w\-_]', '', repo_name)
    
    return repo_name


def estimate_processing_time(file_count: int, code_units: int) -> str:
    """Estimate processing time based on file count and code units"""
    # Rough estimates based on typical processing speeds
    base_time = max(10, file_count * 0.5)  # 0.5 seconds per file minimum
    vector_time = max(5, code_units * 0.1)  # 0.1 seconds per code unit
    
    total_seconds = base_time + vector_time
    
    if total_seconds < 60:
        return f"~{int(total_seconds)} seconds"
    elif total_seconds < 3600:
        minutes = int(total_seconds / 60)
        return f"~{minutes} minute{'s' if minutes > 1 else ''}"
    else:
        hours = int(total_seconds / 3600)
        return f"~{hours} hour{'s' if hours > 1 else ''}"


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage"""
    import re
    # Remove or replace unsafe characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Limit length
    if len(filename) > 255:
        filename = filename[:252] + "..."
    return filename


def get_repo_stats(repo_path: Path) -> Dict[str, Any]:
    """Get comprehensive repository statistics"""
    stats = {
        'total_files': 0,
        'total_directories': 0,
        'languages': {},
        'file_sizes': {},
        'largest_files': [],
        'total_size': 0
    }
    
    try:
        for root, dirs, files in os.walk(repo_path):
            # Filter out hidden and excluded directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in SKIP_DIRECTORIES]
            
            stats['total_directories'] += len(dirs)
            
            for file in files:
                if file.startswith('.'):
                    continue
                    
                file_path = Path(root) / file
                try:
                    file_size = file_path.stat().st_size
                    stats['total_files'] += 1
                    stats['total_size'] += file_size
                    
                    # Track language statistics
                    language = get_file_language(str(file_path))
                    stats['languages'][language] = stats['languages'].get(language, 0) + 1
                    
                    # Track file size distribution
                    size_category = 'small' if file_size < 10240 else 'medium' if file_size < 102400 else 'large'
                    stats['file_sizes'][size_category] = stats['file_sizes'].get(size_category, 0) + 1
                    
                    # Track largest files
                    stats['largest_files'].append({
                        'path': str(file_path.relative_to(repo_path)),
                        'size': file_size,
                        'size_formatted': format_file_size(file_size)
                    })
                    
                except (OSError, PermissionError):
                    continue
        
        # Sort and limit largest files
        stats['largest_files'] = sorted(stats['largest_files'], key=lambda x: x['size'], reverse=True)[:10]
        stats['total_size_formatted'] = format_file_size(stats['total_size'])
        
    except Exception as e:
        import logging
        logging.warning(f"Error getting repo stats: {e}")
    
    return stats

def analyze_project_structure(root_folder: Path) -> Dict[str, Any]:
    """Analyze project structure and identify key characteristics"""
    analysis = {
        "project_type": "Unknown",
        "languages": [],
        "frameworks": [],
        "config_files": [],
        "documentation": [],
        "test_directories": [],
        "build_tools": [],
        "total_files": 0,
        "total_directories": 0,
        "dependencies": [], 
        "entry_points": []   
    }
    
    try:
        for root, dirs, files in os.walk(root_folder):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in SKIP_DIRECTORIES]
            
            analysis["total_directories"] += len(dirs)
            analysis["total_files"] += len(files)
            
            for file in files:
                file_lower = file.lower()
                file_path = Path(root) / file
                if file.endswith('.py'):
                    analysis["languages"].append("Python")
                elif file.endswith(('.js', '.jsx')):
                    analysis["languages"].append("JavaScript")
                elif file.endswith(('.ts', '.tsx')):
                    analysis["languages"].append("TypeScript")
                elif file.endswith('.java'):
                    analysis["languages"].append("Java")
                elif file.endswith(('.cpp', '.cc', '.cxx')):
                    analysis["languages"].append("C++")
                elif file.endswith('.c'):
                    analysis["languages"].append("C")
                elif file.endswith('.go'):
                    analysis["languages"].append("Go")
                elif file.endswith('.rs'):
                    analysis["languages"].append("Rust")
                elif file.endswith('.php'):
                    analysis["languages"].append("PHP")
                elif file.endswith('.rb'):
                    analysis["languages"].append("Ruby")
                if file_lower == 'package.json':
                    analysis["frameworks"].append("Node.js")
                    analysis["config_files"].append(file)
                elif file_lower == 'requirements.txt':
                    analysis["config_files"].append(file)
                elif file_lower == 'setup.py':
                    analysis["config_files"].append(file)
                elif file_lower == 'pom.xml':
                    analysis["frameworks"].append("Maven")
                    analysis["config_files"].append(file)
                elif file_lower == 'build.gradle':
                    analysis["frameworks"].append("Gradle")
                    analysis["config_files"].append(file)
                elif file_lower == 'dockerfile':
                    analysis["frameworks"].append("Docker")
                    analysis["config_files"].append(file)
                elif file_lower in ['makefile', 'makefile.am']:
                    analysis["build_tools"].append("Make")
                elif file_lower.startswith('readme'):
                    analysis["documentation"].append(file)
                elif 'test' in file_lower or 'spec' in file_lower:
                    analysis["test_directories"].append(str(file_path))
                if file_lower in ['requirements.txt', 'package.json', 'pom.xml', 'build.gradle']:
                    analysis["dependencies"].append(file)
                if file_lower in ['main.py', 'app.py', 'index.js', 'server.py']:
                    analysis["entry_points"].append(str(file_path))
            
            for directory in dirs:
                if 'test' in directory.lower() or directory.lower() in ['tests', 'spec', 'specs']:
                    analysis["test_directories"].append(str(Path(root) / directory))
        
        analysis["languages"] = list(set(analysis["languages"]))
        analysis["frameworks"] = list(set(analysis["frameworks"]))
        
        if "Python" in analysis["languages"]:
            if any("django" in f.lower() for f in analysis["config_files"]):
                analysis["project_type"] = "Django Web Application"
            elif any("flask" in f.lower() for f in analysis["config_files"]):
                analysis["project_type"] = "Flask Web Application"
            elif "requirements.txt" in analysis["config_files"] or "setup.py" in analysis["config_files"]:
                analysis["project_type"] = "Python Application"
        elif "JavaScript" in analysis["languages"] or "TypeScript" in analysis["languages"]:
            if "Node.js" in analysis["frameworks"]:
                analysis["project_type"] = "Node.js Application"
            else:
                analysis["project_type"] = "Web Application"
        elif "Java" in analysis["languages"]:
            analysis["project_type"] = "Java Application"
        
        analysis["has_tests"] = len(analysis["test_directories"]) > 0
        analysis["doc_coverage"] = len(analysis["documentation"]) > 0
        
    except Exception as e:
        import logging
        logging.info(f"Error analyzing project structure: {e}")
    
    return analysis


async def clone_repo_via_api(repo_url: str, github_token: str) -> Path:
    """Clone repository using GitHub API (supports private repos)"""
    try:
        parsed = urlparse(repo_url)
        path_parts = parsed.path.strip('/').split('/')
        owner, repo = path_parts[0], path_parts[1].replace('.git', '')
        
        headers = {
            "Authorization": f"token {github_token}",
            "Accept": "application/vnd.github.v3+json"
        }
        zipball_url = f"https://api.github.com/repos/{owner}/{repo}/zipball"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(zipball_url, headers=headers) as response:
                if response.status != 200:
                    raise Exception(f"API Error: {response.status}")
                zip_content = await response.read()
                repos_dir = Path("repos")
                repos_dir.mkdir(exist_ok=True)
                local_path = repos_dir / repo
                
                with zipfile.ZipFile(io.BytesIO(zip_content)) as zip_file:
                    zip_file.extractall(local_path)
                
                # Return Path object instead of string
                return local_path
                
    except Exception as e:
        raise Exception(f"GitHub API clone failed: {str(e)}")


async def get_pr_tool(repo_url: str, pr_number: Optional[int] = None) -> Dict[str, Any]:
    """Fetch PR information using GitHub API, including diffs, commits, and CI/CD check runs.
    If pr_number is None, fetch the latest open PR's details."""
    
    import logging
    from typing import Optional
    from config import MAX_OPEN_PRS
    
    logging.info(f"get_pr_tool called with repo_url={repo_url}, pr_number={pr_number}")
    
    if not GITHUB_TOKEN:
        logging.error("No GITHUB_TOKEN available")
        return {"error": "Please provide GITHUB_TOKEN environment variable or proper access token for accessing pull request information."}
    
    try:
        logging.info(f"Fetching PR data for {repo_url} PR #{pr_number if pr_number else 'latest'}")
        parsed = urlparse(repo_url)

        path = parsed.path.strip('/').split('/')
        if len(path) < 2:
            logging.error(f"Invalid repository URL format: {repo_url}")
            raise ValueError("Invalid repository URL format. Expected: https://github.com/owner/repo")
        owner, repo = path[0], path[1].replace('.git', '')
        
        logging.info(f"Parsed owner={owner}, repo={repo}")
        
        base_url = f"https://api.github.com/repos/{owner}/{repo}"
        headers = {
            "Authorization": f"token {GITHUB_TOKEN}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        logging.info(f"Using base URL: {base_url}")
        
        async with aiohttp.ClientSession() as session:
            if pr_number is None:
                # Fetch list of open PRs
                logging.info("Fetching open PRs...")
                async with session.get(f"{base_url}/pulls", headers=headers, params={"state": "open", "per_page": MAX_OPEN_PRS}) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        logging.error(f"Error fetching PR list: {resp.status} - {error_text}")
                        raise ValueError(f"Error fetching PR list: {resp.status} - {error_text}")
                    
                    prs = await resp.json()
                    logging.info(f"Fetched {len(prs)} open PRs")
                    
                    if not prs:
                        logging.info("No open PRs found")
                        return {"error": "No open pull requests found in the repository."}
                    
                    # Sort by created_at descending to get the latest
                    prs_sorted = sorted(prs, key=lambda x: x['created_at'], reverse=True)
                    latest_pr = prs_sorted[0]
                    pr_number = latest_pr['number']
                    logging.info(f"Selected latest PR #{pr_number}: {latest_pr.get('title', 'No title')}")

            # Now fetch specific PR details (either provided or latest)
            logging.info(f"Fetching details for PR #{pr_number}")
            async with session.get(f"{base_url}/pulls/{pr_number}", headers=headers) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logging.error(f"Error fetching PR {pr_number}: {resp.status} - {error_text}")
                    raise ValueError(f"Error fetching PR {pr_number}: {resp.status} - {error_text}")
                pr_info = await resp.json()
                logging.info(f"Fetched PR info for #{pr_number}")
            
            # Fetch diff
            logging.info("Fetching diff...")
            diff_headers = headers.copy()
            diff_headers["Accept"] = "application/vnd.github.diff"
            async with session.get(f"{base_url}/pulls/{pr_number}", headers=diff_headers) as resp:
                if resp.status != 200:
                    logging.error(f"Error fetching PR diff: {resp.status}")
                    raise ValueError(f"Error fetching PR diff: {resp.status}")
                diff = await resp.text()
                logging.info(f"Fetched PR diff of length {len(diff)}")
            
            # Fetch commits
            logging.info("Fetching commits...")
            async with session.get(f"{base_url}/pulls/{pr_number}/commits", headers=headers) as resp:
                if resp.status != 200:
                    logging.error(f"Error fetching PR commits: {resp.status}")
                    raise ValueError(f"Error fetching PR commits: {resp.status}")
                commits = await resp.json()
                logging.info(f"Fetched {len(commits)} commits for PR")
            
            # Fetch CI/CD check runs (for the head commit)
            logging.info("Fetching check runs...")
            head_sha = pr_info['head']['sha']
            async with session.get(f"{base_url}/commits/{head_sha}/check-runs", headers=headers) as resp:
                if resp.status != 200:
                    logging.warning(f"Error fetching check runs: {resp.status}")
                    checks = {"check_runs": []}
                else:
                    checks = await resp.json()
                    logging.info(f"Fetched {len(checks.get('check_runs', []))} check runs for commit {head_sha}")
            
            # Fetch workflows if available (CI/CD pipelines)
            logging.info("Fetching workflows...")
            async with session.get(f"{base_url}/actions/workflows", headers=headers) as resp:
                workflows = await resp.json() if resp.status == 200 else {}
                if resp.status == 200:
                    logging.info(f"Fetched {len(workflows.get('workflows', []))} workflows")
                else:
                    logging.warning(f"Could not fetch workflows: {resp.status}")
            
            result = {
                "pr_info": pr_info,
                "diff": diff,
                "commits": commits,
                "checks": checks,
                "workflows": workflows,
                "error": None
            }
            
            logging.info(f"Successfully prepared PR data with diff length: {len(diff)}")
            return result
            
    except Exception as e:
        logging.error(f"Exception in get_pr_tool: {str(e)}", exc_info=True)
        return {"error": f"Failed to fetch PR data: {str(e)}"}


def get_repo_hash(repo_url: str) -> str:
    """Generate unique hash for repository URL"""
    return hashlib.md5(repo_url.encode()).hexdigest()


def save_session_data(session_id: str, data: Dict[str, Any]):
    """Save session data to persistent storage with enhanced serialization"""
    from langchain_core.messages import HumanMessage, AIMessage
    
    session_file = PERSISTENT_DIR / f"session_{session_id}.json"
    try:
        serializable_data = data.copy()
        if 'conversation_history' in serializable_data:
            history = []
            for msg in serializable_data['conversation_history']:
                if hasattr(msg, 'content'):
                    msg_type = 'human' if isinstance(msg, HumanMessage) else 'ai'
                    history.append({'type': msg_type, 'content': msg.content})
            serializable_data['conversation_history'] = history
        
        serializable_data['last_updated'] = datetime.now().isoformat()
        
        with open(session_file, 'w') as f:
            json.dump(serializable_data, f, indent=2)
    except Exception as e:
        import logging
        logging.info(f"Error saving session data: {e}")


def load_session_data(session_id: str) -> Dict[str, Any]:
    """Load session data from persistent storage with enhanced deserialization"""
    from langchain_core.messages import HumanMessage, AIMessage
    
    session_file = PERSISTENT_DIR / f"session_{session_id}.json"
    try:
        if session_file.exists():
            with open(session_file, 'r') as f:
                data = json.load(f)
                
                if 'conversation_history' in data:
                    history = []
                    for msg_data in data['conversation_history']:
                        if msg_data['type'] == 'human':
                            history.append(HumanMessage(content=msg_data['content']))
                        else:
                            history.append(AIMessage(content=msg_data['content']))
                    data['conversation_history'] = history
                return data
    except Exception as e:
        import logging
        logging.info(f"Error loading session data: {e}")
    return {}


def save_retriever_data(repo_hash: str, vector_store, repo_name: str, repo_url: str, folder_structure: str, project_analysis: Dict[str, Any]):
    """Save vector store, metadata, and folder structure with enhanced metadata"""
    try:
        retriever_dir = PERSISTENT_DIR / f"retriever_{repo_hash}"
        retriever_dir.mkdir(exist_ok=True)
        vector_store.save_local(str(retriever_dir / "faiss_store"))
        metadata = {
            "repo_name": repo_name,
            "repo_url": repo_url,
            "created_at": datetime.now().isoformat(),
            "folder_structure": folder_structure,
            "project_analysis": project_analysis,
            "version": "2.0" 
        }
        with open(retriever_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
            
    except Exception as e:
        import logging
        logging.info(f"Error saving retriever: {e}")


def load_retriever_data(repo_hash: str):
    """Load vector store, metadata, and folder structure with version check"""
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS
    from config import OPENAI_API_KEY
    
    try:
        retriever_dir = PERSISTENT_DIR / f"retriever_{repo_hash}"
        
        if not retriever_dir.exists():
            return None, None, None, None, None
        metadata_file = retriever_dir / "metadata.json"
        if not metadata_file.exists():
            return None, None, None, None, None
            
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        if metadata.get("version", "1.0") != "2.0":
            import logging
            logging.info("Warning: Metadata version mismatch, may need rebuild")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=OPENAI_API_KEY)
        vector_store = FAISS.load_local(
            str(retriever_dir / "faiss_store"), 
            embeddings,
            allow_dangerous_deserialization=True
        )
        retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 10})
        
        return (retriever, metadata["repo_name"], metadata["repo_url"], 
                metadata.get("folder_structure", ""), metadata.get("project_analysis", {}))
        
    except Exception as e:
        import logging
        logging.info(f"Error loading retriever: {e}")
        return None, None, None, None, None


def find_existing_repositories() -> List[Dict[str, str]]:
    """Find all existing repository data with enhanced details"""
    existing_repos = []
    if PERSISTENT_DIR.exists():
        for retriever_dir in PERSISTENT_DIR.glob("retriever_*"):
            try:
                metadata_file = retriever_dir / "metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    repo_hash = retriever_dir.name.replace('retriever_', '')
                    project_analysis = metadata.get('project_analysis', {})
                    project_type = project_analysis.get('project_type', 'Unknown')
                    languages = project_analysis.get('languages', [])
                    existing_repos.append({
                        'repo_name': metadata['repo_name'],
                        'repo_url': metadata['repo_url'],
                        'repo_hash': repo_hash,
                        'created_at': metadata.get('created_at', 'Unknown'),
                        'project_type': project_type,
                        'languages': ', '.join(languages[:3]) if languages else 'Unknown'
                    })
            except Exception:
                continue
    return existing_repos


def generate_folder_structure(root_folder: Path, indent: str = "", max_depth: int = MAX_FOLDER_DEPTH, current_depth: int = 0) -> str:
    """Generate a comprehensive folder structure representation"""
    if current_depth > max_depth:
        return f"{indent}... (max depth reached)\n"
    
    structure = ""
    try:
        if isinstance(root_folder, str):
            root_folder = Path(root_folder)
        
        items = sorted(os.listdir(root_folder))
        folders = []
        files = []
        
        for item in items:
            if item.startswith('.') or item in SKIP_DIRECTORIES:
                continue
            item_path = root_folder / item
            if item_path.is_dir():
                folders.append(item)
            else:
                files.append(item)
        
        # Add folders first
        for folder in folders[:20]: 
            structure += f"{indent}📁 {folder}/\n"
            folder_path = root_folder / folder
            structure += generate_folder_structure(folder_path, indent + "    ", max_depth, current_depth + 1)
        
        # Add files
        important_files = []
        other_files = []
        
        for file in files:
            if file.lower() in IMPORTANT_FILES:
                important_files.append(file)
            else:
                other_files.append(file)
        
        # Show important files first
        for file in important_files:
            structure += f"{indent}📄 {file}\n"
        
        # Show limited number of other files
        for file in other_files[:15]:
            structure += f"{indent}📄 {file}\n"
        
        if len(other_files) > 15:
            structure += f"{indent}... and {len(other_files) - 15} more files\n"
            
    except PermissionError:
        structure += f"{indent} Permission denied\n"
    except Exception as e:
        structure += f"{indent} Error: {str(e)}\n"

    return structure
