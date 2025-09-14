import logging
import os
import ast
import git
import nbformat
import requests
import base64
import json
import hashlib
import aiosqlite
from pathlib import Path
from typing import Optional, List, Dict, Any,Union, Set
from datetime import datetime, timedelta
from urllib.parse import quote
import zlib
import chainlit as cl
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import BaseModel, Field
from typing import TypedDict
from dotenv import load_dotenv
import aiohttp 
import zipfile
import io
from langsmith import traceable
load_dotenv()
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise ValueError("Please set GITHUB_TOKEN environment variable")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Please set OPENAI_API_KEY environment variable")
PERSISTENT_DIR = Path("gitspeak_persistent")
PERSISTENT_DIR.mkdir(exist_ok=True)
CHECKPOINT_DB = str(PERSISTENT_DIR / "checkpoints.sqlite")
GLOBAL_RETRIEVERS = {}
GLOBAL_SESSIONS = {}
GLOBAL_FOLDER_STRUCTURES = {} 

basic_llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY, temperature=0.1, max_tokens=4000, streaming=True)
code_analyst_llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY, temperature=0.1, max_tokens=4000, streaming=True)
security_llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY, temperature=0.2, max_tokens=4000, streaming=True)
performance_llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY, temperature=0.2, max_tokens=4000, streaming=True)
quality_llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY, temperature=0.3, max_tokens=8000, streaming=True)
agent_selection_llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY, temperature=0.1, max_tokens=2000)
present_teller_llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY, temperature=0.1, max_tokens=500, streaming=True)  

class ChartDataset(BaseModel):
    label: str = Field(description="Dataset label")
    data: List[Union[int, float]] = Field(description="Data values")
    backgroundColor: Optional[List[str]] = Field(default=None, description="Background colors")
    borderColor: Optional[List[str]] = Field(default=None, description="Border colors")
    borderWidth: Optional[int] = Field(default=1, description="Border width")
    fill: Optional[bool] = Field(default=None, description="Fill area under line")
    tension: Optional[float] = Field(default=None, description="Line tension for line charts")

class ChartData(BaseModel):
    type: str = Field(description="Chart type: 'bar', 'line', 'pie', 'doughnut', 'radar', etc.")
    labels: List[str] = Field(description="Labels for the chart")
    datasets: List[ChartDataset] = Field(description="Dataset configuration for Chart.js")
    title: str = Field(description="Chart title")
    width: int = Field(default=800, description="Chart width in pixels")
    height: int = Field(default=600, description="Chart height in pixels")

class MermaidNode(BaseModel):
    id: str = Field(description="Unique node identifier")
    label: str = Field(description="Display text for the node")
    shape: Optional[str] = Field(default="rect", description="Node shape: rect, circle, diamond, hexagon")
    style: Optional[str] = Field(default=None, description="CSS styling for node")

class MermaidEdge(BaseModel):
    from_node: str = Field(description="Source node ID")
    to_node: str = Field(description="Target node ID") 
    label: Optional[str] = Field(default="", description="Edge label text")
    arrow_type: Optional[str] = Field(default="-->", description="Arrow style: -->, -.->, ==>, -.-")

class MermaidFlowchart(BaseModel):
    direction: str = Field(default="TD", description="Flow direction: TD, LR, BT, RL")
    nodes: List[MermaidNode] = Field(description="List of flowchart nodes")
    edges: List[MermaidEdge] = Field(description="List of connections between nodes")
    title: Optional[str] = Field(default=None, description="Diagram title")

class MermaidClassMember(BaseModel):
    name: str = Field(description="Method or attribute name")
    type: str = Field(description="method, attribute, or constructor")
    visibility: Optional[str] = Field(default="public", description="public, private, protected")
    return_type: Optional[str] = Field(default=None, description="Return type for methods")

class MermaidClass(BaseModel):
    name: str = Field(description="Class name")
    attributes: List[MermaidClassMember] = Field(default=[], description="Class attributes")
    methods: List[MermaidClassMember] = Field(default=[], description="Class methods")

class MermaidRelationship(BaseModel):
    from_: str = Field(description="From class", alias="from")
    to: str = Field(description="To class")
    type: str = Field(description="Relationship type")

class MermaidClassDiagram(BaseModel):
    classes: List[MermaidClass] = Field(description="List of classes")
    relationships: List[MermaidRelationship] = Field(description="Class relationships")
    title: Optional[str] = Field(default=None, description="Diagram title")

class MermaidSequenceParticipant(BaseModel):
    id: str = Field(description="Participant identifier")
    display_name: str = Field(description="Display name for participant")

class MermaidSequenceMessage(BaseModel):
    from_participant: str = Field(description="Sender participant ID")
    to_participant: str = Field(description="Receiver participant ID")
    message: str = Field(description="Message text")
    arrow_type: Optional[str] = Field(default="->", description="Arrow type: ->, ->>, -x, -->>")
    activation: Optional[bool] = Field(default=False, description="Show activation box")

class MermaidSequenceDiagram(BaseModel):
    participants: List[MermaidSequenceParticipant] = Field(description="Sequence participants")
    messages: List[MermaidSequenceMessage] = Field(description="Messages between participants")
    title: Optional[str] = Field(default=None, description="Diagram title")

class MermaidMindmapNode(BaseModel):
    id: str = Field(description="Node identifier")
    text: str = Field(description="Node text content")
    level: int = Field(description="Hierarchy level (0=root, 1=main branch, etc.)")
    parent_id: Optional[str] = Field(default=None, description="Parent node ID")
    shape: Optional[str] = Field(default="rect", description="Node shape")

class MermaidMindmap(BaseModel):
    root_text: str = Field(description="Root node text")
    nodes: List[MermaidMindmapNode] = Field(description="Mindmap nodes in hierarchical order")
    title: Optional[str] = Field(default=None, description="Diagram title")

class MermaidDiagramData(BaseModel):
    type: str = Field(description="Diagram type: flowchart, classDiagram, sequenceDiagram, mindmap")
    flowchart: Optional[MermaidFlowchart] = Field(default=None)
    class_diagram: Optional[MermaidClassDiagram] = Field(default=None) 
    sequence_diagram: Optional[MermaidSequenceDiagram] = Field(default=None)
    mindmap: Optional[MermaidMindmap] = Field(default=None)
    title: str = Field(description="Overall diagram title")
    width: int = Field(default=1000, description="Diagram width")
    height: int = Field(default=800, description="Diagram height")

class ReportData(BaseModel):
    summary: str = Field(description="Executive summary of the analysis")
    critical_issues: List[str] = Field(description="List of critical issues found")
    recommendations: List[str] = Field(description="List of actionable recommendations")
    security_score: Optional[int] = Field(default=None, description="Security score (0-100)")
    performance_score: Optional[int] = Field(default=None, description="Performance score (0-100)")
    maintainability_score: Optional[int] = Field(default=None, description="Maintainability score (0-100)")
    complexity_score: Optional[int] = Field(default=None, description="Complexity score (0-100)")
    overall_score: Optional[int] = Field(default=None, description="Overall quality score (0-100)")
    total_issues: Optional[int] = Field(default=None, description="Total number of issues found")
    critical_count: Optional[int] = Field(default=None, description="Number of critical issues")
    high_count: Optional[int] = Field(default=None, description="Number of high priority issues")
    medium_count: Optional[int] = Field(default=None, description="Number of medium priority issues")

class QualityAnalysisOutput(BaseModel):
    should_create_chart: bool = Field(description="Whether a chart/dashboard should be created")
    chart_type: str = Field(description="Type of visualization needed")
    should_create_diagram: bool = Field(default=False, description="Whether a Mermaid diagram should be created")
    diagram_type: str = Field(default="", description="Type of Mermaid diagram: flowchart, classDiagram, sequenceDiagram, mindmap")
    report_data: ReportData = Field(description="Structured report data")
    chart_data: Optional[ChartData] = Field(description="Chart configuration if chart should be created")
    diagram_data: Optional[MermaidDiagramData] = Field(default=None, description="Mermaid diagram configuration")
    image_base64: Optional[str] = Field(default=None, description="Base64 encoded chart image")
    diagram_image_base64: Optional[str] = Field(default=None, description="Base64 encoded diagram image")

class OffTopic(BaseModel):
    answer: str = Field(description="Question is from specified topic? If yes -> 'yes' if not -> 'no'")

class BatchCodeGrading(BaseModel):
    relevance_scores: List[str] = Field(
        description="List of relevance scores for each code chunk. Each score should be 'relevant' or 'not relevant'"
    )

class AgentSelection(BaseModel):
    goto_security_agent: bool = Field(description="Whether to route to the security agent")
    goto_performance_agent: bool = Field(description="Whether to route to the performance agent")
    goto_quality_agent: bool = Field(description="Whether to route to the quality agent")
    reasoning: str = Field(description="Detailed explanation of the agent selection decision")
    priority_order: List[str] = Field(description="Ordered list of agents by priority: ['security', 'performance', 'quality']")

class PresentTellerOutput(BaseModel):
    explanation: str = Field(description="1-2 line explanation of why this agent is being invoked")

class PrDecision(BaseModel):
    call_pr: bool = Field(description="Whether to fetch PR info")
    pr_number: Optional[int] = Field(default=None, description="Specific PR number if mentioned")
    reason: str = Field(description="Reasoning for decision")

class AgentState(TypedDict):
    messages: List[BaseMessage]
    on_topic: str 
    rephrased_question: str
    proceed_to_generate: str
    rephrase_count: int
    question: HumanMessage
    code: list
    repo_hash: str
    repo_url: str 
    security_message: List[BaseMessage]
    performance_message: List[BaseMessage]
    quality_message: List[BaseMessage]
    chart_data: Optional[Dict[str, Any]]
    report_data: Optional[Dict[str, Any]]
    chart_image_base64: Optional[str]
    diagram_data: Optional[Dict[str, Any]]
    diagram_image_base64: Optional[str]
    conversation_history: List[BaseMessage]
    folder_structure: Optional[str]
    project_analysis: Optional[Dict[str, Any]]
    goto_security_agent: bool
    goto_performance_agent: bool
    goto_quality_agent: bool
    agent_selection_reasoning: str
    agent_priority_order: List[str]
    pr_data: Optional[Dict[str, Any]]

class QuickChartTool:
    BASE_URL = "https://quickchart.io/chart"
    
    @staticmethod
    def create_chart(chart_config: Dict[str, Any], width: int = 800, height: int = 600) -> str:
        """Create a chart using QuickChart API and return base64 encoded image"""
        try:
            params = {
                'chart': json.dumps(chart_config),
                'width': width,
                'height': height,
                'format': 'png'
            }
            
            response = requests.get(QuickChartTool.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            
            image_base64 = base64.b64encode(response.content).decode('utf-8')
            return image_base64
            
        except requests.exceptions.RequestException as e:
            logging.info(f"Error creating chart: {e}")
            return None
    
    @staticmethod
    def determine_chart_type(question: str, analysis_data: str) -> str:
        """Determine the most appropriate chart type based on question and data"""
        question_lower = question.lower()
        analysis_lower = analysis_data.lower()
        
        if any(keyword in question_lower for keyword in ['compare', 'comparison', 'versus', 'vs']):
            return 'bar'
        elif any(keyword in question_lower for keyword in ['trend', 'over time', 'timeline', 'progress']):
            return 'line'
        elif any(keyword in question_lower for keyword in ['distribution', 'breakdown', 'proportion', 'percentage']):
            return 'pie'
        elif any(keyword in analysis_lower for keyword in ['score', 'rating', 'metric', 'performance']):
            return 'bar'
        elif any(keyword in analysis_lower for keyword in ['complexity', 'security', 'quality']):
            return 'radar'
        else:
            return 'bar'

class MermaidValidationResult(BaseModel):
    is_valid: bool
    confidence_score: int  # 0-100
    issues_found: List[str]
    corrected_code: Optional[str]
    explanation: str

class MermaidLLMValidator:
    """LLM-based Mermaid syntax validator and corrector"""
    
    VALIDATION_PROMPT = """You are an expert Mermaid diagram syntax validator with comprehensive knowledge of all Mermaid diagram types and their syntax rules.

Your task is to validate and optionally correct Mermaid diagram code. You have deep knowledge of:

DIAGRAM TYPES:
- flowchart/graph (TD, TB, BT, RL, LR directions)
- sequenceDiagram 
- classDiagram
- mindmap
- gitgraph
- gantt
- pie
- journey
- requirement
- erDiagram
- stateDiagram

FLOWCHART SYNTAX:
- Node shapes: [], (), {}, [[]], [()], (())，{{}}，>], [/], [\]
- Arrow types: -->, --->, -.-, -.->，==>, ~~>, -.- , ===
- Labels: -->|text|, -->|"text"|
- Subgraphs: subgraph title ... end

SEQUENCE DIAGRAM SYNTAX:
- participant A as "Name"
- A->B: message, A->>B: async, A-->>B: response
- activate/deactivate
- loops, alts, opts

CLASS DIAGRAM SYNTAX:
- class ClassName { +method() -attribute }
- relationships: <|, *, o, ||, }|

COMMON ISSUES TO FIX:
1. Reserved keywords as node IDs (start, end, click, etc.)
2. Invalid characters in node IDs 
3. Unescaped special characters in labels
4. Malformed arrow syntax
5. Missing quotes around labels with spaces/specials
6. Invalid diagram type declarations
7. Bracket/brace mismatches in context

VALIDATION RULES:
- > in arrows like --> or -> is VALID syntax, not unmatched brackets
- | in labeled arrows like -->|"text"| is VALID  
- Reserved words should be avoided as node IDs
- Node IDs should be alphanumeric + underscore/hyphen
- Labels with spaces/specials should be quoted
- Class definitions can span multiple lines with unmatched braces

Respond with structured validation results and corrections if needed."""

    def __init__(self, validation_llm):
        """Initialize with a configured LLM for validation"""
        self.validation_llm = validation_llm.with_structured_output(MermaidValidationResult)

    async def validate_and_correct(self, mermaid_code: str) -> MermaidValidationResult:
        """Validate Mermaid code using LLM and return structured result"""
        
        messages = [
            SystemMessage(content=self.VALIDATION_PROMPT),
            HumanMessage(content=f"""Please validate this Mermaid diagram code:

```mermaid
{mermaid_code}
```

Instructions:
1. Check if the syntax is valid according to Mermaid specifications
2. Identify any syntax errors, reserved keyword conflicts, or formatting issues
3. If issues exist, provide corrected code
4. Give confidence score (0-100) for your assessment
5. Remember: > in arrows (-->, ->) is valid syntax, not bracket errors
6. Be thorough but practical - focus on real syntax issues

Return structured validation results.""")
        ]
        
        try:
            result = self.validation_llm.invoke(messages)
            return result
        except Exception as e:
            logging.error(f"LLM validation failed: {e}")
            # Fallback result
            return MermaidValidationResult(
                is_valid=True,  # Assume valid if validation fails
                confidence_score=50,
                issues_found=[f"Validation service error: {str(e)}"],
                corrected_code=None,
                explanation="Could not validate due to service error, proceeding with original code"
            )

# Updated MermaidTool with LLM validation
class MermaidTool:
    BASE_URL = "https://mermaid.ink/img/"
    RESERVED_KEYWORDS: Set[str] = {
        'end', 'start', 'click', 'subgraph', 'direction_tb', 'direction_bt', 
        'direction_lr', 'direction_rl', 'graph', 'flowchart', 'sequenceDiagram', 
        'classDiagram', 'mindmap', 'participant', 'activate', 'deactivate',
        'o', 'x' 
    }

    @staticmethod 
    def create_diagram(diagram_data: 'MermaidDiagramData', validation_llm=None) -> Optional[str]:
        """Create a Mermaid diagram with LLM-based validation"""
        try:
            mermaid_code = MermaidTool.generate_mermaid_syntax(diagram_data)
            logging.info(f"Generated Mermaid code:\n{mermaid_code}")
            
            # Use LLM validation if available
            if validation_llm:
                validator = MermaidLLMValidator(validation_llm)
                validation_result = validator.validate_and_correct(mermaid_code)
                
                logging.info(f"LLM validation result: Valid={validation_result.is_valid}, Confidence={validation_result.confidence_score}")
                
                if validation_result.issues_found:
                    logging.info(f"Issues found: {validation_result.issues_found}")
                
                # Use corrected code if available and confidence is high
                if validation_result.corrected_code and validation_result.confidence_score >= 70:
                    logging.info("Using LLM-corrected Mermaid code")
                    mermaid_code = validation_result.corrected_code
                elif not validation_result.is_valid and validation_result.confidence_score >= 80:
                    logging.error(f"LLM detected invalid syntax: {validation_result.explanation}")
                    return None
            else:
                # Fallback to minimal validation
                if not MermaidTool._minimal_validate(mermaid_code):
                    logging.error("Minimal validation failed")
                    return None

            # Generate diagram
            graph_json = {
                "code": mermaid_code,
                "mermaid": {
                    "theme": "default",
                    "themeVariables": {
                        "fontSize": "16px"
                    }
                },
                "autoSync": True,
                "updateEditor": False
            }
            json_str = json.dumps(graph_json, ensure_ascii=True, separators=(',', ':'))
            compressed = zlib.compress(json_str.encode('utf-8'), level=9)
            
            encoded_diagram = base64.urlsafe_b64encode(compressed).decode('ascii')
            url = f"{MermaidTool.BASE_URL}pako:{encoded_diagram}"
            
            logging.info(f"Mermaid API URL length: {len(url)}")
            headers = {
                'Accept': 'image/png',
                'User-Agent': 'GitSpeak/1.0'
            }
            response = requests.get(url, headers=headers, timeout=30)
            
            if response.status_code != 200:
                logging.error(f"Mermaid API error {response.status_code}: {response.text}")
                return None
                
            image_base64 = base64.b64encode(response.content).decode('utf-8')
            logging.info("Mermaid diagram generated successfully")
            return image_base64
            
        except Exception as e:
            logging.error(f"Error creating Mermaid diagram: {str(e)}")
            return None

    @staticmethod
    def _minimal_validate(mermaid_code: str) -> bool:
        """Minimal fallback validation"""
        if not mermaid_code or not mermaid_code.strip():
            return False
        
        lines = mermaid_code.strip().split('\n')
        if not lines:
            return False
        
        # Just check diagram type exists
        first_line = lines[0].strip().lower()
        valid_types = ['flowchart', 'graph', 'sequencediagram', 'classdiagram', 'mindmap', 'gitgraph']
        return any(diagram_type in first_line for diagram_type in valid_types)

    @staticmethod
    def _auto_fix_common_errors(mermaid_code: str) -> str:
        """Keep existing auto-fix for reserved words"""
        lines = mermaid_code.strip().split('\n')
        fixed_lines = []
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('%%'): 
                fixed_lines.append(line)
                continue
            
            # Fix reserved word node IDs (e.g., 'end' -> 'end_node')
            if '[' in stripped or '(' in stripped or '{' in stripped:
                if '[' in stripped:
                    parts = stripped.split('[', 1)
                    if len(parts) == 2:
                        potential_id = parts[0].strip().split()[-1]
                        if potential_id.lower() in MermaidTool.RESERVED_KEYWORDS:
                            safe_id = f"{potential_id}_node"
                            new_line = line.replace(potential_id, safe_id, 1)
                            fixed_lines.append(new_line)
                            logging.info(f"Fixed reserved ID: {potential_id} -> {safe_id}")
                            continue
                
                elif '(' in stripped:
                    parts = stripped.split('(', 1)
                    if len(parts) == 2:
                        potential_id = parts[0].strip().split()[-1]
                        if potential_id.lower() in MermaidTool.RESERVED_KEYWORDS:
                            safe_id = f"{potential_id}_node"
                            new_line = line.replace(potential_id, safe_id, 1)
                            fixed_lines.append(new_line)
                            continue
            
            fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)

    # Keep all existing generation methods unchanged
    @staticmethod
    def generate_mermaid_syntax(diagram_data: 'MermaidDiagramData') -> str:
        """Generate Mermaid syntax from diagram data"""
        if diagram_data.type == "flowchart" and diagram_data.flowchart:
            return MermaidTool._generate_flowchart(diagram_data.flowchart)
        elif diagram_data.type == "classDiagram" and diagram_data.class_diagram:
            return MermaidTool._generate_class_diagram(diagram_data.class_diagram)
        elif diagram_data.type == "sequenceDiagram" and diagram_data.sequence_diagram:
            return MermaidTool._generate_sequence_diagram(diagram_data.sequence_diagram)
        elif diagram_data.type == "mindmap" and diagram_data.mindmap:
            return MermaidTool._generate_mindmap(diagram_data.mindmap)
        else:
            raise ValueError(f"Unsupported diagram type: {diagram_data.type}")

    @staticmethod
    def _generate_flowchart(flowchart) -> str:
        """Generate flowchart syntax with proper formatting, quoting, and reserved ID handling"""
        lines = [f"flowchart {flowchart.direction}"]
        
        # Add nodes with safe IDs and quoting
        for node in flowchart.nodes:
            shape_open, shape_close = MermaidTool._get_node_shapes(node.shape)
            safe_label = MermaidTool._escape_and_truncate_label(node.label)
            safe_id = MermaidTool._safe_id(node.id)
            # Always quote labels to handle specials
            lines.append(f'    {safe_id}{shape_open}"{safe_label}"{shape_close}')
        
        # Add edges with safe IDs and quoted labels
        for edge in flowchart.edges:
            safe_from = MermaidTool._safe_id(edge.from_node)
            safe_to = MermaidTool._safe_id(edge.to_node)
            
            if edge.label:
                safe_edge_label = MermaidTool._escape_and_truncate_label(edge.label)
                lines.append(f'    {safe_from} {edge.arrow_type}|"{safe_edge_label}"| {safe_to}')
            else:
                lines.append(f'    {safe_from} {edge.arrow_type} {safe_to}')
        
        return "\n".join(lines)

    @staticmethod
    def _generate_class_diagram(class_diagram) -> str:
        """Generate class diagram syntax with enhanced escaping and safe IDs"""
        lines = ["classDiagram"]
        
        # Add classes with safe names
        for cls in class_diagram.classes:
            safe_name = MermaidTool._safe_id(cls.name)
            lines.append(f"    class {safe_name} {{")
            
            # Add attributes
            for attr in cls.attributes:
                visibility = MermaidTool._get_visibility_symbol(attr.visibility)
                safe_attr_name = MermaidTool._escape_and_truncate_label(attr.name, max_len=40)
                attr_type = f": {MermaidTool._escape_and_truncate_label(attr.return_type)}" if hasattr(attr, 'return_type') and attr.return_type else ""
                lines.append(f"        {visibility}{safe_attr_name}{attr_type}")
            
            # Add methods
            for method in cls.methods:
                visibility = MermaidTool._get_visibility_symbol(method.visibility)
                safe_method_name = MermaidTool._escape_and_truncate_label(method.name, max_len=40)
                params = "()"  # Simplified; extend if needed
                return_type = f": {MermaidTool._escape_and_truncate_label(method.return_type)}" if hasattr(method, 'return_type') and method.return_type else ""
                lines.append(f"        {visibility}{safe_method_name}{params}{return_type}")
            
            lines.append("    }")
        
        # Add relationships
        for rel in class_diagram.relationships:
            safe_from = MermaidTool._safe_id(rel.from_)
            safe_to = MermaidTool._safe_id(rel.to)
            lines.append(f"    {safe_from} {rel.type} {safe_to}")
        
        return "\n".join(lines)

    @staticmethod
    def _generate_sequence_diagram(sequence) -> str:
        """Generate sequence diagram syntax with quoting for messages and safe participant IDs"""
        lines = ["sequenceDiagram"]
        
        # Add participants
        for participant in sequence.participants:
            safe_id = MermaidTool._safe_id(participant.id)
            safe_name = MermaidTool._escape_and_truncate_label(participant.display_name)
            lines.append(f'    participant {safe_id} as "{safe_name}"')
        
        # Add messages
        for msg in sequence.messages:
            safe_from = MermaidTool._safe_id(msg.from_participant)
            safe_to = MermaidTool._safe_id(msg.to_participant)
            safe_message = MermaidTool._escape_and_truncate_label(msg.message)
            
            if hasattr(msg, 'activation') and msg.activation:
                lines.append(f"    activate {safe_to}")
            lines.append(f'    {safe_from}{msg.arrow_type}{safe_to}: "{safe_message}"')
            if hasattr(msg, 'activation') and msg.activation:
                lines.append(f"    deactivate {safe_to}")
        
        return "\n".join(lines)

    @staticmethod
    def _generate_mindmap(mindmap) -> str:
        """Generate mindmap syntax with enhanced escaping and safe node handling"""
        lines = ["mindmap"]
        safe_root = MermaidTool._escape_and_truncate_label(mindmap.root_text)
        lines.append(f'  root(("{safe_root}"))')
        
        # Sort nodes by level for proper hierarchy
        sorted_nodes = sorted(mindmap.nodes, key=lambda x: (x.level, x.id or ''))
        
        for node in sorted_nodes:
            indent = "  " * (node.level + 1)  # Consistent indent
            safe_text = MermaidTool._escape_and_truncate_label(node.text, max_len=30)
            # For mindmap, use parentheses for rounded nodes
            lines.append(f'{indent}("{safe_text}")')  # Standard mindmap syntax uses ()
        
        return "\n".join(lines)

    @staticmethod
    def _get_node_shapes(shape: str) -> tuple:
        """Get node shape brackets"""
        shapes = {
            "rect": ("[", "]"),
            "circle": ("((", "))"),
            "diamond": ("{", "}"),
            "hexagon": ("{{", "}}"),
            "round": ("(", ")"),  # For mindmap-like
            "stadium": ("([", "])"),
            "subroutine": ("[[", "]]"),
            "cylinder": ("[(", ")]"),
            "asym": (">", "]"),
            "rhombus": ("{", "}"),
            "parallelogram": ("/[", "/]"),
            "parallelogram-alt": ("[/", "\\]"),
            "trapezoid": ("[/", "\\]"),
            "trapezoid-alt": ("\\]", "/]")
        }
        return shapes.get(shape, ("[", "]"))

    @staticmethod
    def _get_visibility_symbol(visibility: str) -> str:
        """Get visibility symbol for class diagrams"""
        symbols = {
            "public": "+",
            "private": "-",
            "protected": "#",
            "package": "~"
        }
        return symbols.get(visibility, "+")

    @staticmethod
    def _escape_and_truncate_label(text: str, max_len: int = 50) -> str:
        """Dynamically escape special characters and truncate while ensuring balanced brackets and no syntax breakers"""
        if not text:
            return ""
        
        # Step 1: Escape core specials for Mermaid (dynamic escaping)
        escaped = text.replace('\\', '\\\\')  # Escape backslash first
        escaped = escaped.replace('"', '\\"')  # Escape quotes
        escaped = escaped.replace('\n', ' ')   # Flatten newlines
        escaped = escaped.replace('\r', ' ')
        escaped = escaped.replace('|', '-')  
        escaped = escaped.replace('-->', ' to ') 
        escaped = escaped.replace('---', ' - ')   
        escaped = escaped.replace(';', ',')      
        escaped = escaped.replace(':', ' - ')     
        if len(escaped) > max_len:
            brackets = {'(': 0, '[': 0, '{': 0, '"': 0, "'": 0}
            reverse_brackets = {')': '(', ']': '[', '}': '{', '"': '"', "'": "'"}
            balance = {k: 0 for k in brackets}
            
            cut_pos = max_len
            for i in range(len(escaped)):
                c = escaped[i]
                if c in brackets:
                    balance[c] += 1
                elif c in reverse_brackets:
                    open_b = reverse_brackets[c]
                    if balance[open_b] > 0:
                        balance[open_b] -= 1
                
                if i >= max_len - 1 and all(v == 0 for v in balance.values()):
                    cut_pos = i + 1
                    break
            
            escaped = escaped[:cut_pos]
            if cut_pos < len(text):  
                escaped += "..."
            closes = ""
            close_map = {'(': ')', '[': ']', '{': '}'}
            for open_b, count in balance.items():
                if count > 0 and open_b in close_map:
                    closes += close_map[open_b] * count
            escaped += closes
        
        return escaped.strip()

    @staticmethod
    def _safe_id(id_string: str) -> str:
        """Create safe ID for Mermaid: alphanumeric, prefixed if needed, avoid reserved"""
        if not id_string:
            return "node"
        safe = id_string.replace(' ', '_').replace('-', '_').replace('.', '_')
        safe = ''.join(c if c.isalnum() or c in ('_', '-') else '_' for c in safe)
        if not safe[0].isalpha():
            safe = 'n_' + safe
        max_length = getattr(MermaidTool, 'MERMAID_MAX_URL_SAFE_LENGTH', 50)
        safe = safe[:max_length]
        safe_lower = safe.lower()
        if safe_lower in MermaidTool.RESERVED_KEYWORDS:
            safe = safe + '_node'
            logging.info(f"Appended '_node' to reserved ID: {id_string}")
        return safe

def generate_folder_structure(root_folder: Path, indent: str = "", max_depth: int = 8, current_depth: int = 0) -> str:
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
            if item.startswith('.') or item in ['__pycache__', 'node_modules', '.git', 'venv', 'env']:
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
            if file.lower() in ['readme.md', 'readme.txt', 'readme.rst', 'main.py', 'app.py', 'index.js', 'package.json', 'requirements.txt', 'setup.py', 'dockerfile', 'makefile']:
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
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules', 'venv', 'env']]
            
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
        logging.info(f"Error analyzing project structure: {e}")
    
    return analysis

def get_repo_hash(repo_url: str) -> str:
    """Generate unique hash for repository URL"""
    return hashlib.md5(repo_url.encode()).hexdigest()

def save_session_data(session_id: str, data: Dict[str, Any]):
    """Save session data to persistent storage with enhanced serialization"""
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
        logging.info(f"Error saving session data: {e}")

def load_session_data(session_id: str) -> Dict[str, Any]:
    """Load session data from persistent storage with enhanced deserialization"""
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
        logging.info(f"Error saving retriever: {e}")

def load_retriever_data(repo_hash: str):
    """Load vector store, metadata, and folder structure with version check"""
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

def extract_code_units(file_path):
    """Extract code units from various file types with enhanced parsing"""
    units = []
    
    if file_path.endswith(".py"):
        with open(file_path, "r", encoding="utf-8", errors='ignore') as f:
            code = f.read()
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    start_line = node.lineno
                    end_line = node.end_lineno
                    chunk = "\n".join(code.splitlines()[start_line-1:end_line])
                    metadata = {"file": file_path, "name": node.name, "type": type(node).__name__, "lines": end_line - start_line + 1}
                    units.append((chunk, metadata))
        except SyntaxError:
            metadata = {"file": file_path, "name": "unknown", "type": "file", "lines": len(code.splitlines())}
            units.append((code, metadata))
    
    elif file_path.endswith(".ipynb"):
        with open(file_path, "r", encoding="utf-8", errors='ignore') as f:
            notebook = nbformat.read(f, as_version=4)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                code = cell.source
                try:
                    tree = ast.parse(code)
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                            chunk = code
                            metadata = {"file": file_path, "name": node.name, "type": type(node).__name__, "cell_index": notebook.cells.index(cell), "lines": len(code.splitlines())}
                            units.append((chunk, metadata))
                except SyntaxError:
                    metadata = {"file": file_path, "name": "unknown", "type": "cell", "cell_index": notebook.cells.index(cell), "lines": len(code.splitlines())}
                    units.append((code, metadata))
    
    elif file_path.endswith((
        ".c", ".cpp", ".js", ".ts", ".tsx", ".jsx", ".go", ".java", ".rb", ".php", 
        ".swift", ".sh", ".bash", ".ps1", ".r", ".R", ".yaml", ".yml", ".json", 
        ".xml", ".html", ".htm", ".css", ".scss", ".sass", ".sql", ".dockerfile", 
        ".Dockerfile", ".md", ".markdown", ".makefile", ".Makefile", ".mk"
    )):
        try:
            with open(file_path, "r", encoding="utf-8", errors='ignore') as f:
                code = f.read()
            metadata = {"file": file_path, "name": os.path.basename(file_path), "type": "file", "lines": len(code.splitlines())}
            units.append((code, metadata))
        except Exception:
            pass  
    
    return units

async def clone_repo_via_api(repo_url: str, github_token: str) -> Path:
    """Clone repository using GitHub API (supports private repos)"""
    try:
        from urllib.parse import urlparse
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
            if file_count > 1500: 
                break
            for file in files:
                if file_count > 1500:
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


async def get_pr_tool(repo_url: str, pr_number: Optional[int] = None) -> Dict[str, Any]:
    """Fetch PR information using GitHub API, including diffs, commits, and CI/CD check runs.
    If pr_number is None, fetch the latest open PR's details."""
    
    logging.info(f"get_pr_tool called with repo_url={repo_url}, pr_number={pr_number}")
    
    if not GITHUB_TOKEN:
        logging.error("No GITHUB_TOKEN available")
        return {"error": "Please provide GITHUB_TOKEN environment variable or proper access token for accessing pull request information."}
    
    try:
        from urllib.parse import urlparse
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
                async with session.get(f"{base_url}/pulls", headers=headers, params={"state": "open", "per_page": 50}) as resp:
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


CODE_ANALYST_PROMPT = """
Focus deeply on understanding and analyzing the user question: {question}. Aim to explore relationships between different functions, files, modules, and configurations in the code. Provide code examples wherever possible to illustrate your findings or recommendations.

You are a Code Analyst, expert in code architecture, design patterns, dependencies, modularity, complexity, and code quality.

Your goal:
- Analyze the user question in detail to understand the exact intent and context.
- You must give the code snippets wherever possible for better understanding liek which code snippts from which file are you referring to.Try to give code whenever tryinh to explain any function or module side by side so that user can understand better.
- Analyze each file in the provided code context to identify key functions, classes, and modules.
- Map relationships between functions, files, and modules based on the provided code and folder structure.
- Identify code flows, data dependencies, and interaction patterns.
- Generate relevant code examples or snippets that help explain your analysis or provide actionable suggestions.
- Provide clear recommendations for refactoring, optimization, or improving code structure based on the question.
- Must remember that  your work is not generating or giving mermaid code it is analyzing the code and giving suggestions.If user ask about mermaid code then leave it do your job of analyzing files because there are other agents who will give mermaid code.

Context provided:
- Folder Structure: {folder_structure}
- Project Analysis: {project_analysis}
- Code Context: {context}

Pull Request Analysis:
{pr_data_str}

IMPORTANT: If the PR Data Reference above contains actual pull request information (not "No PR data." or an error message), analyze it thoroughly:
- Extract and explain the specific changes from the diff
- Identify which files were modified and what changes were made
- Explain the purpose and impact of the changes
- Relate the PR changes to the user's question
- Highlight any improvements in architecture, security, performance, or code quality
- Include CI/CD status and review information if available

If the PR data shows an error or states "No PR data.", then clearly state that PR analysis is not available.

Provide a comprehensive and actionable analysis with code examples, relationship diagrams if relevant, and in-depth insights based on the question and available data.

Question: {question}
History: {history}

Answer:"""

SECURITY_AGENT_PROMPT = """
You are an Elite Security Agent, expert in identifying threats, vulnerabilities, secure architecture, and best practices in code security.
Focus deeply on understanding and analyzing the user question: {question}. Aim to perform a thorough security assessment of each file in the codebase, including configuration files, code modules, and infrastructure definitions.
Your goal:
- Analyze each file in the provided code context to identify potential security vulnerabilities, insecure practices, misconfigurations, or weaknesses.
- Map relationships between code modules and assess potential attack vectors (e.g., data flow from untrusted inputs to sensitive operations).
- - You must give code for security improvements wherever possible.
- Evaluate security posture in code, configurations, authentication flows, dependency management, and infrastructure as code.
- Generate concrete remediation suggestions, with code snippets where possible.
- Highlight any use of insecure libraries, improper key management, exposed secrets, unsafe input handling, weak authorization, etc.
- Provide recommendations to improve the overall security design, including security patterns like Zero Trust or Least Privilege.
- See your work is not generating or giving mermaid code it is analyzing security and vulnerabilities.
- Must remember that  your work is not generating or giving mermaid code it is analyzing the code and giving security suggestions.If user ask about mermaid code then leave it do your job of analyzing files because there are other agents who will give mermaid code.

Context Provided:
- Folder Structure: {folder_structure}
- Project Analysis: {project_analysis}
- Code Context: {context}

When analyzing Pull Requests:
- Summarize any security-related changes in the PR(s), including fixes, new vulnerabilities, or misconfigurations introduced.
- Relate PR changes to the user question and security best practices.
- Highlight security improvements or concerns.

If no PRs are found or an error is present, state it clearly.

PR Data Reference: {pr_data_str}

Structure your analysis file by file, clearly indicating:
- Potential security issues found.
- Risk level (Critical / High / Medium / Low).
- Example vulnerable code snippet (if applicable).
- Suggested secure code refactor or configuration improvement.

Ensure your assessment includes actionable code examples or configuration fixes wherever possible.

Question: {question}
Previous Messages: {messages}

Conduct a comprehensive, detailed security assessment with insights, examples, and clear remediation steps.

Answer:
"""


PERFORMANCE_AGENT_PROMPT = """
Focus deeply on understanding and analyzing the user question: {question}. Aim to analyze the performance of each file, function, and configuration in the codebase.

You are a Senior Performance Agent, expert in code optimization, scalability, and system-level performance.

Your goal:
- Analyze code and configurations to identify CPU, memory, I/O bottlenecks, inefficient algorithms, and blocking operations.
- Map relationships between code modules, data flow, and system resources.
- Detect inefficient DB queries, missing indexes, improper caching, and synchronous calls where async would help.
- Provide concrete code-level optimizations with examples where applicable.
- Suggest architectural improvements for scalability (e.g., sharding, load balancing).
- Highlight concurrency issues and offer patterns (async, thread pools, etc.).
- Give suggestions on tuning configurations or deployment setup.
- See your work is not generating or giving mermaid code it is analyzing performance and bottlenecks.
- You must give code for optimization wherever possible.
- Must remember that  your work is not generating or giving mermaid code it is analyzing the code and giving performance suggestions.If user ask about mermaid code then leave it do your job of analyzing files because there are other agents who will give mermaid code.

Context Provided:
- Folder Structure: {folder_structure}
- Project Analysis: {project_analysis}
- Code Context: {context}

When analyzing Pull Requests:
- Summarize any performance-related changes in the PR(s), including optimizations, code refactors, new performance bottlenecks introduced, or removed inefficiencies.
- Relate PR changes to the user question.

If no PRs are found or an error is present, state it clearly.

PR Data Reference: {pr_data_str}

Structure your analysis file by file, clearly indicating:
- Performance issue detected.
- Risk level (Critical / High / Medium / Low).
- Example inefficient code snippet.
- Suggested optimized code or configuration fix.

Question: {question}
Previous Messages: {messages}
Security: {security_message}

Provide a comprehensive performance assessment with actionable recommendations and code examples.

Answer:
"""


QUALITY_AGENT_PROMPT = """
Focus deeply on understanding and analyzing the user question: {question}. Aim to assess code quality, maintainability, testing coverage, and documentation of each file.

You are a Quality Agent, expert in software quality, testing strategies, code maintainability, and best practices with advanced visualization capabilities.

Your goal:
- Analyze code analysis from all other agents for analysis,security , readability, modularity, duplication, code smells, and adherence to best practices.
- Detect missing or inadequate unit tests, improper exception handling, poor naming, and overly complex functions.
- Map relationships between modules to assess separation of concerns and cohesion.
- Identify gaps in inline comments and API documentation.
- Provide concrete refactoring suggestions with code examples.
- Recommend how to structure tests (unit vs integration) and increase coverage.
- Highlight configuration improvements for quality gates, linters, and CI checks.

VISUALIZATION CAPABILITIES:
You have access to two powerful visualization tools:

1. **CHART TOOL (QuickChart API)**:
   - Can generate: bar, line, pie, doughnut, radar, polar area charts
   - Best for: Quality metrics, scores, comparisons, trends, distributions
   - Use when question asks for: metrics, scores, dashboards, comparisons, breakdowns
   - Chart types:
     * Bar: Comparing metrics across files/modules
     * Line: Trends over time or progression
     * Pie/Doughnut: Distribution breakdowns (issue types, severity levels)
     * Radar: Multi-dimensional quality scores (security, performance, maintainability)
   - Always populate realistic scores (0-100) based on analysis findings
   - Include professional styling with appropriate colors and labels

2. **MERMAID DIAGRAM TOOL**:
   - Can generate: flowcharts, class diagrams, sequence diagrams, mindmaps
   - Best for: Architecture visualization, code relationships, process flows
   - Use when question asks for: architecture, structure, flow, relationships, mindmaps
   - Diagram types:
     * Flowchart: Code execution paths, decision trees, process flows
     * Class Diagram: OOP relationships, inheritance, dependencies  
     * Sequence Diagram: Interaction flows, API calls, method sequences
     * Mindmap: Feature breakdowns, concept hierarchies, knowledge maps

WHEN TO CREATE VISUALIZATIONS:
- Charts: When user asks for metrics, scores, quality assessments, comparisons, dashboards
- Diagrams: When user asks for architecture, relationships, flows, structure visualization, mindmaps
- Always create when explicitly requested: "show me a chart", "create a diagram", "dashboard", "visualize"
- For explanatory questions about specific code functionality: provide detailed text analysis without unnecessary visuals

Context Provided:
- Folder Structure: {folder_structure}
- Project Analysis: {project_analysis}
- Code Context: {context}

When analyzing Pull Requests:
- Summarize any quality-related changes in the PR(s), including added tests, refactors improving readability, or worsened code quality.
- Relate PR changes to the user question.
- Consider visualizing PR impact if metrics are involved

If no PRs are found or an error is present, state it clearly.

PR Data Reference: {pr_data_str}

Structure your analysis file by file, clearly indicating:
- Quality issue found.
- Risk level (Critical / High / Medium / Low).
- Example problematic code snippet.
- Suggested refactored code or testing improvement.

CRITICAL VISUALIZATION INSTRUCTIONS:
- Only generate visualizations when they enhance understanding of the analysis
- Always populate ALL metric fields with realistic, justified scores (0-100) based on findings
- Include new quality metrics: reliability_score, usability_score, test_coverage_score
- For charts: Use professional color schemes, clear labels, appropriate chart types
- For diagrams: Ensure proper syntax, clear node relationships, meaningful labels
- Set should_create_chart=true AND should_create_diagram=true when both are beneficial

Question: {question}
Code Analysis: {messages}
Security: {security_message}
Performance: {performance_message}

Provide a comprehensive quality assessment with actionable refactors, test strategies, examples, and appropriate visualizations when requested.

Answer:
"""

AGENT_SELECTION_PROMPT = """Analyze question: {question}

You select agents: Code Analyst (always first), then Security, Performance, Quality based on intent.

Route to:
- Security: Vulns, threats, auth, crypto.
- Performance: Speed, scalability, efficiency.
- Quality: Maintainability, tests, docs.

Reasoning: Explain choices, priority.

History: {history}

Output structured selection."""

PRESENT_TELLER_PROMPT = """You explain agent invocation briefly (1-2 lines) based on reasoning, priority, question.

Reasoning: {reasoning}
Priority: {priority_order}
Question: {question}
Agent: {current_agent}

Provide explanation."""



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
        return END

def security_router(state: AgentState):
    if state["goto_performance_agent"]:
        return "present_teller_performance"
    elif state["goto_quality_agent"]:
        return "present_teller_quality"
    else:
        return END

def performance_router(state: AgentState):
    if state["goto_quality_agent"]:
        return "present_teller_quality"
    else:
        return END

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

    history_str = "\n".join([f"{'User' if isinstance(msg, HumanMessage) else 'AI'}: {msg.content[:2500]}" for msg in state.get("conversation_history", [])])
    
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

async def create_workflow():
    workflow = StateGraph(AgentState)
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
    global workflow_app
    if workflow_app is None:
        workflow_app = await create_workflow()
    return workflow_app

@cl.on_chat_start
async def start():
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
                timeout=300
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
                timeout=300
            ).send()
            
            if user_input and ("github.com" in user_input.get("output", "") or "gitlab.com" in user_input.get("output", "")):
                await setup_new_repository(user_input.get("output", "").strip())
                return
            else:
                await cl.Message(
                    content="Valid URL. Ex: https://github.com/user/repo", 
                    author="GitSpeak"
                ).send()

async def setup_new_repository(repo_url: str):
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

@cl.on_message
async def main(message: cl.Message):
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
        thread_config = {"configurable": {"thread_id": f"{repo_hash}_{datetime.now().isoformat()}"}, "checkpointer": checkpointer}
        
        history_str = "\n".join([f"{'User' if isinstance(msg, HumanMessage) else 'AI'}: {msg.content[:500]}" for msg in conversation_history])
        
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
