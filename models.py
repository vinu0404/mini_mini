from typing import Optional, List, Dict, Any, Union, Set, TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage

# Chart Models
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

# Mermaid Models
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

# Report Models
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

# Response Models
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

# State Model
class AgentState(TypedDict):
    messages: List[BaseMessage]
    on_topic: str 
    rephrased_question: str
    proceed_to_generate: str
    rephrase_count: int
    question: BaseMessage
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