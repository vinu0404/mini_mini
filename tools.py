import json
import base64
import zlib
import logging
import requests
from typing import Set, Dict, Any, Optional
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from typing import List, Optional, Dict, Any

from config import (
    QUICKCHART_BASE_URL,
    MERMAID_BASE_URL,
    HTTP_TIMEOUT,
    MERMAID_MAX_URL_SAFE_LENGTH
)
from models import MermaidDiagramData


class QuickChartTool:
    BASE_URL = QUICKCHART_BASE_URL
    
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
            
            response = requests.get(QuickChartTool.BASE_URL, params=params, timeout=HTTP_TIMEOUT)
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
    BASE_URL = MERMAID_BASE_URL
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
            response = requests.get(url, headers=headers, timeout=HTTP_TIMEOUT)
            
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