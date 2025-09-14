from langchain_openai import ChatOpenAI
from config import (
    OPENAI_API_KEY,
    BASIC_TEMP,
    CODE_ANALYST_TEMP,
    SECURITY_TEMP,
    PERFORMANCE_TEMP,
    QUALITY_TEMP,
    AGENT_SELECTION_TEMP,
    PRESENT_TELLER_TEMP,
    MAX_TOKENS_STANDARD,
    MAX_TOKENS_QUALITY,
    MAX_TOKENS_AGENT_SELECTION,
    MAX_TOKENS_PRESENT_TELLER
)

# LLM Instances Configuration
basic_llm = ChatOpenAI(
    model="gpt-4o", 
    api_key=OPENAI_API_KEY, 
    temperature=BASIC_TEMP, 
    max_tokens=MAX_TOKENS_STANDARD, 
    streaming=True
)

code_analyst_llm = ChatOpenAI(
    model="gpt-4o", 
    api_key=OPENAI_API_KEY, 
    temperature=CODE_ANALYST_TEMP, 
    max_tokens=MAX_TOKENS_STANDARD, 
    streaming=True
)

security_llm = ChatOpenAI(
    model="gpt-4o", 
    api_key=OPENAI_API_KEY, 
    temperature=SECURITY_TEMP, 
    max_tokens=MAX_TOKENS_STANDARD, 
    streaming=True
)

performance_llm = ChatOpenAI(
    model="gpt-4o", 
    api_key=OPENAI_API_KEY, 
    temperature=PERFORMANCE_TEMP, 
    max_tokens=MAX_TOKENS_STANDARD, 
    streaming=True
)

quality_llm = ChatOpenAI(
    model="gpt-4o", 
    api_key=OPENAI_API_KEY, 
    temperature=QUALITY_TEMP, 
    max_tokens=MAX_TOKENS_QUALITY, 
    streaming=True
)

agent_selection_llm = ChatOpenAI(
    model="gpt-4o", 
    api_key=OPENAI_API_KEY, 
    temperature=AGENT_SELECTION_TEMP, 
    max_tokens=MAX_TOKENS_AGENT_SELECTION
)

present_teller_llm = ChatOpenAI(
    model="gpt-4o", 
    api_key=OPENAI_API_KEY, 
    temperature=PRESENT_TELLER_TEMP, 
    max_tokens=MAX_TOKENS_PRESENT_TELLER, 
    streaming=True
)