"""
agent.py — puts the pieces together:

    agent = LLM (llm.py) + system prompt (prompts.py) + tools (tools.py) + memory
"""

from langchain.agents import create_agent
from langchain_community.vectorstores import FAISS
from langgraph.checkpoint.memory import InMemorySaver

from app.llm import get_llm
from app.prompts import get_system_prompt
from app.tools import make_search_tool

# Memory: remembers every conversation (one "thread" per chat session) while the server runs
memory = InMemorySaver()


def build_agent(provider: str, vector_db: FAISS, document_name: str):
    search_pdf, found_chunks = make_search_tool(vector_db)
    agent = create_agent(
        model=get_llm(provider),
        tools=[search_pdf],
        system_prompt=get_system_prompt(document_name),
        checkpointer=memory,
    )
    return agent, found_chunks


def ask(agent, question: str, session_id: str) -> str:
    """Send one message. The same session_id = the agent remembers earlier messages."""
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config={"configurable": {"thread_id": session_id}},
    )
    return result["messages"][-1].text
