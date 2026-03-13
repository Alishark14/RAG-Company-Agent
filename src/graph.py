from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from src.tools import get_retriever_tool

class State(TypedDict):
    messages: Annotated[list, add_messages]

def create_app(index_name: str):
    # 1. Setup Tools
    tool = get_retriever_tool(index_name)
    tools = [tool]
    
    # 2. Setup LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0
    ).bind_tools(tools)

    # 3. Define Node Logic
    def chatbot_node(state: State):
        return {"messages": [llm.invoke(state["messages"])]}

    # 4. Build Graph
    builder = StateGraph(State)
    builder.add_node("agent", chatbot_node)
    builder.add_node("tools", ToolNode(tools))
    
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")
    
    # 5. Compile with Memory
    return builder.compile(checkpointer=MemorySaver())
