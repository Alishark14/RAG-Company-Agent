from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools.retriever import create_retriever_tool

class State(TypedDict):
    messages: Annotated[list, add_messages]

def create_app(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    tool = create_retriever_tool(retriever, "company_docs_search", "Search policies.")
    tools = [tool]

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0).bind_tools(tools)

    def chatbot_node(state: State):
        return {"messages": [llm.invoke(state["messages"])]}

    builder = StateGraph(State)
    builder.add_node("agent", chatbot_node)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")
    
    return builder.compile(checkpointer=MemorySaver())
