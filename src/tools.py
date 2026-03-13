from langchain_pinecone import PineconeVectorStore, PineconeEmbeddings
from langchain_core.tools.retriever import create_retriever_tool
import os

def get_retriever_tool(index_name: str):
    # Initialize the specific Pinecone embedding model you are using
    embeddings = PineconeEmbeddings(
        model="llama-text-embed-v2",
        pinecone_api_key=os.environ["PINECONE_API_KEY"]
    )
    
    # Connect to existing index
    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        pinecone_api_key=os.environ["PINECONE_API_KEY"]
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    
    return create_retriever_tool(
        retriever,
        "company_docs_search",
        "Search for company policy information. Use this for questions about "
        "vacation, expenses, confidentiality, security, or workplace conduct."
    )
