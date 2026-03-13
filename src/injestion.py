import os
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

def ingest_policy_data(policy_text: str, index_name: str):
    # Setup Embeddings (NVIDIA Inference)
    embeddings = PineconeEmbeddings(
        model="llama-text-embed-v2",
        pinecone_api_key=os.environ["PINECONE_API_KEY"]
    )

    # Chunking
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents([policy_text])
    
    # Pinecone setup
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    if index_name not in [i["name"] for i in pc.list_indexes()]:
        pc.create_index(
            name=index_name,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    # Vector store instantiation
    return PineconeVectorStore.from_documents(
        documents=docs,
        embedding=embeddings,
        index_name=index_name
    )
