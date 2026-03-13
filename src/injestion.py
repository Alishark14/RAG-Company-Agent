from pinecone import Pinecone
from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

def initialize_index(policy_text, api_key):
    embeddings = PineconeEmbeddings(model="llama-text-embed-v2", pinecone_api_key=api_key)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents([policy_text])
    
    return PineconeVectorStore.from_documents(
        docs, embeddings, index_name="rag-project"
    )
