import os
import time
from src.ingestion import ingest_policy_data
from src.graph import create_app

# --- Configuration ---
os.environ["GOOGLE_API_KEY"] = "YOUR_KEY" # Load from .env in production
os.environ["PINECONE_API_KEY"] = "YOUR_KEY"
POLICY_TEXT = "..." # Paste your policy_text here

def main():
    # 1. Ingest
    vectorstore = ingest_policy_data(POLICY_TEXT, "rag-project")
    # 2. Build Graph
    app = create_app(vectorstore)
    # 3. Execution
    config = {"configurable": {"thread_id": "company_qa_session_001"}}
    # ... your ask_agent loop ...

if __name__ == "__main__":
    main()
