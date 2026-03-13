from src.ingestion import initialize_index
from src.graph import create_app
import os

def main():
    api_key = os.getenv("PINECONE_API_KEY")
    # Load policy and build graph...
    app = create_app(vectorstore)
    # Start the CLI interaction loop...

if __name__ == "__main__":
    main()
