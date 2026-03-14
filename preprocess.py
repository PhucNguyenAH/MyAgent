import os
import getpass
from langchain_google_community import GCSDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
import dotenv

dotenv.load_dotenv()

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

def main():
    print("Loading documents from Google Cloud Storage...")
    
    # Initialize the GCS loader instead of the local PDF loader
    loader = GCSDirectoryLoader(
        project_name="myagent-490109",
        bucket="anh-portfolio-pdfs",
    )
    docs = loader.load()

    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)

    print("Connecting to PostgreSQL and indexing documents...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name="my_docs",
        connection=os.environ.get("DATABASE_URL"),
        use_jsonb=True,
        pre_delete_collection=True,
    )

    vector_store.add_documents(documents=all_splits)
    print(f"Successfully indexed {len(all_splits)} chunks into the database.")

if __name__ == "__main__":
    main()