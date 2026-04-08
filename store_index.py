import os
import json
from typing import List

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

# ---------------- CONFIGURATION & ENV ----------------
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
DATA_DIR = "Data/"
MANIFEST_FILE = "processed_files.json"     # Tracks completed files
PROGRESS_FILE = "upload_progress.json"     # Tracks chunk progress for current batch
INDEX_NAME = "medical-chatbot"
BATCH_SIZE = 100 

if not PINECONE_API_KEY:
    raise ValueError("Missing Pinecone API key ❌")

# ---------------- HELPER FUNCTIONS ----------------

def load_json(filepath, default_type=set):
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            data = json.load(f)
            return set(data) if default_type == set else data
    return default_type()

def save_json(filepath, data):
    with open(filepath, "w") as f:
        json.dump(list(data) if isinstance(data, set) else data, f)

def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    return [
        Document(
            page_content=doc.page_content,
            metadata={"source": doc.metadata.get("source")}
        ) for doc in docs
    ]

def text_split(extracted_data):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    return splitter.split_documents(extracted_data)

def download_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ---------------- MAIN EXECUTION ----------------

def run_pipeline():
    embeddings = download_embeddings()
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # 1. Ensure Index Exists
    if INDEX_NAME not in [i.name for i in pc.list_indexes()]:
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    
    # 2. Identify NEW files
    processed_files = load_json(MANIFEST_FILE, set)
    
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        return

    all_pdfs = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
    new_pdfs = [f for f in all_pdfs if f not in processed_files]

    if not new_pdfs:
        print("Everything is up to date ✅")
        return

    # 3. Extract and Split all new files
    print(f"Preparing {len(new_pdfs)} new files...")
    all_new_chunks = []
    for pdf_name in new_pdfs:
        loader = PyPDFLoader(os.path.join(DATA_DIR, pdf_name))
        chunks = text_split(filter_to_minimal_docs(loader.load()))
        all_new_chunks.extend(chunks)

    # 4. Batched Upload with Resume Logic
    if all_new_chunks:
        progress = load_json(PROGRESS_FILE, dict)
        # Resuming from the last index, or 0 if starting fresh
        start_index = progress.get("last_index", 0)
        
        print(f"Total chunks to upload: {len(all_new_chunks)}")
        if start_index > 0:
            print(f"Resuming from chunk index: {start_index} ↩️")

        # Initialize the vector store connection
        vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)

        for i in range(start_index, len(all_new_chunks), BATCH_SIZE):
            batch = all_new_chunks[i : i + BATCH_SIZE]
            
            # Retry logic for network blips
            for attempt in range(3):
                try:
                    vectorstore.add_documents(batch)
                    
                    # Update progress immediately after successful batch
                    progress["last_index"] = i + BATCH_SIZE
                    save_json(PROGRESS_FILE, progress)
                    
                    print(f"Uploaded: {min(i + BATCH_SIZE, len(all_new_chunks))}/{len(all_new_chunks)}")
                    break 
                except Exception as e:
                    print(f"Retry {attempt+1} failed at index {i}: {e}")
                    if attempt == 2:
                        print("❌ Critical error. Progress saved. Run the script again to resume.")
                        return

        # 5. Finalize - Only happens if EVERY batch succeeds
        processed_files.update(new_pdfs)
        save_json(MANIFEST_FILE, processed_files)
        
        # Clean up the progress file because we are done
        if os.path.exists(PROGRESS_FILE):
            os.remove(PROGRESS_FILE)
            
        print("All new data uploaded and indexed successfully! 🚀")

if __name__ == "__main__":
    run_pipeline()