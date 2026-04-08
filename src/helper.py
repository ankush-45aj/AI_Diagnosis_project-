from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain.schema import Document

# Extract Data From the PDF File
def load_pdf_file(data_path):
    loader = DirectoryLoader(
        data_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    return loader.load()

# Filter metadata
def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    minimal_docs: List[Document] = []
    for doc in docs:
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": doc.metadata.get("source")}
            )
        )
    return minimal_docs

# Split the Data into Text Chunks
def text_split(extracted_data):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20
    )
    return splitter.split_documents(extracted_data)

# Download Embeddings
def download_hugging_face_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )