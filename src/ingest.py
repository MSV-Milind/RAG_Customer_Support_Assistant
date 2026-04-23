import logging
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHROMA_DIR,
    COLLECTION_NAME,
    DATA_DIR,
    EMBED_MODEL,
)

logger = logging.getLogger(__name__)


# Embedding
def get_embeddings() -> HuggingFaceEmbeddings:
    logger.info("Loading embedding model: %s", EMBED_MODEL)
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},  
        encode_kwargs={"normalize_embeddings": True},
    )


# PDF Loading
def load_pdfs(data_dir: Path = DATA_DIR) -> list:
    pdf_files = sorted(data_dir.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {data_dir}. "
            "Place your Wikipedia (or other) PDFs there and re-run."
        )

    documents = []
    for pdf_path in pdf_files:
        logger.info("Loading PDF: %s", pdf_path.name)
        loader = PyPDFLoader(str(pdf_path))
        pages  = loader.load()
        for page in pages:
            page.metadata["source"] = pdf_path.name
        documents.extend(pages)
        logger.info("  → %d pages loaded", len(pages))

    logger.info("Total pages loaded: %d", len(documents))
    return documents


# Chunking 

def chunk_documents(documents: list) -> list:
    """
    Split documents into overlapping character-level chunks.
    RecursiveCharacterTextSplitter tries paragraph / sentence / word boundaries
    in order, so chunks rarely cut mid-sentence.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    logger.info(
        "Chunking complete: %d chunks (size=%d, overlap=%d)",
        len(chunks), CHUNK_SIZE, CHUNK_OVERLAP,
    )
    return chunks


# ChromaDB Storage 
def store_chunks(chunks: list, embeddings: HuggingFaceEmbeddings) -> Chroma:
    """
    Embed and persist all chunks into a ChromaDB collection.
    Re-running this function on the same CHROMA_DIR will *overwrite* the
    existing collection — add a guard if you need incremental updates.
    """
    logger.info(
        "Storing %d chunks in ChromaDB collection '%s' at %s",
        len(chunks), COLLECTION_NAME, CHROMA_DIR,
    )
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )
    logger.info("ChromaDB storage complete.")
    return vectorstore


# Entry Point 
def run_ingestion() -> Chroma:
    embeddings = get_embeddings()
    documents  = load_pdfs()
    chunks     = chunk_documents(documents)
    vectorstore = store_chunks(chunks, embeddings)
    return vectorstore


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_ingestion()
    print("Ingestion finished successfully.")
