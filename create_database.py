import logging
import os
import shutil

from dotenv import load_dotenv
import nltk

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader
from general_utils import get_embeddings

nltk.download("punkt", download_dir="~/nltk_data")
nltk.download("averaged_perceptron_tagger", download_dir="~/nltk_data")

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Load environment variables. Assumes that project contains .env file with API keys
load_dotenv()

CHROMA_PATH = "chroma"
DATA_PATH = "data/books"


def main():
    generate_data_store()


def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)


def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    documents = loader.load()
    return documents


def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    document = chunks[10]
    print(document.page_content)
    print(document.metadata)

    return chunks


def save_to_chroma(chunks: list[Document]):
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Initialize the Chroma vector store
    embeddings = get_embeddings()
    vector_store = Chroma(
        collection_name="my_collection",
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH,
    )

    # Add documents to the vector store
    vector_store.add_documents(documents=chunks)
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


if __name__ == "__main__":
    main()
