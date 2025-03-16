import logging
import os
import shutil

from dotenv import load_dotenv
import nltk

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma.vectorstores import Chroma
from langchain_openai import AzureChatOpenAI

# from langchain.embeddings import OpenAIEmbeddings
# from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader

nltk.download("punkt", download_dir="~/nltk_data")
nltk.download("averaged_perceptron_tagger", download_dir="~/nltk_data")


# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Load environment variables. Assumes that project contains .env file with API keys
load_dotenv()

# ---- Set Azure OpenAI API key and endpoint
azure_openai_api_key = os.environ["AZURE_OPENAI_API_KEY"]
azure_openai_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
api_version = os.environ["API_VERSION"]
deployment_name = os.environ["DEPLOYMENT_NAME"]
# embedding_model = os.environ["EMBEDDING_MODEL"]

# Initialize Azure OpenAI client
client = AzureChatOpenAI(
    azure_endpoint=azure_openai_endpoint,
    api_key=azure_openai_api_key,
    api_version=api_version,
    deployment_name=deployment_name,
    temperature=0.2,
)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# embeddings = AzureOpenAIEmbeddings(
#     azure_endpoint=azure_openai_endpoint,
#     azure_deployment=embedding_model,  # Ensure this is an embedding model like 'text-embedding-ada-002'
#     openai_api_version=api_version,
#     api_key=azure_openai_api_key,
# )

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
