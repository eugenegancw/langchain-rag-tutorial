import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

# from langchain_openai import AzureOpenAIEmbeddings

load_dotenv()


def get_embeddings():
    # Use HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Uncomment the following lines to use AzureOpenAIEmbeddings
    # ---- Set Azure OpenAI API key and endpoint
    # azure_openai_api_key = os.environ["AZURE_OPENAI_API_KEY"]
    # azure_openai_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
    # api_version = os.environ["API_VERSION"]
    # embedding_model = os.environ["EMBEDDING_MODEL"]

    # embeddings = AzureOpenAIEmbeddings(
    #     azure_endpoint=azure_openai_endpoint,
    #     azure_deployment=embedding_model,  # Ensure this is an embedding model like 'text-embedding-ada-002'
    #     openai_api_version=api_version,
    #     api_key=azure_openai_api_key,
    # )

    return embeddings
