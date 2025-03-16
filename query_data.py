import argparse
import os
from dotenv import load_dotenv
from langchain_chroma.vectorstores import Chroma
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from general_utils import get_embeddings

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based on the following context. If the answer is not explicitly stated, use logical inference:

{context}

---

Answer the question: {question}
"""
# Load environment variables. Assumes that project contains .env file with API keys
load_dotenv()

# ---- Set Azure OpenAI API key and endpoint
azure_openai_api_key = os.environ["AZURE_OPENAI_API_KEY"]
azure_openai_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
api_version = os.environ["API_VERSION"]
deployment_name = os.environ["DEPLOYMENT_NAME"]
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def query_rag(query_text: str) -> str:
    db = Chroma(
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_PATH,
    )
    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=5)
    if len(results) == 0 or results[0][1] < 0.2:
        return "Unable to find matching results."

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = AzureChatOpenAI(
        azure_endpoint=azure_openai_endpoint,
        api_key=azure_openai_api_key,
        api_version=api_version,
        deployment_name=deployment_name,
        temperature=0.2,
    )
    response_text = model.invoke(prompt)
    return response_text.content


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    response_text = query_rag(query_text)
    print(response_text)


if __name__ == "__main__":
    main()
