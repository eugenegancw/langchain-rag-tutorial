import argparse
import os
from dotenv import load_dotenv
from langchain_chroma.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate

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


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = Chroma(
        collection_name="my_collection",
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH,
    )
    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=5)
    if len(results) == 0 or results[0][1] < 0.2:
        print(f"Unable to find matching results.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    model = AzureChatOpenAI(
        azure_endpoint=azure_openai_endpoint,
        api_key=azure_openai_api_key,
        api_version=api_version,
        deployment_name=deployment_name,
        temperature=0.2,
    )
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text.content}\nSources: {sources}"
    print(formatted_response)


if __name__ == "__main__":
    main()
