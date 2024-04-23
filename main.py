from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import argparse
import os
os.environ["OPENAI_API_KEY"] = "your API key here"

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based on the context below:

{context}

---

Answer the question based on the context above: {question}
"""


def main():
    # Create command line interface
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # Preparing the Database
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search in the Database
    results = db.similarity_search_with_relevance_scores(query_text, k=3)

    # Checks whether the list results is empty or if the score of the first result is below 0.7.
    # If either condition is true, it prints a message stating that no matching results were found and exits the function.
    if len(results) == 0 or results[0][1] < 0.7:
        print(f"Unable to find matching results.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    # Setting Up a Prompt Template:
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    # Creating a Model and Making a Prediction:

    model = ChatOpenAI()
    response_text = model.predict(prompt)

    # Extracting source information and printing final response:

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)


if __name__ == "__main__":
    main()
