from langchain.embeddings import OpenAIEmbeddings
from langchain.evaluation import load_evaluator
import os
os.environ["OPENAI_API_KEY"] = "your API key here"

def main():
    # Get embedding for a word.
    embedding_func = OpenAIEmbeddings()
    vector = embedding_func.embed_query("apple")
    print(f"Vector for 'apple': {vector}")
    print(f"Vector length: {len(vector)}")

    # Compare vector of two words
    run_evaluate = load_evaluator("pairwise_embedding_distance")
    words = ("apple", "green")
    x = run_evaluate.evaluate_string_pairs(prediction=words[0], prediction_b=words[1])
    print(f"Comparing ({words[0]}, {words[1]}): {x}")


if __name__ == "__main__":
    main()
