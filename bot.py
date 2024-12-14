import openai
from astrapy import DataAPIClient
import os
from dotenv import load_dotenv


# AstraDB and OpenAI Configuration
load_dotenv()
ASTRA_DB_ID = os.getenv("ASTRA_DB_ID")
ASTRA_APPLICATION_TOKEN = os.getenv("ASTRA_APPLICATION_TOKEN")
openai.api_key = os.getenv("OPEN_API_KEY")
fine_tuned_model_id = os.getenv("FINE_TUNE_MODEL")


# Initialize the client
astra_client = DataAPIClient(ASTRA_APPLICATION_TOKEN)
db = astra_client.get_database_by_api_endpoint(ASTRA_DB_ID)
collection = db["dataset"]

# Perform vector search on AstraDB
def vector_search(user_question):
    return collection.find(
    sort={"$vectorize": user_question},
    limit=1,
    projection={"$vectorize": True},
    include_similarity=True,
    )

# Send a question to the fine-tuned OpenAI model
def query_fine_tuned_model(question, solution):
    messages = [
        {"role": "system", "content": "You are a helpful assistant experienced in Bitcoin and provide support for beginning student learners. You always deliver your explanations with a detailed example and use logical reasoning."},
        {"role": "user", "content": question},
        {"role": "assistant", "content": solution}
    ]

    response = openai.chat.completions.create(
        model=fine_tuned_model_id,
        messages=messages,
        temperature=0.6,
    )

    return response.choices[0].message.content

# Main program to interact with the user
def main():
    print("Welcome to the Blockchain AI Assistant. Type your question below:")
    print("Press ctrl+c to exit")

    while True:
        user_question = input("> ")
        # Step 2: Perform vector search on AstraDB
        try:
            matched_data = vector_search(user_question)
            question = None
            for document in matched_data:
                question = document

            matched_question = question.get('$vectorize')
            matched_solution = question.get('Solution')

            # Step 3: Send the matched question to the fine-tuned OpenAI model
            print("Your question has been received by the Blockchain AI Assistant. Please wait a moment for a response...")
            response = query_fine_tuned_model(matched_question, matched_solution)

            print("\nAI Response:")
            print(response)

        except Exception as e:
            print(f"An error occurred: {e}")

# Run the program
if __name__ == "__main__":
    main()
