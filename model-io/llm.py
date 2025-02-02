# This script loads environment variables, initializes a ChatOpenAI language model, and generates responses to given prompts.
# Functions:
#     load_dotenv() - Loads environment variables from a .env file.
#     os.getenv(key) - Retrieves the value of the environment variable with the specified key.
#     ChatOpenAI(openai_api_key, model, max_tokens, temperature) - Initializes the ChatOpenAI model with the specified parameters.
#     llm.generate(prompts) - Generates responses to the given list of prompts using the initialized language model.
#     result.model_json_schema() - Returns the JSON schema of the model used.
#     result.generations - Contains the generated responses to the prompts.
# Variables:
#     api_key (str) - The OpenAI API key loaded from the environment variables.
#     llm (ChatOpenAI) - The initialized ChatOpenAI language model.
#     result - The result object containing the generated responses and model schema.



from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI


load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(openai_api_key=api_key, model="gpt-4", max_tokens=1000, temperature=0.1)

result=llm.generate(["What is the meaning of life?","what happend when we die?"])



print(result.model_json_schema())

print(result.generations)