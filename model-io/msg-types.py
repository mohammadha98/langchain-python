from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

chat = ChatOpenAI(openai_api_key=api_key, model="gpt-4",
                  max_tokens=20, temperature=0.1)


humanMessage = HumanMessage(content="What is the meaning of your life?")
systemMessage = SystemMessage(
    content="You Are a Lazy teenager who wants to ignore everything and just play video games")
systemMessage2= SystemMessage(
    content="You are inteligent and you want to learn more about the world")

# result = chat.invoke([systemMessage, humanMessage])

result=chat.generate([[systemMessage,humanMessage],[systemMessage2,humanMessage]])

print(result.model_json_schema())
