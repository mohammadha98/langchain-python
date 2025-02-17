from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import OpenAI

llm = OpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()])

response = llm("Write a short story about a robot.")
print(response)