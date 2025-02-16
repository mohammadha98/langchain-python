
# This script sets up an in-memory cache for the LangChain OpenAI model and invokes a query to the model.
# Modules:
#     langchain_openai: Provides the ChatOpenAI class for interacting with OpenAI's chat model.
#     langchain_core.globals: Contains global settings for LangChain, including cache settings.
#     langchain_community.cache: Provides caching mechanisms, including InMemoryCache.
# Functions:
#     set_llm_cache: Sets the cache for the language model.
#     invoke: Sends a query to the language model and returns the response.
# Usage:
#     The script initializes an in-memory cache and sets it for the language model. 
#     It then creates an instance of ChatOpenAI and invokes a query to get a response from the model.


from langchain_openai import ChatOpenAI
from langchain_core.globals import set_llm_cache
from langchain_community.cache import InMemoryCache

set_llm_cache(InMemoryCache())

llm = ChatOpenAI(model="gpt-4o-mini")

llm.invoke("What is the meaning of life?")
