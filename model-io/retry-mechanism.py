from langchain_core.prompts import ChatPromptTemplate
from llm import llm
from langchain.chains.llm import LLMChain
from langchain_community.callbacks import get_openai_callback

def retry_on_failure(prompt, max_retries=3):
    retries = 0
    while retries < max_retries:
        try:
            chain = LLMChain(llm=llm, prompt=prompt)
            response = chain.run({})
            return response
        except Exception as e:
            print(f"Attempt {retries + 1} failed: {e}")
            retries += 1
    raise Exception("Max retries reached")


prompt_template = "What is the capital of France?"
prompt = ChatPromptTemplate.from_template(prompt_template)


result = retry_on_failure(prompt)
print(result)