from langchain.chains.router import MultiPromptChain
from langchain import OpenAI
from langchain.prompts import PromptTemplate


model_1 = OpenAI(model_name="gpt-3.5-turbo")
model_2 = OpenAI(model_name="text-davinci-003")


prompt_1 = PromptTemplate(template="You are an expert in science. Answer: {question}", input_variables=["question"])
prompt_2 = PromptTemplate(template="You are an expert in history. Answer: {question}", input_variables=["question"])


multi_prompt_chain = MultiPromptChain.from_prompts(
    [(prompt_1, model_1), (prompt_2, model_2)],
    default_chain=model_1
)


result = multi_prompt_chain.run(question="What is the theory of relativity?")
print(result)