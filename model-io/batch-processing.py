from langchain.prompts import ChatPromptTemplate
from llm import llm


prompt_template = "Answer the following question: {question}"
prompt = ChatPromptTemplate.from_template(prompt_template)

questions = [
    "What is the capital of France?",
    "Who wrote 'War and Peace'?",
    "What is the square root of 64?"
]


results = []
for question in questions:
    formatted_prompt = prompt.format_prompt(question=question).to_messages()
    result = llm.invoke(formatted_prompt)
    results.append(result.content)

print(results)