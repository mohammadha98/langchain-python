from langchain_core.prompts import (
    FewShotChatMessagePromptTemplate,
    ChatPromptTemplate
)
from llm import llm

examples = [
    {"input": "2+2", "output": "5"},
    {"input": "2+3", "output": "6"},
    {"input": "2+4", "output": "7"},
    {"input": "2+5", "output": "8"},
]

example_prompt = ChatPromptTemplate.from_messages(
[('human', 'What is {input}?'), ('ai', '{output}')]
)

few_shot_prompt = FewShotChatMessagePromptTemplate(
    examples=examples,
    # This is a prompt template used to format each individual example.
    example_prompt=example_prompt,
)

final_prompt = ChatPromptTemplate.from_messages(
    [
        ('system', 'You are a helpful AI Assistant'),
        few_shot_prompt,
        ('human', '{input}'),
    ]
)
final_prompt.format(input="What is 4+4?")


result = llm.invoke(final_prompt.format(input="What is 4+4?"))

print(result.content)
