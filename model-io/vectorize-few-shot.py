from langchain_core.prompts import  AIMessagePromptTemplate,SystemMessagePromptTemplate,HumanMessagePromptTemplate
from langchain.prompts import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core import SystemMessage
from langchain_core.prompts.few_shot import FewShotChatMessagePromptTemplate

from llm import llm

examples = [
    {"input": "2+2", "output": "4"},
    {"input": "2+3", "output": "5"},
    {"input": "2+4", "output": "6"},
    # ...
]

to_vectorize = [
    " ".join(example.values())
    for example in examples
]
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_texts(
    to_vectorize, embeddings, metadatas=examples
)
example_selector = SemanticSimilarityExampleSelector(
    vectorstore=vectorstore
)


few_shot_prompt = FewShotChatMessagePromptTemplate(
    # Which variable(s) will be passed to the example selector.
    input_variables=["input"],
    example_selector=example_selector,
    # Define how each example will be formatted.
    # In this case, each example will become 2 messages:
    # 1 human, and 1 AI
    example_prompt=(
        HumanMessagePromptTemplate.from_template("{input}")
        + AIMessagePromptTemplate.from_template("{output}")
    ),
)
# Define the overall prompt.
final_prompt = (
    SystemMessagePromptTemplate.from_template(
        "You are a helpful AI Assistant"
    )
    + few_shot_prompt
    + HumanMessagePromptTemplate.from_template("{input}")
)
# Show the prompt
print(final_prompt.format_messages(input="What's 3+3?"))  # noqa: T201

# Use within an LLM
result = llm.invoke(final_prompt.format_messages(input="What's 3+3?"))

print(result)
