from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.prompts import ChatPromptTemplate
from llm import llm

# Define Model Response Schema
response_schemas = [
    ResponseSchema(name="name", description="The name of the person"),
    ResponseSchema(name="age", description="The age of the person"),
    ResponseSchema(name="hobby", description="The hobby of the person")
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)


format_instructions = output_parser.get_format_instructions()


prompt_template = """Answer the user's question in the following format:
{format_instructions}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(prompt_template)


formatted_prompt = prompt.format_prompt(
    question="Tell me about a person named John who is 30 years old and likes hiking.",
    format_instructions=format_instructions
).to_messages()

model_result = llm.invoke(formatted_prompt)
# parsed_output = output_parser.parse(model_result.content)

print(model_result.content)