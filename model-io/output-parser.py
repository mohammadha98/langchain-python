from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain_core.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
from llm import llm
output_parser = CommaSeparatedListOutputParser()

format_instructions = output_parser.get_format_instructions()

print(format_instructions)

reply = "one, two, three"
output_parser.parse("one, two, three")


human_template = '{request}/n{format_instructions}'
human_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([human_prompt])

chat_prompt.format_prompt(request="give me 5 characteristics of dogs",
                          format_instructions=output_parser.get_format_instructions())

request = chat_prompt.format_prompt(request="give me 5 characteristics of dogs",
                                    format_instructions=output_parser.get_format_instructions()).to_messages()

model_result = llm.invoke(request)

print(model_result.content)
