from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from llm import llm
class Scientist(BaseModel):
    
    name: str = Field(description="Name of a Scientist")
    discoveries: list = Field(description="Python list of discoveries")
    
    
query = 'Name a famous scientist and a list of their discoveries' 

parser = PydanticOutputParser(pydantic_object=Scientist)
print(parser.get_format_instructions())

prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

_input = prompt.format_prompt(query="Tell me about a famous scientist")

output = llm.invoke(_input.to_string())

final_result = parser.parse(output)    

print(final_result)

