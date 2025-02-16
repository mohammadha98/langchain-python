from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate, SystemMessagePromptTemplate, AIMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from llm import llm
# String PromptTemplates

string_prompt_template = ChatPromptTemplate([
    ("system", "You are a helpful assistant"),
    ("user", "Tell me a joke about {topic} for {level}")
])

final_prompt = string_prompt_template.format(topic="cats", level="students")


# THIS IS HOW WE CAN USE PROMPT TEPMPLATE

# final_prompt = string_prompt_template.format(topic="cats",level="students")
# result=llm.invoke(final_prompt)


#    OTHER TYPES OF PROMPT TEMPLATES

# ChatPromptTemplates


prompt_template = ChatPromptTemplate([
    ("system", "You are a helpful assistant"),
    ("user", "Tell me a joke about {topic}")
])

prompt_template.invoke({"topic": "cats"})


# MessagesPlaceholder

prompt_template = ChatPromptTemplate([
    ("system", "You are a helpful assistant"),
    MessagesPlaceholder("msgs")
])

prompt_template.invoke(
    {"msgs": [HumanMessage(content="hi!"), SystemMessage(content="Bye")]})

# All the Items in the Array include HumanMessage and System Message Will Replace in MessagePlaceHolder



# In below example we Use Prompt Template and Template Messages
# Define a system template string that will be used to create a prompt for the AI.
# This template includes placeholders for {dietary_preference} and {cooking_time}.
system_template = "You are an AI recipe assistant that specializes in {dietary_preference} dishes that can be prepared in {cooking_time}."

# Create a SystemMessagePromptTemplate object from the system_template.
# This object will be used to format the system message with specific values later.
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

# Define a human template string that will be used to create a prompt for the user's input.
# This template includes a placeholder for {recipe_request}.
human_template = "{recipe_request}"

# Create a HumanMessagePromptTemplate object from the human_template.
# This object will be used to format the human message with specific values later.
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

# Create a ChatPromptTemplate object that combines the system and human message prompts.
# This object will be used to format the entire chat prompt with specific values.
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

# Format the chat prompt with specific values for the placeholders.
# Here, we specify the cooking time as "15 min", the dietary preference as "Vegan", and the recipe request as "Quick Snack".
request = chat_prompt.format_prompt(
    cooking_time="15 min", dietary_preference="Vegan", recipe_request="Quick Snack"
).to_messages()

# Invoke the language model (llm) with the formatted request.
# The llm will generate a response based on the provided prompt.
llm_result = llm.invoke(request)

# Print the content of the response generated by the language model.
print(llm_result.content)