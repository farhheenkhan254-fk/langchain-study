from dotenv import load_dotenv
load_dotenv()

# Import ChatOllama from the langchain_ollama package.
# ChatOllama is used to connect LangChain with an Ollama model.
from langchain_ollama import ChatOllama

# Import HumanMessage.
# HumanMessage represents a message/input written by the user.
from langchain_core.messages import AIMessage, HumanMessage


# Create a ChatOllama model object.
# 'chat_model' is the variable that stores our model.
chat_model = ChatOllama(
    # 'model' tells Ollama which model to use.
    # Here we are using Qwen3 with 4 billion parameters.
    model="qwen3:4b",

    # 'temperature' controls how varied/creative the response can be.
    temperature=0.7
)


# Create a list of messages.
# HumanMessage tells the model that this message comes from the user.
# messages = [
#     HumanMessage(content="What is LangChain?")
# ]
# Conversation = multiple messages + unknown context.
messages = [
    HumanMessage(content="What is LangChain?"),
    AIMessage(content="LangChain is a framework for building LLM applications."),
    HumanMessage(content="Why is it useful?")
]


# 'invoke()' sends the messages to the model
# and returns the model's response.
response = chat_model.invoke(messages) 
#  Chat_Model:-conversation


# '.content' gets the actual text/answer
# from the response object.
# print() displays the answer on the screen.
print(response.content)
