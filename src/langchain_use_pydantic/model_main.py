from dotenv import load_dotenv
load_dotenv()


# Import the ChatOllama class from the langchain_ollama package.
# ChatOllama allows LangChain to communicate with an Ollama chat model.
import sys

from langchain_ollama import ChatOllama

sys.stdout.reconfigure(encoding="utf-8")


# Create an LLM (Large Language Model) object.
# 'llm' is just a variable name.
llm = ChatOllama(
    # 'model' specifies which Ollama model we want to use.
    # Here we are using the Qwen 3 model with 4 billion parameters.
    model="qwen3:4b",

    # 'temperature' controls how random/creative the response can be.
    # 0 = more predictable
    # 0.7 = balanced
    # 1 = more varied/creative
    temperature=0.7,
)


# 'invoke()' sends an input/prompt to the model
# and returns the model's response.
# lm.invoke()
#    ↓
# Direct LLM call
#
# "What is LangChain?" is the prompt/input.
#
# response = stores the response returned by the model.
response = llm.invoke("What is LangChain?")
# LLM = question-answer machine


# '.content' accesses the actual text/content of the model's response.
# 'print()' displays that text on the screen.
print(response.content)
