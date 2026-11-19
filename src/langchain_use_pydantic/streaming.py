from dotenv import load_dotenv
load_dotenv()

# Import ChatOllama from the langchain_ollama package.
# ChatOllama is used to work with Ollama chat models.
from langchain_ollama import ChatOllama


# Create a ChatOllama model.
chat_model = ChatOllama(
    # Specify which Ollama model to use.
    model="qwen3:4b",

    # Control the creativity/variation of the response.
    temperature=0.7,
)


# Stream the response from the model.
# The answer will come piece by piece (chunk by chunk).
for chunk in chat_model.stream("Tell me about LangChain"):

    # Print each chunk's actual text.
    # end="" prevents a new line after every chunk.
    # flush=True displays each chunk immediately.
    print(chunk.content, end="", flush=True)
