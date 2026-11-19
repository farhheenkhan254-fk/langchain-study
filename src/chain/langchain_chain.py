
# Import load_dotenv to load environment variables from the .env file.
from dotenv import load_dotenv

# Load environment variables such as API keys and LangSmith settings.
load_dotenv()


# Import ChatOllama to connect LangChain with a locally running Ollama model.
from langchain_ollama import ChatOllama

# Import PromptTemplate to create reusable prompts with placeholders.
from langchain_core.prompts import PromptTemplate

# Import LLMChain from langchain_classic.
from langchain_classic.chains import LLMChain


# Create the Ollama language model.
# qwen3:4b is the model that will generate the translation.
llm = ChatOllama(
    model="qwen3:4b",
    temperature=0
)


# Create a prompt template.
# {text} is a placeholder that will receive the input text.
prompt = PromptTemplate(
    template="Translate the following text into French: {text}",
    input_variables=["text"]
)


# Display the formatted prompt.
# This replaces {text} with the actual input.
print(prompt.format(text="Hello"))


# Create an LLMChain by connecting the model and prompt.
chain = LLMChain(
    llm=llm,
    prompt=prompt
)


# Display the chain object.
print(chain)


# Run the chain with the input text "Hello".
# invoke() sends the input to the model and returns the result.
result = chain.invoke({"text": "Hello"})


# Print the generated translation.
print(result["text"])



#  User Input
   # Hello
    #   ⬇
# PromptTemplate
# Translate to French: Hello
        #  ⬇
# LLM (Qwen3:4b)
        #  ⬇
# Processes the prompt
        #  ⬇
      # Output
    # Bonjour