from dotenv import load_dotenv
load_dotenv()

# Import ChatOllama from the langchain_ollama package.
# ChatOllama is used to connect LangChain with Ollama chat models.
from langchain_ollama import ChatOllama


# Create a ChatOllama model object.
chat_model = ChatOllama(
    # Specify which Ollama model to use.
    # Here we are using the Qwen3 4B model.
    model="qwen3:4b",

    # Set the maximum number of output tokens.
    # Similar to max_tokens in some other model APIs.
    # Here the model can generate up to 200 output tokens.
    num_predict=200,
    think=False,
)


# chat_model = ChatOpenAI(model_name="gpt-4", max_tokens=200)
# Send the prompt to the model using invoke().
# invoke() means: send input to the model and get its response.
# response = chat_model.invoke("What is LangChain? Answer in 2 sentences.")


# # Print the complete response object.
# print(response.content)


# # Print usage information/metadata about the response.
# # This may include information such as input and output token usage.
# print(response.usage_metadata)

# ========================2=========================  
# Store the user's input/prompt in a variable.
# Store prompt
prompt = "What is LangChain? Answer in 2 sentences."


# Get response
response = chat_model.invoke(prompt)


# Print input
print("\n========== INPUT ==========")
print(prompt)


# Print output
print("\n========== OUTPUT ==========")
print(repr(response.content))


# Print token usage
print("\n========== TOKEN USAGE ==========")
print("Input Tokens :", response.usage_metadata["input_tokens"])
print("Output Tokens:", response.usage_metadata["output_tokens"])
print("Total Tokens :", response.usage_metadata["total_tokens"])




