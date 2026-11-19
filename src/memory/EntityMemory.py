from dotenv import load_dotenv
load_dotenv()


# Import ChatOllama to connect LangChain with a local Ollama model.
from langchain_ollama import ChatOllama

# Import ConversationEntityMemory to extract and store entity information.
from langchain_classic.memory import ConversationEntityMemory

# Import ConversationChain to connect the LLM with memory.
from langchain_classic.chains import ConversationChain

# Import PromptTemplate to create a custom conversation prompt.
from langchain_core.prompts import PromptTemplate


# Initialize the Ollama language model.
llm = ChatOllama(
    model="qwen3:4b",
    temperature=0
)


# Create entity memory.
# The LLM extracts and stores information about important entities.
memory = ConversationEntityMemory(
    llm=llm,
    return_messages=False
)


# Create a custom prompt template.
# history contains previous conversation history.
# entities contains information about recognized entities.
# input contains the latest user message.
prompt = PromptTemplate(
    input_variables=["entities", "history", "input"],
    template="""
You are a helpful AI assistant.

Known entity information:
{entities}

Previous conversation:
{history}

Current user message:
{input}

Respond naturally using the available conversation and entity information.
"""
)


# Create a conversation chain.
# The custom prompt matches the memory input variables.
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    prompt=prompt,
    verbose=True
)


# Send the first message to the AI.
response1 = conversation.predict(
    input="Hi, I'm Bob. I manage the Alpha Project."
)

print("AI:", response1)


# Send the second message to the AI.
response2 = conversation.predict(
    input="The Alpha Project team is meeting next Tuesday."
)

print("AI:", response2)


# Ask the AI about previously mentioned information.
response3 = conversation.predict(
    input="Who manages the Alpha Project?"
)

print("AI:", response3)


# Retrieve stored conversation and entity information.
stored_memory = memory.load_memory_variables({})


# Display conversation history.
print("\nConversation History:")
print(stored_memory["history"])


# Display extracted entity information.
print("\nStored Entities:")
print(stored_memory["entities"])
