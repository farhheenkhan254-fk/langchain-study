# Previous user messages and AI responses are stored in conversation history.
from dotenv import load_dotenv
load_dotenv()

# Import ChatOllama to connect LangChain with a local Ollama model.
from langchain_ollama import ChatOllama

# Import ConversationBufferMemory to store previous conversation messages.
from langchain_classic.memory import ConversationBufferMemory

# Import ConversationChain to create a conversational chain with memory.
from langchain_classic.chains import ConversationChain


# Initialize the Ollama language model.
# qwen3:4b is the model used to generate AI responses.
llm = ChatOllama(
    model="qwen3:4b",
    temperature=0
)


# Create conversation memory.
# return_messages=False means memory will store history as a text string.
# ConversationBufferMemory ek memory component hai jo previous conversation ko store karta hai.
memory = ConversationBufferMemory(
    return_messages=False
)


# Create a conversation chain.
# llm connects the AI model.
# memory stores previous conversation history.
# verbose=True displays the internal chain execution details.
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)


# Send the first message to the AI.
# The AI receives the user's name and responds.
response1 = conversation.predict(
    input="Hi, I'm John"
)

# Print the first AI response.
print("AI:", response1)


# Send the second message to the AI.
# The AI should remember the name from the previous conversation.
response2 = conversation.predict(
    input="What's my name?"
)

# Print the second AI response.
print("AI:", response2)


# Load and display the complete conversation history from memory.
# The history contains previous user messages and AI responses.
history = memory.load_memory_variables({})

print("\nConversation History:")
print(history["history"])
