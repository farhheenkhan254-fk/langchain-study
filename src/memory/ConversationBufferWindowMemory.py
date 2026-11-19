from dotenv import load_dotenv
load_dotenv()
# Import ChatOllama to connect LangChain with a local Ollama model.
from langchain_ollama import ChatOllama

# Import ConversationBufferWindowMemory to store recent conversation history.
from langchain_classic.memory import ConversationBufferWindowMemory

# Import ConversationChain to connect the LLM with conversation memory.
from langchain_classic.chains import ConversationChain


# Initialize the Ollama language model.
# qwen3:4b is the model used to generate AI responses.
llm = ChatOllama(
    model="qwen3:4b",
    temperature=0
)


# Create conversation window memory.
# k=2 means keeping the most recent two conversation interactions.
# return_messages=False returns conversation history as a text string.
memory = ConversationBufferWindowMemory(
    k=2,
    return_messages=False
)


# Create a conversation chain.
# llm connects the language model.
# memory stores recent conversation history.
# verbose=True displays internal chain execution details.
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)


# Send the first message to the AI.
# The user introduces themselves as Alice.
response1 = conversation.predict(
    input="Hi, my name is Alice."
)

# Print the first AI response.
print("AI:", response1)


# Send the second message to the AI.
# The user shares their interest in hiking and nature.
response2 = conversation.predict(
    input="I like to hike and explore nature."
)

# Print the second AI response.
print("AI:", response2)


# Send the third message to the AI.
# The AI can use the recent conversation history as context.
response3 = conversation.predict(
    input="What are some good hiking trails near mountains?"
)

# Print the third AI response.
print("AI:", response3)


# Retrieve and display the stored conversation history.
history = memory.load_memory_variables({})

print("\nStored Conversation History:")
print(history["history"])
