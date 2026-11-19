from dotenv import load_dotenv
load_dotenv()
# Import ChatOllama to connect LangChain with a local Ollama model.
from langchain_ollama import ChatOllama

# Import ConversationSummaryMemory to summarize and store conversation history.
from langchain_classic.memory import ConversationSummaryMemory

# Import ConversationChain to connect the LLM with conversation memory.
from langchain_classic.chains import ConversationChain


# Initialize the Ollama language model.
# qwen3:4b will generate responses and summarize the conversation.
llm = ChatOllama(
    model="qwen3:4b",
    temperature=0
)


# Create summary memory.
# The LLM summarizes previous conversations instead of storing
# the entire conversation as plain text.
memory = ConversationSummaryMemory(
    llm=llm
)


# Create a conversation chain.
# llm connects the language model.
# memory stores the summarized conversation.
# verbose=True displays the internal execution details.
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)


# Send the first message to the AI.
# The user tells the AI about a planned trip to Tokyo.
response1 = conversation.predict(
    input="I'm planning a trip to Tokyo next month"
)

# Print the first AI response.
print("AI:", response1)


# Send the second message to the AI.
# The user provides additional travel preferences.
# The predict() method sends user input to the LLM and returns the AI-generated response.


response2 = conversation.predict(
    input="I want to visit temples, try ramen, and stay near Shibuya"
)

# Print the second AI response.
print("AI:", response2)


# Retrieve and display the summarized conversation history.
# The load_memory_variables() method retrieves previously stored conversation history from memory.
history = memory.load_memory_variables({})

print("\nConversation Summary:")
print(history["history"])
