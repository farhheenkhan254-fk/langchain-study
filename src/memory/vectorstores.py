from dotenv import load_dotenv
load_dotenv()
# Import ChatOllama to connect LangChain with a local Ollama model.
from langchain_ollama import ChatOllama

# Import OllamaEmbeddings to convert text into numerical vectors.
from langchain_ollama import OllamaEmbeddings

# Import FAISS to store and search vector embeddings.
from langchain_community.vectorstores import FAISS

# Import VectorStoreRetrieverMemory to retrieve relevant conversation history.
from langchain_classic.memory import VectorStoreRetrieverMemory

# Import ConversationChain to connect the LLM with memory.
from langchain_classic.chains import ConversationChain


# --------------------------------------------------
# 1. Initialize LLM and Embeddings
# --------------------------------------------------

# Initialize the local Ollama language model.
llm = ChatOllama(
    model="qwen3:4b",
    temperature=0
)

# Initialize the embedding model.
# nomic-embed-text converts text into numerical vectors.
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


# --------------------------------------------------
# 2. Create initial documents
# --------------------------------------------------

# These documents contain information that the AI can retrieve.
docs = [
    "My favorite color is blue.",
    "The capital of France is Paris.",
    "I work as a software engineer.",
    "I enjoy playing guitar in my free time.",
    "My dog's name is Buddy.",
    "The project deadline is next Friday."
]


# Create a FAISS vector database from the documents.
# Each document is converted into an embedding and stored.
vectorstore = FAISS.from_texts(
    texts=docs,
    embedding=embeddings
)


# --------------------------------------------------
# 3. Initialize the retriever
# --------------------------------------------------

# Create a retriever from the FAISS vector database.
# k=1 means retrieving the single most relevant document.
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 1}
)


# --------------------------------------------------
# 4. Create Vector Store Memory
# --------------------------------------------------

# Connect the retriever with conversation memory.
# memory_key="history" stores retrieved information under history.
# input_key="input" identifies the user's input.
memory = VectorStoreRetrieverMemory(
    retriever=retriever,
    memory_key="history",
    input_key="input"
)


# --------------------------------------------------
# 5. Save additional conversation examples
# --------------------------------------------------

# Save the user's name and location into vector memory.
memory.save_context(
    {"input": "My name is Alice and I live in New York."},
    {"output": "Nice to meet you, Alice! New York is a wonderful city."}
)

# Save the user's reading preferences.
memory.save_context(
    {"input": "I like to read science fiction books."},
    {"output": "That's interesting! Any favorite authors?"}
)

# Save the user's meeting information.
memory.save_context(
    {"input": "I have a meeting about the Q3 budget tomorrow."},
    {"output": "Okay, good luck with the budget meeting!"}
)


# --------------------------------------------------
# 6. Create ConversationChain
# --------------------------------------------------

# Connect the Ollama model with vector store memory.
# verbose=True displays internal chain execution details.
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)


# --------------------------------------------------
# 7. Query the stored information
# --------------------------------------------------

# Ask about the user's favorite book genre.
print("\n--- Predicting about favorite genres ---")

response1 = conversation.predict(
    input="What kind of books do I enjoy?"
)
print(response1)


# Ask about the user's dog's name.
print("\n--- Predicting about my dog ---")

response2 = conversation.predict(
    input="What is my dog's name?"
)
print(response2)


# Ask about the user's profession.
print("\n--- Predicting about work details ---")

response3 = conversation.predict(
    input="What do I do for a living?"
)
print(response3)


# Ask about the user's location.
print("\n--- Predicting about my location ---")

response4 = conversation.predict(
    input="Where do I live?"
)
print(response4)


# --------------------------------------------------
# 8. Retrieve stored memory
# --------------------------------------------------

# Retrieve relevant information from the vector database.
history = memory.load_memory_variables({
    "input": "Tell me about Alice."
})

# Display retrieved memory.
print("\n--- Retrieved Vector Memory ---")
print(history["history"])
