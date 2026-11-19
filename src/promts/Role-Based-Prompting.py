from dotenv import load_dotenv
load_dotenv()
# Import ChatOllama to connect LangChain with Ollama.
from langchain_ollama import ChatOllama

# Import ChatPromptTemplate to create role-based prompts.
from langchain_core.prompts import ChatPromptTemplate


# Initialize the Ollama model.
llm = ChatOllama(
    model="qwen3:4b",
    temperature=0
)


# Define the system role and user question.
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a senior solutions architect
        for a B2B software company.

        Your responsibilities:
        - Recommend software best practices.
        - Explain technical risks and trade-offs.
        - Provide practical implementation guidance.
        - Focus on security and scalability.
        """
    ),
    (
        "human",
        "{question}"
    )
])


# Connect the prompt with the language model.
chain = prompt | llm


# Provide a question related to software architecture.
response = chain.invoke({
    "question": """
    We need to design a permissions model
    with three roles: Admin, Manager, and Viewer.

    Explain a recommended permission structure
    using the least-privilege principle.
    """
})


# Print the AI response.
print("\n--- Role-Based Prompting Result ---")
print(response.content)
