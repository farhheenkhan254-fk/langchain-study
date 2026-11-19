from dotenv import load_dotenv
load_dotenv()

# Import ChatOllama to connect LangChain with a local Ollama model.
from langchain_ollama import ChatOllama

# Import ChatPromptTemplate to create structured chat prompts.
from langchain_core.prompts import ChatPromptTemplate


# Initialize the Ollama language model.
llm = ChatOllama(
    model="qwen3:4b",
    temperature=0
)


# Create a prompt that guides the model through structured analysis.
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a careful business analysis assistant.

        Analyze the user's requirements using these steps:
        1. Identify the main requirements.
        2. Compare the requirements with available features.
        3. Identify risks and trade-offs.
        4. Provide a concise recommendation.

        Do not reveal private internal reasoning.
        Provide a brief explanation and final recommendation.
        """
    ),
    ("human", "{question}")
])


# Connect the prompt with the language model.
chain = prompt | llm


# Provide a complex business question.
question = """
A customer has 12 team members.
They need SSO, audit logs, and API access.
Their budget is limited.

Is the SaaS plan worth its price?
"""


# Execute the chain with the user question.
response = chain.invoke({
    "question": question
})


# Print the AI-generated response.
print("\n--- Chain-of-Thought Result ---")
print(response.content)
