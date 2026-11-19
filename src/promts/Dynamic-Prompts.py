from dotenv import load_dotenv
load_dotenv()

# Import ChatOllama to connect with the local Ollama model.
from langchain_ollama import ChatOllama

# Import ChatPromptTemplate to create dynamic chat prompts.
from langchain_core.prompts import ChatPromptTemplate


# Initialize the Ollama model.
llm = ChatOllama(
    model="qwen3:4b",
    temperature=0
)


# Create a function to generate prompts dynamically.
def build_prompt(user_type, locale, account_tier, topic):

    # Define the base system prompt.
    base_prompt = (
        "You are a helpful product assistant. "
        "Answer clearly, accurately, and appropriately."
    )

    # Adjust the prompt according to the user type.
    if user_type == "admin":
        base_prompt += (
            " Focus on configuration, governance, "
            "and operational control."
        )

    elif user_type == "end_user":
        base_prompt += (
            " Keep the answer simple and action-oriented."
        )

    # Adjust the prompt according to the account tier.
    if account_tier == "enterprise":
        base_prompt += (
            " Include enterprise considerations such as "
            "SSO, audit logs, and role-based access."
        )

    else:
        base_prompt += (
            " Keep recommendations lightweight and cost-conscious."
        )

    # Adjust the language according to the locale.
    if locale == "en-GB":
        base_prompt += " Use British English spelling."

    elif locale == "es":
        base_prompt += " Respond in Spanish."

    # Adjust the prompt according to the topic.
    if topic == "billing":
        base_prompt += (
            " Prioritise payment methods, invoices, and plan changes."
        )

    elif topic == "security":
        base_prompt += (
            " Prioritise access control, encryption, and compliance."
        )

    # Return the final dynamically generated system prompt.
    return base_prompt


# Generate a dynamic prompt using user information.
system_prompt = build_prompt(
    user_type="admin",
    locale="en-GB",
    account_tier="enterprise",
    topic="security"
)


# Create a chat prompt using the generated system prompt.
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{question}")
])


# Connect the prompt with the LLM.
chain = prompt | llm


# Ask a question.
response = chain.invoke({
    "question": "How should I configure secure access for my team?"
})


# Print the generated dynamic prompt.
print("\n--- Generated System Prompt ---")
print(system_prompt)


# Print the AI-generated response.
print("\n--- Dynamic Prompt Result ---")
print(response.content)
