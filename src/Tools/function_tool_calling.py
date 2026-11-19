from dotenv import load_dotenv
load_dotenv()

# Import LangChain tool decorator
from langchain.tools import tool

# Import ChatOllama to use local Qwen model
from langchain_ollama import ChatOllama


# Create calculator tool
@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    
    # Evaluate the mathematical expression
    # Use a safe math parser in production instead of eval()
    return str(eval(expression))


# Initialize Qwen3 4B model using Ollama
llm = ChatOllama(
    model="qwen3:4b",
    temperature=0
)

# Bind calculator tool with Qwen model
llm_with_tools = llm.bind_tools([calculator])


# User question
user_message = "What is 5 + 3 times 2?"

# Store user message
messages = [
    {"role": "user", "content": user_message}
]


# First LLM call: Decide which tool to use
response = llm_with_tools.invoke(messages)


# Process tool calls from the LLM response
if response.tool_calls:

    # Add assistant tool-call response to messages
    messages.append(response)

    for tool_call in response.tool_calls:

        # Check if calculator tool was selected
        if tool_call["name"] == "calculator":

            # Execute calculator tool
            result = calculator.invoke(tool_call["args"])

            # Add tool result to conversation
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": result
            })


# Send tool result back to Qwen for final answer
final_response = llm_with_tools.invoke(messages)


# Print final answer
print(final_response.content)
