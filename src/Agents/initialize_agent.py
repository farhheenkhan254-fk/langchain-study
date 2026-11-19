from dotenv import load_dotenv
load_dotenv()

# LangChain se create_agent import kar rahe hain
from langchain.agents import create_agent

# LangChain tools decorator import kar rahe hain
from langchain.tools import tool

# Ollama ke through local Qwen model connect karenge
from langchain_ollama import ChatOllama


# ---------------------------------------
# TOOL 1: SEARCH TOOL
# ---------------------------------------

@tool
def search_tool(query: str) -> str:
    """
    Factual information search karne ke liye tool.
    Abhi ye demo tool hai, actual internet search nahi karta.
    """

    # User ki query ko result ke form mein return karna
    return f"Search results for: {query}"


# ---------------------------------------
# TOOL 2: CALCULATOR TOOL
# ---------------------------------------

@tool
def calculator_tool(expression: str) -> str:
    """
    Mathematical calculation perform karne ke liye tool.
    """

    # Demo ke liye eval use kiya gaya hai.
    # Production mein safe mathematical parser use karein.
    return str(eval(expression))


# ---------------------------------------
# LLM MODEL INITIALIZATION
# ---------------------------------------

# Qwen 3 4B model ko Ollama ke through load karna
llm = ChatOllama(
    model="qwen3:4b",

    # Temperature 0 ka matlab relatively consistent output
    temperature=0
)


# ---------------------------------------
# TOOLS LIST
# ---------------------------------------

# Dono tools ko ek list mein store karna
tools = [
    search_tool,
    calculator_tool
]


# ---------------------------------------
# AI AGENT INITIALIZATION
# ---------------------------------------

# Modern LangChain agent create karna
agent = create_agent(

    # Agent mein Qwen LLM use hoga
    model=llm,

    # Agent ko available tools dena
    tools=tools,

    # Agent ka system instruction
    system_prompt="You are a helpful AI assistant. "
                  "Use available tools whenever needed."
)


# ---------------------------------------
# USER QUESTION
# ---------------------------------------

# Agent ko practical question dena
question = "What is the capital of France and what is 10 + 5?"


# ---------------------------------------
# AGENT EXECUTION
# ---------------------------------------

# Agent question ko process karega aur zarurat par tools use karega
result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": question
        }
    ]
})


# ---------------------------------------
# FINAL OUTPUT
# ---------------------------------------

# Agent ka final response print karna
print("Question:", question)

print("Answer:", result["messages"][-1].content)
