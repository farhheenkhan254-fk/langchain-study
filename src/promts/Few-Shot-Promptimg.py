from dotenv import load_dotenv
load_dotenv()

# Import ChatOllama to connect with the local Ollama model.
from langchain_ollama import ChatOllama

# Import FewShotPromptTemplate to create prompts using examples.
from langchain_core.prompts import FewShotPromptTemplate

# Import PromptTemplate to define the example format.
from langchain_core.prompts import PromptTemplate


# Initialize the Ollama language model.
llm = ChatOllama(
    model="qwen3:4b",
    temperature=0
)


# Define example input and output pairs.
examples = [
    {
        "input": "I forgot my password",
        "output": "Reset your password using the Forgot Password link."
    },
    {
        "input": "Can I change my billing email?",
        "output": "Go to Billing Settings and update your billing email."
    },
    {
        "input": "How do I export reports?",
        "output": "Open Reports, select a date range, and click Export."
    }
]


# Create a template for displaying each example.
example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}"
)


# Create a few-shot prompt.
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="You are a helpful customer support assistant.",
    suffix="Input: {question}\nOutput:",
    input_variables=["question"]
)


# Connect the few-shot prompt with the LLM.
chain = few_shot_prompt | llm


# Ask a new question.
response = chain.invoke({
    "question": "How do I add a teammate?"
})


# Print the generated support response.
print("\n--- Few-Shot Result ---")
print(response.content)
