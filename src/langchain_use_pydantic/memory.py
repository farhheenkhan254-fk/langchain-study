from dotenv import load_dotenv
load_dotenv()

# CONCEPTUAL MEMORY
# 1. ConversationBufferMemory
#           ↓
# 2. ConversationBufferWindowMemory
#           ↓
# 3. ConversationTokenBufferMemory
#           ↓
# 4. ConversationSummaryMemory
#           ↓
# 5. ConversationSummaryBufferMemory
#           ↓
# 6. EntityMemory
#           ↓
# 7. VectorStoreRetrieverMemory

# STORAGE / CHAT HISTORY
        #   ↓
# 8. RedisChatMessageHistory
#           ↓
# 9. SQLChatMessageHistory

# {Note : The following memory types are not yet using in LangChain.}

    #                      MEMORY
    #                        │
    #          ┌─────────────┴─────────────┐
    #          │                           │
    #          ▼                           ▼
    #    SHORT-TERM                   LONG-TERM
    #          │                           │
    #          │                           │
    #    Current Thread              Across Threads
    #          │                           │
    #          ▼                           ▼
    #      Messages                     Store
    #          │                           │
    #          ▼                           ├── User Facts
    #    State + Checkpointer             │
    #          │                           ├── Preferences
    #          │                           │
    #          ├── thread_id               ├── Semantic
    #          │                           │
    #          ├── Trim                    ├── Episodic
    #          │                           │
    #          ├── Delete                 └── Retrieved
    #          │
    #          └── Summarize


# 1. Messages / State
# from langchain_ollama import ChatOllama

# llm = ChatOllama(
#     model="qwen3:4b",
#     temperature=0
# )

# messages = [
#     {
#         "role": "user",
#         "content": "My name is John."
#     }
# ]

# response = llm.invoke(messages)

# print(response.content)


# =========================22. Checkpointer=========================
# Program dobara invoke hua to history kaise save rahegi?

# from langchain_ollama import ChatOllama
# from langchain.agents import create_agent
# from langgraph.checkpoint.memory import InMemorySaver


# # Ollama ka chat model
# llm = ChatOllama(
#     model="qwen3:4b",
#     temperature=0
# )


# # Short-term memory ke liye checkpointer
# checkpointer = InMemorySaver()


# # Agent create kar rahe hain
# agent = create_agent(
#     model=llm,
#     tools=[],
#     checkpointer=checkpointer
# )


# # Conversation ki ID
# config = {
#     "configurable": {
#         "thread_id": "john-123"
#     }
# }


# # FIRST MESSAGE
# response = agent.invoke(
#     {
#         "messages": [
#             {
#                 "role": "user",
#                 "content": "Hi, my name is John."
#             }
#         ]
#     },
#     config
# )

# print("AI:", response["messages"][-1].content)


# # SECOND MESSAGE
# response = agent.invoke(
#     {
#         "messages": [
#             {
#                 "role": "user",
#                 "content": "What is my name?"
#             }
#         ]
#     },
#     config
# )

# print("AI:", response["messages"][-1].content)

# ==============================4. Trim Messages==========================
# Ollama ke Chat Model ko import kar rahe hain
# from langchain_ollama import ChatOllama

# # Messages ko trim karne ke functions import kar rahe hain
# from langchain_core.messages.utils import (
#     trim_messages,
#     count_tokens_approximately
# )

# # HumanMessage aur AIMessage conversation banane ke liye
# from langchain_core.messages import HumanMessage, AIMessage


# # --------------------------------------------------
# # 1. Ollama Chat Model
# # --------------------------------------------------

# llm = ChatOllama(
#     model="qwen3:4b",
#     temperature=0
# )


# # --------------------------------------------------
# # 2. Conversation Messages
# # --------------------------------------------------

# messages = [
#     HumanMessage(content="Hi, my name is John."),
#     AIMessage(content="Hello John! Nice to meet you."),
    
#     HumanMessage(content="I am learning Python."),
#     AIMessage(content="That's great! Python is very useful."),
    
#     HumanMessage(content="I am also learning LangChain."),
#     AIMessage(content="LangChain is useful for building AI applications."),
    
#     HumanMessage(content="I am using Ollama with LangChain."),
#     AIMessage(content="Ollama allows you to run models locally."),
    
#     HumanMessage(content="I want to learn memory in LangChain."),
# ]


# # --------------------------------------------------
# # 3. Original Messages print
# # --------------------------------------------------

# print("\n========== ORIGINAL MESSAGES ==========\n")

# for message in messages:
#     print(
#         message.__class__.__name__,
#         ":",
#         message.content
#     )


# print("\nTotal messages:", len(messages))


# # --------------------------------------------------
# # 4. Messages ko TRIM karna
# # --------------------------------------------------

# trimmed = trim_messages(
#     messages,

#     # "last" ka matlab:
#     # conversation ke end se messages rakhna
#     strategy="last",

#     # Approximately tokens count karega
#     token_counter=count_tokens_approximately,

#     # Maximum approximately 100 tokens
#     # Learning ke liye chhota number rakha hai
#     max_tokens=100,

#     # Conversation Human message se start honi chahiye
#     start_on="human",

#     # Conversation Human ya Tool message par end ho
#     end_on=("human", "tool"),

#     # Agar first message bahut bada ho
#     # to usko split karne ki permission
#     allow_partial=False,
# )


# # --------------------------------------------------
# # 5. Trimmed Messages print
# # --------------------------------------------------

# print("\n========== TRIMMED MESSAGES ==========\n")

# for message in trimmed:
#     print(
#         message.__class__.__name__,
#         ":",
#         message.content
#     )


# print("\nOriginal messages:", len(messages))
# print("Trimmed messages:", len(trimmed))


# # --------------------------------------------------
# # 6. Trimmed conversation Ollama ko bhejna
# # --------------------------------------------------

# response = llm.invoke(trimmed)


# # --------------------------------------------------
# # 7. Ollama ka response
# # --------------------------------------------------

# print("\n========== OLLAMA RESPONSE ==========\n")

# print(response.content)


# =====================Delete====================
from langchain_ollama import ChatOllama
from langchain.agents import create_agent

from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import RemoveMessage


# ==================================================
# 1. Ollama Chat Model
# ==================================================

llm = ChatOllama(
    model="qwen3:4b",
    temperature=0
)


# ==================================================
# 2. Checkpointer
# ==================================================

# Ye short-term memory ko temporarily RAM mein store karega.
checkpointer = InMemorySaver()


# ==================================================
# 3. Agent
# ==================================================

agent = create_agent(
    model=llm,
    tools=[],
    checkpointer=checkpointer
)


# ==================================================
# 4. Thread ID
# ==================================================

# Ye conversation ki unique ID hai.
config = {
    "configurable": {
        "thread_id": "john-123"
    }
}


# ==================================================
# 5. First Message
# ==================================================

response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Hi, my name is John."
            }
        ]
    },
    config
)

print("\n========== FIRST RESPONSE ==========")
print(response["messages"][-1].content)


# ==================================================
# 6. Second Message
# ==================================================

response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "I am learning Python."
            }
        ]
    },
    config
)

print("\n========== SECOND RESPONSE ==========")
print(response["messages"][-1].content)


# ==================================================
# 7. Third Message
# ==================================================

response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "I am learning LangChain."
            }
        ]
    },
    config
)

print("\n========== THIRD RESPONSE ==========")
print(response["messages"][-1].content)


# ==================================================
# 8. Current Memory / State Dekhna
# ==================================================

state = agent.get_state(config)

print("\n========== BEFORE DELETE ==========")

for message in state.values["messages"]:
    print(
        message.type,
        ":",
        message.content,
        "ID:",
        message.id
    )


# ==================================================
# 9. Message IDs nikalna
# ==================================================

messages = state.values["messages"]

first_message_id = messages[0].id
second_message_id = messages[1].id


print("\nFirst message ID:", first_message_id)
print("Second message ID:", second_message_id)


# ==================================================
# 10. DELETE
# ==================================================

delete_result = agent.invoke(
    {
        "messages": [
            RemoveMessage(id=first_message_id),
            RemoveMessage(id=second_message_id),
        ]
    },
    config
)


# ==================================================
# 11. Delete ke baad Memory Dekhna
# ==================================================

state = agent.get_state(config)

print("\n========== AFTER DELETE ==========")

for message in state.values["messages"]:
    print(
        message.type,
        ":",
        message.content,
        "ID:",
        message.id
    )

