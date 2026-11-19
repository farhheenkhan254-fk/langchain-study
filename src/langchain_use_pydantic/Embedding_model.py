from dotenv import load_dotenv
load_dotenv()

# LangChain ke Ollama integration se OllamaEmbeddings class import kar rahe hain.
# Iska use text ko embedding/vector mein convert karne ke liye hota hai.
from langchain_ollama import OllamaEmbeddings
# LangSmith ka traceable decorator import kar rahe hain.
# Ye kisi bhi function ko LangSmith mein trace hone layak banata hai.
from langsmith import traceable


# Embedding model ka object create kar rahe hain.
# Yahan Ollama ka "nomic-embed-text" embedding model use hoga.
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

# embed_query() ko ek traceable function ke andar wrap kar rahe hain,
# taaki ye call LangSmith mein trace ke roop mein dikhe.
# run_type="embedding" batata hai ki ye ek embedding call hai.
@traceable(run_type="embedding")
def get_embedding(text: str):
    return embeddings.embed_query(text)


# Jis text ka embedding/vector banana hai,
# us text ko ek variable mein store kar rahe hain.
text = "What is LangChain?"


# embed_query() user ke query/text ko embedding vector mein convert karta hai.
# Input:  "What is LangChain?"
# Output: numbers ki list/vector
# vector = embeddings.embed_query(text)
vector = get_embedding(text)


# Original text ko print kar rahe hain.
print("Text:", text)

# Generated embedding/vector ko print kar rahe hain.
# Ye hundreds of numerical values ki list hogi.
print("Vector:", vector)
# print("Vector:", vector[0])

# Vector ke andar kitne numbers hain, ye check kar rahe hain.
# Nomic embedding model ke case mein yahan 768 aa sakta hai.
print("Vector Length:", len(vector))
