from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI

load_dotenv()

openai_client = OpenAI()

#vector Embeddings
embedding_model = OpenAIEmbeddings(
    model="embedding",
    check_embedding_ctx_length=False
    )

vector_db = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="learning_rag",
    embedding=embedding_model
)

# Take user input
user_query = input("Enter your query: ")

# Relevant chunks from the vector store
search_results = vector_db.similarity_search(query=user_query)

context = "\n\n\n".join([f"Page Content: {result.page_content}\nPage Number:{result.metadata['page_label']}\nFile Location: {result.metadata['source']}" for result in search_results])


SYSTEM_PROMPT = f"""
You are a helpful assistant that answers questions based on the available context retrieved from a PDF file along with page_contents and page numbers.
 You should only answer the user based on the following context and navigate the user to open the right number to know more.
 {context}"""
 
response = openai_client.chat.completions.create(
    model="qwen3",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query}
    ]
)

print(f"🤖: {response.choices[0].message.content}")
 
 