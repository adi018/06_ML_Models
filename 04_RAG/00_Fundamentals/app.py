import os
import chromadb
from openai import OpenAI
from chromadb.utils.embedding_functions import EmbeddingFunction
import requests
from chromadb.utils import embedding_functions

# Custom embedding function that calls Ollama's local embedding API
class OllamaEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model="nomic-embed-text", url="http://localhost:11434/api/embeddings"):
        """
        Initialize the OllamaEmbeddingFunction.
        Params:
        - model (str): The model to use for embedding.
        - url (str): The URL of the Ollama API.
        """
        self.model = model
        self.url = url

    def __call__(self, texts):
        """
        Generate embeddings for the given texts.
        Params:
        - texts (str or List[str]): The text or list of texts to embed.
        Returns:
        - List[np.ndarray]: A list of embeddings.
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = []
        for text in texts:
            response = requests.post(self.url, json={
                "model": self.model,
                "prompt": text
            })
            response.raise_for_status()
            data = response.json()
            embeddings.append(data["embedding"])
        return embeddings

# Function to load documents from a directory
def load_documents_from_directory(directory_path):
    """
    Load documents from a directory.
    Params:
    - directory_path (str): The path to the directory containing the documents.
    Returns:
    - List[Dict]: A list of dictionaries containing the document ID and text.
    """

    print("==== Loading documents from directory ====")
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            with open(
                os.path.join(directory_path, filename), "r", encoding="utf-8"
            ) as file:
                documents.append({"id": filename, "text": file.read()})
    return documents

# Function to split text into chunks
def split_text(text, chunk_size=1000, chunk_overlap=20):
    """
    Split the text into chunks of a specified size with an overlap.
    
    Params:
    - text (str): The text to split into chunks.
    - chunk_size (int): The size of each chunk.
    - chunk_overlap (int): The number of characters to overlap between chunks.
    
    Returns:
    - List[str]: A list of text chunks.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks

# Function to generate embeddings using OpenAI API
def get_openai_embedding(text):
    """
    Generate an embedding for the given text using the OpenAI API.
    Params:
    - text (str): The text to embed.
    Returns:
    - np.ndarray: The embedding for the text.
    """
    response = client.embeddings.create(input=text, model="nomic-embed-text")
    embedding = response.data[0].embedding
    print("==== Generating embeddings... ====")
    return embedding

# Function to query documents
def query_documents(question, n_results=2):
    """
    Query the collection for relevant documents based on a question.
    Params:
    - question (str): The question to query for.
    - n_results (int): The number of results to return.
    Returns:
    - List[str]: A list of relevant document chunks.
    """
    # query_embedding = get_openai_embedding(question)
    results = collection.query(query_texts=question, n_results=n_results)

    # Extract the relevant chunks
    relevant_chunks = [doc for sublist in results["documents"] for doc in sublist]
    print("==== Returning relevant chunks ====")
    return relevant_chunks
    # for idx, document in enumerate(results["documents"][0]):
    #     doc_id = results["ids"][0][idx]
    #     distance = results["distances"][0][idx]
    #     print(f"Found document chunk: {document} (ID: {doc_id}, Distance: {distance})")

# Function to generate a response from OpenAI
def generate_response(question, relevant_chunks):
    """
    Generate a response to a question using the OpenAI API.
    Params:
    - question (str): The question to answer.
    - relevant_chunks (List[str]): The relevant document chunks.
    Returns:
    - str: The response to the question.
    """

    context = "\n\n".join(relevant_chunks)
    prompt = (
        "You are an assistant for question-answering tasks. Use the following pieces of "
        "retrieved context to answer the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the answer concise."
        "\n\nContext:\n" + context + "\n\nQuestion:\n" + question
    )

    response = client.chat.completions.create(
        model="llama3.2:latest",
        messages=[
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": question,
            },
        ],
    )

    answer = response.choices[0].message
    return answer

# Use the custom embedding function
embedding_fn = OllamaEmbeddingFunction(model="nomic-embed-text")

# Set up Chroma with persistent storage
chroma_client = chromadb.PersistentClient(path="chroma_persistent_storage")
collection = chroma_client.get_or_create_collection(
    name="document_qa_collection",
    embedding_function=embedding_fn
)

# Dummy API key since OpenAI client requires it, but Ollama ignores it
openai_key = "ollama"

# Create OpenAI client for chat/completion (also via Ollama)
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key=openai_key
)

# Load documents from the directory
directory_path = "./news_articles"
documents = load_documents_from_directory(directory_path)

print(f"Loaded {len(documents)} documents")
# Split documents into chunks
chunked_documents = []
for doc in documents:
    chunks = split_text(doc["text"])
    print("==== Splitting docs into chunks ====")
    for i, chunk in enumerate(chunks):
        chunked_documents.append({"id": f"{doc['id']}_chunk{i+1}", "text": chunk})
    # end for
# end for

# print(f"Split documents into {len(chunked_documents)} chunks")

# Generate embeddings for the document chunks
for doc in chunked_documents:
    print("==== Generating embeddings... ====")
    doc["embedding"] = get_openai_embedding(doc["text"])

# print(doc["embedding"])

# Upsert documents with embeddings into Chroma
for doc in chunked_documents:
    print("==== Inserting chunks into db;;; ====")
    collection.upsert(
        ids=[doc["id"]], documents=[doc["text"]], embeddings=[doc["embedding"]]
    )

# Example query
# query_documents("tell me about AI replacing TV writers strike.")
# Example query and response generation
question = "tell me about databricks"
relevant_chunks = query_documents(question)
answer = generate_response(question, relevant_chunks)

print(answer)