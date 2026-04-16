import json
import os
import chromadb
from chromadb.utils import embedding_functions

CHROMA_DATA_DIR = "./chroma_data"
DATA_FILE = "esdm_data.json"

def build_vector_db():
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found.")
        return

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    client = chromadb.PersistentClient(path=CHROMA_DATA_DIR)
    
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    try:
        client.delete_collection(name="esdm_exercises")
    except Exception:
        pass
        
    collection = client.create_collection(
        name="esdm_exercises", 
        embedding_function=sentence_transformer_ef
    )

    documents = []
    ids = []
    metadatas = []

    exercises = data if isinstance(data, list) else data.get("exercises", [])

    for i, exercise in enumerate(exercises):
        title = exercise.get("title", "")
        description = exercise.get("description", "")
        domain = exercise.get("domain", "")
        
        doc_text = f"Title: {title}\nDomain: {domain}\nDescription: {description}"
        documents.append(doc_text)
        ids.append(f"exercise_{i}")
        metadatas.append({
            "title": title,
            "domain": domain
        })

    if documents:
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Successfully embedded {len(documents)} exercises into ChromaDB at {CHROMA_DATA_DIR}.")
    else:
        print("No exercises found to embed.")

if __name__ == "__main__":
    build_vector_db()
