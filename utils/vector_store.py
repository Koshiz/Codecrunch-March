import faiss
import numpy as np
import pickle
import os

VECTOR_DB_PATH = 'vector_db/index.faiss'
METADATA_PATH = 'vector_db/metadata.pkl'

def save_vector_db(embeddings, metadata):
    os.makedirs('vector_db', exist_ok=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index, VECTOR_DB_PATH)
    with open(METADATA_PATH, 'wb') as f:
        pickle.dump(metadata, f)

def load_vector_db():
    index = faiss.read_index(VECTOR_DB_PATH)
    with open(METADATA_PATH, 'rb') as f:
        metadata = pickle.load(f)
    return index, metadata

def search_vector_db(query_embedding, top_k=5):
    index, metadata = load_vector_db()
    distances, indices = index.search(np.array([query_embedding]), top_k)
    results = [metadata[i] for i in indices[0]]
    return results
