
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

names = [
    "Geetha", "Gita", "Gitu", "Geeta", "Githa", "Geetanjali", "Gopal", "Ganesh", "Gaurav", "Gagan",
    "Sita", "Seetha", "Sitha", "Sunita", "Savitha", "Savitri", "Saritha", "Sanitha", "Sonia", "Sneha",
    "Latha", "Leetha", "Latika", "Lalitha", "Laxmi", "Lakshmi", "Lavanya", "Leela", "Lina", "Linda"
]


model = SentenceTransformer("all-MiniLM-L6-v2")
name_embeddings = model.encode(names, convert_to_numpy=True)

dimension = name_embeddings.shape[1]  
index = faiss.IndexFlatL2(dimension)
index.add(name_embeddings)

def search_similar_names(query_name, top_k=5):
    query_vector = model.encode([query_name], convert_to_numpy=True)
    
    distances, indices = index.search(query_vector, top_k)
    similarities = 1 / (1 + distances[0])  
    
    results = [(names[idx], float(sim)) for idx, sim in zip(indices[0], similarities)]
    return results


user_input = input(str('enter a name to match from vectordb: '))
results = search_similar_names(user_input, top_k=10)

print(f"User Input: {user_input}")
print(f"\nBest Match: {results[0][0]} with score {results[0][1]:.4f}")
print("\nRanked List of Matches:")
for name, score in results:
    print(f"{name} - {score:.4f}")
