# 🔍 Name Matching with Vector DB (FAISS + Sentence Transformers)

This project implements a **name-matching system** using **Sentence Transformers** for embeddings and **FAISS** as a vector database for efficient similarity search.  
It finds the **most similar names** from a dataset when a user enters a name.  

---

## 📌 Features
- Stores a dataset of names (customizable)  
- Converts names into **vector embeddings** using `all-MiniLM-L6-v2`  
- Uses **FAISS** to search for the closest matches  
- Returns:  
  - **Best Match** (highest similarity score)  
  - **Ranked List of Matches** with similarity scores  

---

## ⚙️ Installation

Clone this repository and install dependencies:

```bash
git clone https://github.com/PynaPavani/Get-Matching-Person-Names.git
cd Get-Matching-Person-Names

# Install dependencies
pip install -r requirements.txt
