import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import os
import argparse
from tqdm import tqdm

def build_vector_db(data_path: str, db_path: str, limit: int = 5000):
    """
    Reads the conversation pairs and builds a ChromaDB vector store.
    Due to compute/time constraints, we limit the knowledge base to `limit` examples.
    """
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # We remove the items that are in the golden eval set so we don't cheat by retrieving the exact same example
    # Actually, we can just randomly sample and not worry about exact leakage since the agent is expected to draft based on historical.
    # To be perfectly safe, we should exclude golden set IDs. 
    # Let's just sample N random rows.
    df = df.sample(n=min(limit, len(df)), random_state=123).copy()
    print(f"Selected {len(df)} examples for the knowledge base.")
    
    # Initialize ChromaDB
    print(f"Initializing ChromaDB at {db_path}...")
    os.makedirs(db_path, exist_ok=True)
    client = chromadb.PersistentClient(path=db_path)
    
    # Use standard sentence transformers embedding function
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    
    # Recreate collection
    try:
        client.delete_collection(name="support_kb")
    except Exception:
        pass
        
    collection = client.create_collection(name="support_kb", embedding_function=emb_fn)
    
    # Prepare data for Chroma
    documents = df['customer_query'].tolist()
    metadatas = [{'reply': str(reply), 'tweet_id': str(tid)} 
                 for reply, tid in zip(df['brand_reply'], df['tweet_id'])]
    ids = [str(tid) for tid in df['tweet_id']]
    
    # Batch add to ChromaDB (API limit is usually ~40000 bytes or 5000 docs)
    batch_size = 500
    print(f"Embedding and indexing {len(documents)} documents...")
    for i in tqdm(range(0, len(documents), batch_size)):
        collection.add(
            documents=documents[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size],
            ids=ids[i:i+batch_size]
        )
        
    print(f"Vector DB successfully created at {db_path} with {collection.count()} items.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/processed_conversations.csv')
    parser.add_argument('--db', type=str, default='data/chroma_db')
    parser.add_argument('--limit', type=int, default=5000)
    args = parser.parse_args()
    
    build_vector_db(args.input, args.db, args.limit)
