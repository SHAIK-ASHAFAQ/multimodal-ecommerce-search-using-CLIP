import chromadb
import torch
from transformers import CLIPProcessor, CLIPModel

# 1. Connect to your existing ChromaDB Database
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="amazon_products")

# 2. Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def search_products(user_query, number_of_results=2):
    print(f"\n🔍 User Searched For: '{user_query}'")
    
    # 3. Tokenize the input text query
    inputs = processor(text=[user_query], return_tensors="pt", padding=True).to(device)
    
    with torch.no_grad():
        outputs = model.text_model(**inputs)
        raw_tensor = outputs.pooler_output
        text_features = model.text_projection(raw_tensor)
        
        # FIX: Keep the original nested matrix format to match your embed script
        query_embedding = text_features.cpu().numpy().tolist()
    
    # 4. Query ChromaDB using the clean vector list
    results = collection.query(
        query_embeddings=query_embedding, # Directly feed the extracted matrix container
        n_results=number_of_results
    )
    
    # 5. Print out the formatted search results
    print("✨ Top Matching Products Found:")
    
    # Safely extract arrays from ChromaDB's outer batch list container
    ids = results['ids'][0]
    metadatas = results['metadatas'][0]
    distances = results['distances'][0]
    
    if len(ids) == 0:
        print(" ❌ No products found in the vector space.")
    else:
        for i in range(len(ids)):
            product_id = ids[i]
            metadata = metadatas[i]
            distance = distances[i]
            
            print(f" -> [{metadata['category']}] {metadata['title']} (ID: {product_id}, Distance Score: {distance:.4f})")

# --- Test the Search Engine ---
search_products("something to play video games with")
search_products("furniture for sitting comfortably")
