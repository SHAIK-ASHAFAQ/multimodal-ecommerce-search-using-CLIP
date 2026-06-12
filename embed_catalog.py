import os
import pandas as pd
import chromadb
from transformers import CLIPProcessor, CLIPModel
import torch

# 1. Initialize Vector Database
chroma_client = chromadb.PersistentClient(path="./chroma_db") 
collection = chroma_client.get_or_create_collection(name="amazon_products")

# 2. Initialize CLIP Model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 3. Load Real Amazon Data
print("📂 Loading Amazon dataset...")
try:
    # Read the CSV file (Ensure 'amazon.csv' is in the same folder)
    df = pd.read_csv("amazon.csv")
    
    # LIMIT DATA FOR SPEED: Only processing the first 100 products for your demo
    # (Remove .head(100) if you want to index all 1,000+ items later)
    #df = df.head(100)
    
    # Create a list of dictionaries to match our original format
    products_to_index = []
    for index, row in df.iterrows():
        products_to_index.append({
            "id": str(row['product_id']),
            "title": str(row['product_name']),
            "category": str(row['category']).split('|')[0] # Clean up category string
        })
        
    print(f"✅ Loaded {len(products_to_index)} products from CSV.")

except Exception as e:
    print(f"❌ Error loading CSV: {e}")
    print("⚠️ Make sure 'amazon.csv' is in your project folder!")
    products_to_index = []

print(f"Generating vector embeddings using {device}...")

# 4. Generate Embeddings and Upsert to ChromaDB
for product in products_to_index:
    # Skip empty titles
    if not product["title"] or len(product["title"]) < 2:
        continue
        
    # Truncate long titles to fit CLIP's 77 token limit securely
    truncated_title = product["title"][:77]
        
    inputs = processor(text=[truncated_title], return_tensors="pt", padding=True).to(device)
    
    with torch.no_grad():
        # Query the text model graph directly
        outputs = model.text_model(**inputs)
        raw_tensor = outputs.pooler_output
        text_features = model.text_projection(raw_tensor)
        
        # FIX: Unpack the outer dimension batch index [0] to flatten into a clean list of floats
        embedding = text_features.cpu().numpy()[0].tolist()
    
    # Store clean flattened embedding list securely inside ChromaDB
    collection.upsert(
        ids=[product["id"]],
        embeddings=[embedding],
        metadatas=[{"title": product["title"][:200], "category": str(product["category"])}]
    )

print("✅ Successfully indexed real Amazon products into ChromaDB!")
