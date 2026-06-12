import streamlit as st
import chromadb
import torch
from transformers import CLIPProcessor, CLIPModel

# 1. Page Configuration & Styling
st.set_page_config(page_title="Multimodal Semantic Product Search", page_icon="📦", layout="wide")
st.title("📦 Multimodal Semantic Product Discovery")
st.markdown("Type a natural language concept below to query our multi-modal vector database.")

# 2. Cached Initialization (Prevents reloading the model on every click)
@st.cache_resource
def load_backend():
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_collection(name="amazon_products")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return collection, model, processor, device

collection, model, processor, device = load_backend()

# 3. User Input Layout
user_query = st.text_input("🔍 What are you looking for today?", placeholder="e.g., equipment for high-performance gaming")
num_results = st.slider("Number of recommendations to fetch", min_value=1, max_value=5, value=3)

# 4. Search Execution Engine
if user_query:
    with st.spinner("Analyzing semantic vector space..."):
        # Tokenize and encode query string
        inputs = processor(text=[user_query], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = model.text_model(**inputs)
            query_embedding = model.text_projection(outputs.pooler_output).cpu().numpy().tolist()
        
        # Query local database index
        results = collection.query(query_embeddings=query_embedding, n_results=num_results)
        
        # FIX: Extract the inner list from ChromaDB's batch container format
        ids = results['ids'][0] if results['ids'] else []
        metadatas = results['metadatas'][0] if results['metadatas'] else []
        distances = results['distances'][0] if results['distances'] else []
        
        if len(ids) == 0:
            st.warning("❌ No products found in the vector space.")
        else:
            st.subheader("✨ Top Recommended Products")
            
            # Display results cleanly using Streamlit Containers
            for i in range(len(ids)):
                with st.container(border=True):
                    col1, col2 = st.columns([1, 4]) # 1:4 width ratio for metrics vs text
                    with col1:
                        # Now safely formats a float number!
                        st.metric(label="Distance Score", value=f"{distances[i]:.2f}")
                    with col2:
                        st.markdown(f"### **{metadatas[i]['title']}**")
                        st.caption(f"Category: {metadatas[i]['category']} | Product ID: {ids[i]}")
