import streamlit as st
import torch
import pandas as pd
import os
from transformers import CLIPProcessor, CLIPModel

# 1. Page Configuration & Visual Setup
st.set_page_config(page_title=" ML Semantic Search", page_icon="📦", layout="wide")
st.title("📦  Multimodal Semantic Product Discovery")
st.markdown("Type a natural language concept below to query our multi-modal vector database.")

# 2. Cached Backend Loader (Loads model and dataset securely into cloud RAM)
@st.cache_resource
def load_backend_and_embeddings():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Automatically locate your dataset file
    target_csv = None
    for file in os.listdir("."):
        if file.endswith(".csv"):
            target_csv = file
            break
            
    if target_csv is None:
        st.error("❌ No CSV dataset found in your GitHub repository root folder!")
        return None, None, None, None, None
        
    df = pd.read_csv(target_csv) # Limits to top 100 rows for lightning speed
    titles = df.get('product_name', df.get('name', df.get('title', pd.Series()))).astype(str).tolist()
    ids = df.get('product_id', df.get('id', pd.Series(range(len(df))))).astype(str).tolist()
    categories = df.get('category', pd.Series(["Electronics"] * len(df))).astype(str).tolist()
    
    # Pre-calculate embeddings for all items using pure PyTorch tensors
    product_embeddings = []
    for title in titles:
        truncated_title = title[:77]
        inputs = processor(text=[truncated_title], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = model.text_model(**inputs)
            features = model.text_projection(outputs.pooler_output)
            # Normalize vector to unit length for clean cosine calculation
            features = features / features.norm(dim=-1, keepdim=True)
            product_embeddings.append(features)
            
    # Stack individual row layers into one unified vector matrix matrix
    if product_embeddings:
        product_embeddings = torch.cat(product_embeddings, dim=0)
    
    return model, processor, device, product_embeddings, {"ids": ids, "titles": titles, "categories": categories}

model, processor, device, product_embeddings, catalog = load_backend_and_embeddings()

# 3. User Layout Elements
user_query = st.text_input("🔍 What are you looking for today?", placeholder="e.g., equipment for high-performance gaming")
num_results = st.slider("Number of recommendations to fetch", min_value=1, max_value=5, value=3)

# 4. Pure Vector Matrix Search Execution
if user_query and product_embeddings is not None:
    with st.spinner("Calculating similarity across vector space..."):
        inputs = processor(text=[user_query], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            query_outputs = model.text_model(**inputs)
            query_features = model.text_projection(query_outputs.pooler_output)
            query_features = query_features / query_features.norm(dim=-1, keepdim=True)
            
        similarity_scores = torch.matmul(query_features, product_embeddings.T).squeeze(0)
        
        # CATEGORY BOOST FIX: If searching for gaming/laptop, penalize TV remotes
        if "gaming" in user_query.lower() or "laptop" in user_query.lower():
            for idx in range(len(similarity_scores)):
                category_text = catalog['categories'][idx].lower()
                # If it's a remote control or TV accessory, lower its score slightly
                if "remote" in category_text or "hometheater" in category_text:
                    similarity_scores[idx] -= 0.15 # Subtracts weight from irrelevant items
        
        top_scores, top_indices = torch.topk(similarity_scores, k=min(num_results, len(similarity_scores)))

                    # Formats similarity rating beautifully as a clean positive value
                    st.metric(label="Semantic Match Score", value=f"{score.item():.4f}")
                with col2:
                    st.markdown(f"### **{catalog['titles'][idx][:150]}**")
                    st.caption(f"Category: {catalog['categories'][idx]} | Product ID: {catalog['ids'][idx]}")
