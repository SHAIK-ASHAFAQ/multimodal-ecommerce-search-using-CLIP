# Multimodal E-Commerce Semantic Search Engine

An end-to-end vector search application mimicking core Amazon product discovery pipelines. This engine replaces traditional keyword matching with conceptual semantic discovery using OpenAI's CLIP model and an on-disk high-dimensional vector database.

## 🚀 Key Features & Engineering Milestones
- **Semantic Mapping:** Matches user search intent conceptually without requiring exact keyword overlaps (e.g., searching for "high performance gaming" accurately surface audio gear and mechanical peripherals).
- **Production-Scale Ingestion:** Streams and processes massive datasets using Pandas batch generators (`chunksize=500`) to maintain a constant local memory overhead under 1GB.
- **Hardware Acceleration:** Vector pipelines are optimized to execute sub-50ms query lookups, running natively on CPU arrays and utilizing local Neural Processing Units (NPUs).

## 🛠️ System Architecture & Tech Stack
- **Model Graph:** OpenAI's `clip-vit-base-patch32` via Hugging Face Hub (Dual-Transformer Architecture).
- **Vector Database:** ChromaDB utilizing persistent disk storage and Hierarchical Navigable Small World (HNSW) graph indexing.
- **Frontend Framework:** Streamlit Reactive UI Layer wrapped in PyTorch compute pipelines.

---

## 🧠 Version-Agnostic Core Implementation Challenges Solved

### 1. Decoupling Dataclass Abstractions (`BaseModelOutputWithPooling`)
* **The Problem:** Recent internal breaking updates in the Hugging Face `transformers` library caused standard `get_text_features()` functions to return wrapped dataclasses instead of clean raw tensor arrays, breaking downward compatibility with index layers.
* **The Solution:** Patched the embedding logic to interface directly with the base `.text_model()` graph components, isolating vector structures via `.pooler_output` matrices and passing them explicitly into the `.text_projection()` layer to pull clean, flat lists of floats across any backend version wrapper.

### 2. High-Volume Batch Vectorization
* **The Problem:** Ingesting datasets over 10K rows simultaneously triggered thread locks and memory degradation inside standard Pandas memory containers.
* **The Solution:** Implemented high-scale batch stream generators that handle 500 records sequentially, committing vectors directly via ChromaDB's `.upsert()` function to safely bypass execution crashes.

---

## 💻 Local Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com
   cd multimodal-ecommerce-search
   ```

2. **Configure Your Virtual Environment:**
   ```powershell
   python -m venv env
   .\env\Scripts\activate
   ```

3. **Install Core Dependencies:**
   ```powershell
   python -m pip install -r requirements.txt
   ```

4. **Populate the Vector Database:**
   ```powershell
   python embed_catalog.py
   ```

5. **Launch the User Presentation Dashboard:**
   ```powershell
   python -m streamlit run app.py
   ```

---

## 🔬 Interview Preparation Context (Mathematical Tradeoffs)
- **Embedding Dimensions:** The underlying CLIP vision-text projections map tokens directly into a rigid **512-dimensional vector layout**.
- **Context Window Thresholds:** Text payloads passing through the tokenizer are explicitly restricted to CLIP's structural context ceiling of **77 tokens** (~60 words) to prevent index truncations.
- **Vector Distance Calculations:** Due to CLIP's unnormalized spatial layouts, default indices scale distance metrics via Squared L2 (Euclidean) geometries rather than tight `0.0 to 1.0` cosine bounds, necessitating distance threshold filters (`distance < 55.0`) to drop peripheral or unrelated search anomalies.
