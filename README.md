# 🛒 Semantic Recommender System

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95.1-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Stencil](https://img.shields.io/badge/Sentence--Transformers-all--mpnet--base--v2-blue)](https://www.sbert.net/)

An AI-powered product recommendation engine developed for the **Smart India Hackathon 2022**. This system leverages state-of-the-art NLP models to provide semantically relevant product suggestions based on user queries and product descriptions.

---

## 🚀 Overview

Traditional e-commerce search relies on keyword matching. Our recommendation system goes beyond keywords by understanding the *intent* and *context* of product descriptions. Using the `all-mpnet-base-v2` transformer model, it maps products into a high-dimensional vector space where "similar" products are physically closer to each other.

### Key Features
- **Semantic Search**: Understands synonyms and context (e.g., "running shoes" vs "athletic footwear").
- **Real-time Inference**: Quick recommendation generation using pre-computed embeddings.
- **REST API**: Simple JSON interface for easy integration with frontend applications.

---

## 🛠 Tech Stack

- **Backend**: [FastAPI](https://fastapi.tiangolo.com/)
- **NLP Model**: [Sentence-Transformers](https://www.sbert.net/) (`all-mpnet-base-v2`)
- **Data Handling**: [Pickle](https://docs.python.org/3/library/pickle.html) for embedding persistence.
- **Similarity Metric**: Cosine Similarity.

### Why FastAPI?
We migrated from Flask to FastAPI to improve the production readiness of the system:
- **Asynchronous Execution**: Better handling of concurrent requests during ML inference.
- **Type Safety & Validation**: Automatic request validation using Pydantic models.
- **Auto-Generated Docs**: Interactive OpenAPI (Swagger) documentation available out-of-the-box.
- **Performance**: Built on Starlette and Uvicorn, offering significantly lower overhead.

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/debg48/SIH_ecom.git
cd SIH_ecom
```

### 2. Set up virtual environment
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the application
```bash
python app.py
# OR
uvicorn app:app --host 0.0.0.0 --port 5000 --reload
```

---

## 🔌 API Usage

The system exposes a single GET endpoint:

**Endpoint**: `/`  
**Query Parameter**: `q` (The product name or description)

**Example Request**:
```bash
curl "http://localhost:5000/?q=men's formal shirt"
```

**Example Response**:
```json
{
    "data": [
        ["Product A", 0.92],
        ["Product B", 0.89],
        ["Product C", 0.85]
    ],
    "success": true
}
```

---

## � Changelog

### v2.0.0 - Modernization Update
- **Backend Migration**: Successfully migrated from Flask to **FastAPI**.
- **Performance**: Improved concurrency and reduced startup latency.
- **Documentation**: Added automatic Swagger UI and Redoc support.
- **CORS**: Implemented robust CORSMiddleware for frontend-backend separation.

---

## 🗺 Future Roadmap

We are planning to modernize the architecture to improve scalability and performance:

### 🗄️ 1. Vector Database Integration
Currently, embeddings are stored in `.pkl` files and loaded into memory. We plan to migrate to a dedicated **Vector Database** (e.g., **ChromaDB**, **Qdrant**, or **Pinecone**):
- **Efficient Retrieval**: Fast k-NN (k-Nearest Neighbors) search without loading everything into RAM.
- **Scalability**: Handle millions of products without performance degradation.
- **Dynamic Updates**: Add or remove products without retraining or reloading the entire dataset.

### 🏃 2. Switch to Lightweight Model
We plan to switch from `all-mpnet-base-v2` to a more lightweight model like `all-MiniLM-L6-v2` or the latest efficient Small Language Models (SLMs):
- **Faster Inference**: Reduced latency for real-time recommendations.
- **Lower Resource Usage**: Smaller memory footprint and faster loading times.
- **Edge Compatibility**: Easier to deploy on smaller instances or edge devices.

---

## 🎓 Acknowledgments
Developed during **Smart India Hackathon 2022** by **Debgandhar Ghosh**.

---

## 📞 Contact
Feel free to reach out for consultancy or professional inquiries. Paid customization and specialized integrations are also available.

📧 **Email**: [debgandhar4000@gmail.com](mailto:debgandhar4000@gmail.com)
