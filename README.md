# 🛒 Semantic Recommender System

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.2.2-000000?style=flat&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Stencil](https://img.shields.io/badge/Sentence--Transformers-all--mpnet--base--v2-blue)](https://www.sbert.net/)

An AI-powered product recommendation engine developed for the **Smart India Hackathon 2022**. This system leverages state-of-the-art NLP models to provide semantically relevant product suggestions based on user queries and product descriptions.

---

## 🚀 Overview

Traditional e-commerce search relies on keyword matching. **SIH_ecom** goes beyond keywords by understanding the *intent* and *context* of product descriptions. Using the `all-mpnet-base-v2` transformer model, it maps products into a high-dimensional vector space where "similar" products are physically closer to each other.

### Key Features
- **Semantic Search**: Understands synonyms and context (e.g., "running shoes" vs "athletic footwear").
- **Real-time Inference**: Quick recommendation generation using pre-computed embeddings.
- **REST API**: Simple JSON interface for easy integration with frontend applications.

---

## 🛠 Tech Stack

- **Backend**: [Flask](https://flask.palletsprojects.com/)
- **NLP Model**: [Sentence-Transformers](https://www.sbert.net/) (`all-mpnet-base-v2`)
- **Data Handling**: [Pickle](https://docs.python.org/3/library/pickle.html) for embedding persistence.
- **Similarity Metric**: Cosine Similarity.

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/your-username/SIH_ecom.git
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

## 🗺 Future Roadmap (The Migration Plan)

We are planning to modernize the architecture to improve scalability and performance:

### ⚡ 1. Migrate to FastAPI
The current Flask implementation will be migrated to **FastAPI** to take advantage of:
- **Asynchronous Processing**: Non-blocking handles for better concurrency.
- **Automatic Documentation**: Built-in Swagger UI and Redoc for API testing.
- **Type Safety**: Pydantic models for request/response validation.
- **Improved Performance**: Significantly faster than Flask for high-load scenarios.

### 🗄️ 2. Vector Database Integration
Currently, embeddings are stored in `.pkl` files and loaded into memory. We plan to migrate to a dedicated **Vector Database** (e.g., **ChromaDB**, **Qdrant**, or **Pinecone**):
- **Efficient Retrieval**: Fast k-NN (k-Nearest Neighbors) search without loading everything into RAM.
- **Scalability**: Handle millions of products without performance degradation.
- **Dynamic Updates**: Add or remove products without retraining or reloading the entire dataset.

---

## 🎓 Acknowledgments
Developed during **Smart India Hackathon 2022** by **Debgandhar Ghosh**.

---

## 📞 Contact
Feel free to reach out for consultancy or professional inquiries. Paid customization and specialized integrations are also available.

📧 **Email**: [debgandhar4000@gmail.com](mailto:debgandhar4000@gmail.com)
