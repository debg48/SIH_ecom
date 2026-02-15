# 🛒 Semantic Recommender System

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95.1-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20DB-DC244C?style=flat&logo=data:image/png;base64,&logoColor=white)](https://qdrant.tech/)
[![Stencil](https://img.shields.io/badge/Sentence--Transformers-all--MiniLM--L6--v2-blue)](https://www.sbert.net/)

An AI-powered product recommendation engine developed for the **Smart India Hackathon 2022**. This system leverages state-of-the-art NLP models and a vector database to provide semantically relevant product suggestions based on user queries and product descriptions.

---

## 🚀 Overview

Traditional e-commerce search relies on keyword matching. Our recommendation system goes beyond keywords by understanding intent and context. Using the `all-MiniLM-L6-v2` transformer model and **Qdrant** as a vector database, it maps products into a high-dimensional vector space for fast, scalable semantic search.

### Key Features
- **Semantic Search**: Understands synonyms and context.
- **Vector Database**: Powered by Qdrant for scalable, production-grade similarity search.
- **REST API**: Clean FastAPI interface with health checks and structured error responses.
- **Configuration Driven**: All paths and settings managed via `config.yaml` with environment variable support.
- **Data Pipeline**: Scripts for embedding generation and data migration with multiple modes.

---

## 📁 Project Structure

```
SIH_ecom/
├── app/
│   ├── main.py        # Entry point: API routes, health check, CORS
│   └── loader.py      # Configuration loader with env var expansion
├── util/
│   └── recom.py       # Recommender class (Qdrant-backed vector search)
├── scripts/
│   ├── generate_embeddings.py  # Generate embeddings from CSV → pickle
│   └── migrate.py              # Upload embeddings to Qdrant (replace/drop/append)
├── data/
│   ├── products.csv   # Source product data
│   └── emb.pkl        # Pre-computed product embeddings
├── config.yaml        # Centralized configuration file
├── .env               # Environment variables (Qdrant credentials)
├── requirements.txt   # Project dependencies
└── README.md
```

---

## 🛠 Tech Stack

- **Backend**: [FastAPI](https://fastapi.tiangolo.com/)
- **NLP Model**: [Sentence-Transformers](https://www.sbert.net/) (`all-MiniLM-L6-v2`)
- **Vector Database**: [Qdrant Cloud](https://qdrant.tech/) for embedding storage and search
- **Configuration**: YAML with `${ENV_VAR}` expansion via `python-dotenv`

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

### 4. Configure environment
Create a `.env` file in the project root:
```env
QDRANT_URL=https://your-cluster.cloud.qdrant.io:6333
QDRANT_API_KEY=your-api-key-here
```

### 5. Generate embeddings & migrate data
```bash
# Generate embeddings from products.csv
python scripts/generate_embeddings.py

# Upload to Qdrant (choose a mode)
python scripts/migrate.py --mode replace   # Upsert by ID
python scripts/migrate.py --mode drop      # Drop & recreate collection
python scripts/migrate.py --mode append    # Append after existing data
```

### 6. Run the application
```bash
python app/main.py
# OR
uvicorn app.main:app --host 0.0.0.0 --port 5000 --reload
```

---

## 🔌 API Usage

### Recommendation Endpoint

**Endpoint**: `GET /`  
**Query Parameter**: `product_name` (Alias: `q`)

**Example Request**:
```bash
curl "http://localhost:5000/?product_name=men's formal shirt"
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

### Health Check

**Endpoint**: `GET /health`

```json
{
    "status": "healthy",
    "recommender_ready": true
}
```

---

## 📊 Data Generation

The product data used in this project was synthetically generated using **LLMs** to ensure realistic and diverse descriptions of Indian indigenous crafts.

- **Model**: `microsoft/phi-3-mini-4k-instruct` (via Hugging Face)
- **Categories**: Handloom Textiles, Wood Craft, and Terracotta Craft.
- **Quantity**: 300 unique product items.
- **Process**: A custom prompt-engineering script was used to generate structured JSON data, defining product IDs, names, and natural language descriptions based on regional techniques and materials.

Check out the generation script here:
[![Google Colab](https://img.shields.io/badge/Colab-Notebook-F9AB00?style=flat&logo=googlecolab&logoColor=white)](https://colab.research.google.com/drive/18eI0KWHEa5YUxJXlpZNbXwtt-DzmJZqO?usp=sharing)

---

## � Future Roadmap

- **Authentication & Rate Limiting**: Add API key auth and request throttling for production.
- **Multi-Modal Search**: Support image-based queries alongside text.
- **User Behavior Tracking**: Integrate click-through analytics for feedback-driven re-ranking.
- **A/B Testing Framework**: Compare recommendation strategies in real-time.
- **Containerization**: Dockerize the application with `docker-compose` for Qdrant + API deployment.
- **CI/CD Pipeline**: Automated testing and deployment via GitHub Actions.

---

## �📝 Changelog

### v2.2.0 - Qdrant Integration & Robustness
- **Vector Database**: Migrated from local pickle-based search to **Qdrant Cloud** for scalable vector storage.
- **Model Upgrade**: Switched to `all-MiniLM-L6-v2` (~5x faster inference, 384-dim vectors).
- **Secure Config Loader**: New `app/loader.py` with `${ENV_VAR}` expansion from `.env` files.
- **Data Pipeline**: Added `scripts/generate_embeddings.py` and `scripts/migrate.py` with 3 migration modes (Replace, Drop, Append).
- **Error Handling**: Comprehensive try-catch blocks, input validation, and logging across all modules.
- **Health Endpoint**: New `GET /health` for monitoring service readiness.
- **Graceful Degradation**: App starts even if Qdrant is unavailable, returning `503` on recommendation requests.

### v2.1.0 - Architecture Refinement
- **Backend Migration**: Successfully migrated from Flask to **FastAPI** for better performance and type safety.
- **Consolidated Structure**: Simplified into `app/` and `util/` for clarity.
- **Configuration System**: Introduced `config.yaml` for environment-agnostic setup.
- **API Improvements**: Renamed query parameter to `product_name` (retaining `q` as an alias).
- **Data Isolation**: Moved all persistence files to the `data/` directory.

---

## 🎓 Acknowledgments
Developed during **Smart India Hackathon 2022** by **Debgandhar Ghosh**.

---

## 📞 Contact
📧 **Email**: [debgandhar4000@gmail.com](mailto:debgandhar4000@gmail.com)
