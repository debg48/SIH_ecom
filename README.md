# 🛒 Semantic Recommender System

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95.1-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Stencil](https://img.shields.io/badge/Sentence--Transformers-all--mpnet--base--v2-blue)](https://www.sbert.net/)

An AI-powered product recommendation engine developed for the **Smart India Hackathon 2022**. This system leverages state-of-the-art NLP models to provide semantically relevant product suggestions based on user queries and product descriptions.

---

## 🚀 Overview

Traditional e-commerce search relies on keyword matching. Our recommendation system goes beyond keywords by understanding intent and context. Using the `all-mpnet-base-v2` transformer model, it maps products into a high-dimensional vector space.

### Key Features
- **Semantic Search**: Understands synonyms and context.
- **REST API**: Clean FastAPI interface with industry-standard structuring.
- **Configuration Driven**: All paths and settings managed via `config.yaml`.

---

## 📁 Project Structure

```
SIH_ecom/
├── app/
│   └── main.py      # Entry point: API routes and application logic
├── util/
│   └── recom.py     # Recommender class handling ML inference
├── data/
│   ├── emb.pkl      # Pre-computed product embeddings
│   └── prod.pkl     # Product name mappings
├── config.yaml      # Centralized configuration file
├── requirements.txt # Project dependencies
└── README.md
```

---

## 🛠 Tech Stack

- **Backend**: [FastAPI](https://fastapi.tiangolo.com/)
- **NLP Model**: [Sentence-Transformers](https://www.sbert.net/) (`all-mpnet-base-v2`)
- **Data Handling**: [Pickle](https://docs.python.org/3/library/pickle.html) for embedding persistence.
- **Configuration**: YAML.

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
python app/main.py
# OR
uvicorn app.main:app --host 0.0.0.0 --port 5000 --reload
```

---

## 🔌 API Usage

The system exposes a single GET endpoint:

**Endpoint**: `/`  
**Query Parameter**: `product_name` (Alias: `q`)

**Example Request**:
```bash
curl "http://localhost:5000/?product_name=men's formal shirt"
# OR
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

## 📝 Changelog

### v2.1.0 - Architecture Refinement
- **Backend Migration**: Successfully migrated from Flask to **FastAPI** for better performance and type safety.
- **Consolidated Structure**: Simplified into `app/` and `util/` for clarity.
- **Configuration System**: Introduced `config.yaml` for environment-agnostic setup.
- **API Improvements**: Renamed query parameter to `product_name` (retaining `q` as an alias).
- **Data Isolation**: Moved all persistence files to the `data/` directory.

---

## 🗺 Future Roadmap

- **Vector Database**: Migrate to ChromaDB/Qdrant for better scalability.
- **Lightweight Models**: Integration with `all-MiniLM-L6-v2` for faster edge inference.

---

## 🎓 Acknowledgments
Developed during **Smart India Hackathon 2022** by **Debgandhar Ghosh**.

---

## 📞 Contact
📧 **Email**: [debgandhar4000@gmail.com](mailto:debgandhar4000@gmail.com)
