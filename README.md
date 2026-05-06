# XAI Threat Intelligence System

A **production-grade Explainable AI system** that detects potential threats in images and provides **visual + quantitative explanations**, built using a **scalable microservices architecture, Redis caching, CI/CD automation, and cloud deployment**.

---

## Why This Project Is Different

This is **not just a model** — it is a **complete, production-style ML system** designed with real-world engineering constraints:

- Explainable AI as a **first-class component**
- Redis caching for **low-latency inference**
- Microservices architecture (**API + UI + Cache separation**)
- Fully containerized system (**Docker + Compose**)
- End-to-end **CI/CD pipeline**
- **Cloud deployment on AWS EC2**
- Interactive UI with explainability + metrics
- Auto-generated **JSON + PDF reports**

---

## What the System Does

Given an input image, the system performs:

### 1. Prediction
- Detects object (e.g., *revolver, knife, dog*)

### 2. Quantitative Analysis
- **Confidence Score**
- **Threat Score**
- **Trust Score** (cross-method consistency)

### 3. Explainability Generation
- GradCAM++
- ScoreCAM
- Integrated Gradients

### 4. Output
- Human-readable explanation
- Visual heatmaps
- Structured reports (JSON + PDF)
---

## System Architecture

User → Streamlit UI → FastAPI API → ML Model + XAI → Redis Cache

---

## Microservices Breakdown

| Service    | Responsibility                          |
|------------|-----------------------------------------|
| API        | Model inference, XAI, scoring engine    |
| Streamlit  | UI, visualization, report download      |
| Redis      | Caching repeated predictions            |

---

## Tech Stack

**Backend**
- FastAPI
- PyTorch
- OpenCV
- NumPy

**Frontend**
- Streamlit

**Explainability**
- GradCAM++
- ScoreCAM
- Integrated Gradients

**DevOps & Infra**
- Docker
- Docker Compose
- GitHub Actions (CI/CD)
- Redis
- AWS EC2

---

## Core System Features

### Explainability Flow

GradCAM++ → ScoreCAM → Integrated Gradients

### Attention Analysis
- Quantifies where the model is focusing
- Measures consistency across XAI methods

### Threat Intelligence Engine

Combines:
- Model confidence
- Attention focus
- Label risk level
- Cross-method consistency

---

## Redis Caching Layer

- Eliminates redundant inference
- Instant response for repeated inputs
- Automatic cache detection

**UI Indicators:**
- Fresh inference
- Served from cache

---

## Report Generation

- **JSON report** — structured data
- **PDF report** — includes prediction, metrics, explanation, and visual maps

---

## Cloud Deployment (AWS EC2)

The system is deployed on **AWS EC2**, simulating real production infrastructure:

- Hosted multi-container system on EC2 instance
- Managed container networking across services
- Exposed API and UI endpoints
- Solved real-world issues:
  - Port conflicts
  - Docker networking
  - Resource constraints (free-tier optimization)

---

## CI/CD Pipeline (GitHub Actions)

Automated pipeline that:

- Installs dependencies
- Runs unit tests
- Builds Docker images
- Spins up full system (API + UI + Redis)
- Performs health checks

This ensures:
- Code reliability
- Reproducible builds
- Production readiness

---

## Running Locally

### 1. Clone the repo

    git clone https://github.com/your-username/XAI-Threat-Detection.git
    cd XAI-Threat-Detection

### 2. Run with Docker

    docker compose up --build

### 3. Access services

- UI → http://localhost:8501
- API → http://localhost:8000

---

## Running Tests

    pytest

---

## Project Structure

    XAI-Threat-Detection/
    │
    ├── app/
    │   ├── model/
    │   │   ├── model.py
    │   │   ├── gradcam.py
    │   │   ├── threat.py
    │   │   └── explainer.py
    │   ├── main.py
    │   └── Dockerfile
    │
    ├── streamlit/
    │   ├── streamlit_app.py
    │   ├── requirements.txt
    │   └── Dockerfile
    │
    ├── tests/
    ├── docker-compose.yml
    ├── requirements.txt
    └── .github/workflows/ci.yml

    ---

## Engineering Depth & Challenges Solved

This project reflects **real-world system engineering**, not just ML:

### Docker & System Design
- Fixed Docker build context issues
- Managed multi-container communication
- Handled dependency isolation

### Networking
- Solved inter-container communication
- API ↔ UI ↔ Redis integration

### CI/CD Debugging
- Resolved environment mismatches between local & CI
- Ensured reproducible builds

### Performance Optimization
- Introduced Redis caching
- Reduced redundant computation

---

## Future Improvements

- Kubernetes-based scaling
- Authentication & logging layer
- Real-time video threat detection
- Monitoring & observability (Prometheus/Grafana)

---

## Contribution

Open to ideas, improvements, and collaborations.

---

## Author

**Sree Vishnu**

Built with a focus on: **Machine Learning + Systems Design + DevOps Integration**

---

## Philosophy

> A strong ML engineer doesn't just build models — they build systems that actually work in the real world.
