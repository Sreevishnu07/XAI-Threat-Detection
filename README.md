# XAI Threat Intelligence System

A **production-grade Explainable AI system** that detects potential threats in images and provides **visual + quantitative explanations**, built using a **microservices architecture, Redis caching, and CI/CD pipeline**.

---

## Why this project stands out

This is not just a machine learning model — it is a **complete ML system** designed like real-world products:

- Explainable AI (GradCAM++, ScoreCAM, Integrated Gradients)
- Redis caching for faster repeated inference
- Fully containerized microservices (Docker + Compose)
- CI/CD pipeline that builds, tests, and runs the entire system
- Interactive UI with explainability flow + metrics
- Auto-generated JSON + PDF reports

---

## What it does

Given an image, the system:

1. Predicts the object (e.g., *revolver, knife, dog*)
2. Calculates:
   - Confidence
   - Threat score
   - Trust score (consistency across XAI methods)
3. Generates:
   - GradCAM++
   - ScoreCAM
   - Integrated Gradients
4. Produces:
   - Human-readable explanation
   - Visual explainability maps
   - Downloadable reports

---

## System Architecture

```
User → Streamlit UI → FastAPI API → ML Model + XAI → Redis Cache
```

---

## Microservices Architecture

| Service     | Role |
|------------|------|
| API        | Model inference + XAI + scoring |
| Streamlit  | UI + visualization |
| Redis      | Caching repeated predictions |

---

## Tech Stack

**Backend:** FastAPI, PyTorch, OpenCV, NumPy  
**Frontend:** Streamlit  
**Explainability:** GradCAM++, ScoreCAM, Integrated Gradients  
**DevOps:** Docker, Docker Compose, GitHub Actions CI/CD, Redis  

---

## Project Structure

```
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
```

---

## Features

### Explainability Flow
GradCAM++ → ScoreCAM → Integrated Gradients

### Attention Analysis
- Quantifies model focus  
- Measures consistency across methods  

### Threat Intelligence Engine
Combines:
- Confidence  
- Focus  
- Label risk  
- Consistency  

### Redis Caching
- Instant response for repeated inputs  
- Detects cached results automatically  
- UI indicates:
  - Fresh inference  
  - Served from cache  

### Report Generation
- JSON report  
- PDF report with prediction, metrics, explanation, and maps  

---

## Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/your-username/XAI-Threat-Detection.git
cd XAI-Threat-Detection
```

### 2. Run with Docker
```bash
docker compose up --build
```

### 3. Access
- UI → http://localhost:8501  
- API → http://localhost:8000  

---

## Running Tests
```bash
pytest
```

---

## CI/CD Pipeline

GitHub Actions automatically:
- Installs dependencies  
- Runs tests  
- Builds Docker images  
- Starts full system (API + Streamlit + Redis)  
- Performs health checks  

---

## Key Engineering Highlights

- Designed a **microservice-based ML system**
- Built **explainability-first architecture**
- Integrated **Redis caching for performance**
- Solved real-world DevOps challenges:
  - Docker build context issues  
  - Container networking  
  - CI environment differences  
- Implemented a **production-style CI/CD pipeline**

---

## Future Improvements

- Cloud deployment (AWS / Render / Railway)  
- Authentication & logging  
- Real-time video inference  
- Kubernetes scaling  

---

## Contribution

Open to improvements, ideas, and collaborations.

---

## Authored By Sree Vishnu

Built with a focus on **Machine Learning + Systems + DevOps integration** — not just model accuracy.
