#  XAI Threat Detection System

A **containerized ML microservice system** with **Explainable AI (Grad-CAM)**, built using Docker, FastAPI, Streamlit, and GitHub Actions CI/CD.

---

##  Project Overview

This project demonstrates how to design and deploy a **production-style machine learning system** using modern DevOps practices.

Instead of focusing only on model accuracy, this system emphasizes:

*  Containerization (Docker)
*  CI/CD automation (GitHub Actions)
*  Microservice architecture
*  Explainable AI (XAI)

---

##  Architecture

```
User → Streamlit UI → FastAPI API → ML Model → Prediction + Explanation
                    ↓
                 Docker
                    ↓
            GitHub Actions CI/CD
```

---

##  Tech Stack

* **Backend:** FastAPI
* **Frontend:** Streamlit
* **ML Framework:** PyTorch (planned)
* **Containerization:** Docker & Docker Compose
* **CI/CD:** GitHub Actions

---

##   Project Structure

```
xai-threat-detection/
│
├── app/                # FastAPI backend + ML logic
├── ui/                 # Streamlit frontend
├── tests/              # Test cases (pytest)
├── data/               # Input data (mounted via Docker)
├── logs/               # Logs (persistent volume)
│
├── docker-compose.yml  # Multi-service orchestration
├── requirements.txt    # Python dependencies
├── .env                # Environment variables
└── .github/workflows/  # CI/CD pipelines
```

---

##  Getting Started

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd xai-threat-detection
```

---

### 2. Run using Docker

```bash
docker-compose up --build
```

---

### 3. Access services

* API → http://localhost:8000
* UI → http://localhost:8501

---

##  Running Tests

Tests are executed inside the container:

```bash
docker-compose run api pytest
```

---

## Current Status

* Project structure initialized
* Testing setup with pytest
* Docker services in progress
* CI/CD pipeline setup

---

## Future Improvements

* Add trained CNN model (ResNet18)
* Integrate Grad-CAM for explainability
* Enhance UI visualization
* Add Redis caching layer
* Extend CI/CD with Docker build + integration tests

---

## Contribution

This project is part of a DevOps-focused ML system design. Contributions and suggestions are welcome.

---

## 📜 License

MIT License
