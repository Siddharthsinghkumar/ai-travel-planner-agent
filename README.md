# ✈️ AI Travel Planner Agent

A **local-first, agentic AI travel planning system** that converts natural language travel requests into structured flight recommendations, ranked results, and real-world booking handoffs.

This project demonstrates **agent orchestration, tool usage, validation, and human-in-the-loop automation** — not just a chatbot UI.

---

## 🚀 What This System Does

- Parses free-form travel queries (dates, routes, preferences)
- Searches and validates flight data using external APIs
- Scores flights based on price, duration, and user intent
- Fetches destination weather with fallback providers
- Generates explanations using a **local LLM (Ollama)**
- Performs **deterministic booking handoff** to real booking platforms
- Persists session history for auditability (PostgreSQL-ready)

---

## ⚡ Quick Start

```bash
git clone https://github.com/<your-username>/ai-travel-planner-agent.git
cd ai-travel-planner-agent
pip install -r requirements.txt
ollama run mistral
streamlit run ui/streamlit_app.py
```

(Optional API mode)
```bash
uvicorn api.app:app --reload
```

---

## 🏗️ Architecture Overview

```mermaid
flowchart TD
    UI["Streamlit UI"] --> API["FastAPI Layer"]
    API --> Planner["Planner Agent"]

    Planner --> Flights["Flight Search Tool"]
    Planner --> Weather["Weather Tool"]
    Planner --> Pricing["Price Scoring Engine"]
    Planner --> LLM["LLM Router"]

    LLM --> Ollama["Local LLM - Ollama"]
    LLM --> Cloud["Cloud LLM - Optional Fallback"]

    Planner --> Handoff["Deterministic Booking Handoff"]
    Planner --> DB["Session History and Audit Log"]


---

## 🧠 Core Design Principles

- Local-first inference
- Tool-driven reasoning
- Deterministic workflows
- Human-in-the-loop automation
- Stateless, API-ready architecture

---

## 🔁 Deterministic Booking Handoff

Instead of brittle browser automation, this system performs a **deterministic handoff**:

- Builds deep-link booking URLs
- Pre-fills route, date, passengers
- Redirects to trusted booking platforms
- User completes payment manually

This mirrors real systems like Google Flights and Skyscanner.

---

## 🖥️ Tech Stack

- Python 3.x
- Ollama (local LLM)
- FastAPI
- Streamlit
- Pydantic
- PostgreSQL-ready persistence

---

## 👤 Author

**Siddharth Singh**  
