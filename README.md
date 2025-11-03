# RadicAI Take-home Assignment – LLM-based Ad Campaign Agent

**Author:** Ashis Khan   
**Objective:** Develop a **Generative AI (LLM) agent** that transforms a short ad campaign brief into a structured JSON campaign plan and multiple creative variants, demonstrating prompt design, system design, and practical GenAI implementation skills.


# Overview
This project implements an **LLM-powered Ad Campaign Planner** capable of generating ad creatives programmatically with full schema validation, grounding via knowledge base, and metrics logging.

It supports two modes of operation:

**Mock mode** : Deterministic offline generator (for reproducibility).

**Live mode** : Uses OpenAI API (`client.chat.completions.create`) to produce real campaign plans.

This project implements an **LLM-powered campaign planner** that:
1. Reads a campaign brief in JSON.
2. Builds prompts and optionally injects product facts from a local **knowledge base (KB)**.
3. Generates a machine-readable campaign plan and ad creatives in strict JSON format.
4. Validates the structure against a JSON Schema before saving.
5. Supports both **mock** (no API key) and **live** (OpenAI) modes.

It is **fully reproducible**, **schema-validated**, and **grounding-aware**.


# Key Features

## Capability with Description 

- **Prompt Builder**: Dynamically composes system and user prompts based on campaign brief and knowledge 
- **Schema validation** : Enforces `schema/campaign_schema.json` for ingestibility.
- **Grounding (KB)** : Fetches real product facts from `kb/products.json` to minimize hallucinations.
- **Creative Scoring**: Heuristic scoring based on headline, body, and CTA strength.
- **Logging** : Saves every run (brief, raw reply, final JSON, and errors) under `OUTPUTS/`.
- **Metrics Logging**: Captures latency, token usage, and aggregated creative quality
- **Tests** : Automated tests (Pytest) for schema and logic validation.
- **Streamlit UI**: Interactive front-end to run in mock/live mode, upload briefs, and visualize 
- **Dockerized Deployment**: Reproducible environment for reviewers to test seamlessly.


# Setup & Installation

## Setup of env file
export OPENAI_API_KEY="api_key"
export OPENAI_MODEL="gpt-4o-mini"

## Test with cli commands
- ### For mock without openai live key
python agent/run_agent.py examples/brief1.json --out examples/out_mock.json

- ### Test live with openai key
python agent/run_agent.py examples/brief1.json --out examples/out_live.json --live

- ### Run the visual interface:
streamlit run app.py

- ### Test with test_agent.py file (For mock offline testing)
pytest -q

- ### Test with test_live_openai.py file (For live testing with original openai apikey)
pytest -q -m Live


```bash
git clone https://github.com/AshisKhan/RadicAI-take-home-exercise-repository.git
cd project
pip install -r requirements.txt

