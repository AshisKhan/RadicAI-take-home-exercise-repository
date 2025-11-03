# System Design – RadicAI LLM-based Ad Campaign Agent

## 1. Overview
This system automates ad campaign generation using a **Large Language Model (LLM)**.  
It transforms a marketing brief into structured ad creatives with validated JSON output, grounded context, and scoring metrics.

The design focuses on:
- **Reproducibility** – same code supports mock & live OpenAI runs
- **Robustness** – JSON schema validation and error fallback
- **Observability** – logs, scores, latency, and token metrics
- **Modularity** – reusable agent pipeline for different campaign goals



## 2. Architecture

### Components
| Module | Description |
|--------------|-------------|
| `agent/run_agent.py` | Core orchestration layer – loads brief, builds prompt, calls LLM (mock or live), validates, logs, and scores output. |
| `agent/utils.py` | Helper functions for file I/O, schema validation, and grounding. |
| `examples/` | Sample briefs to simulate marketing inputs. |
| `tests/` | Unit tests for mock and live runs (pytest). |
| `outputs/` | Captures structured results, logs, and metrics. |
| `app.py` | Streamlit interface to run campaigns interactively (mock/live). |



## 3. Data Flow


1. **Input:** Marketing brief (uploaded JSON or example)  
2. **Prompt Building:** Combines system + user context; optionally enriches via knowledge base (`kb.json`)  
3. **LLM Execution:**  
   - *Mock mode* → uses deterministic local generator  
   - *Live mode* → calls OpenAI ChatCompletion API  
4. **Validation:** Ensures output matches schema (`ad_groups → creatives`)  
5. **Scoring:** Assigns heuristic quality scores to each creative  
6. **Logging:** Captures token usage, latency, and metadata in `metrics.json`  
7. **UI Display:** Streamlit app visualizes creatives, scores, and cost metrics



## 4. Extensibility

| Concern | Approach |
|----------|-----------|
| **Prompt optimization** | Templates can be externalized for fine-tuning or A/B testing. |
| **Model swapping** | Uses environment variables for `OPENAI_MODEL` — easily replace with Anthropic, Mistral, etc. |
| **Observability** | Metrics JSON enables dashboards (Grafana / Streamlit) for latency, cost, and token analysis. |
| **Deployment** | Fully containerized via Docker; UI accessible via port 8501. |



## 5. Error Handling & Robustness
- **Graceful fallback:** Live API failures switch to mock generator.
- **Schema enforcement:** All outputs validated before saving.
- **Logging isolation:** Each run gets a unique timestamped folder.
- **Safe paths:** Uses `pathlib` for OS-agnostic reliability.



## 6. Metrics & Evaluation
Each run produces:
```json
{
  "mode": "live",
  "model": "gpt-4o-mini",
  "num_ad_groups": 2,
  "num_creatives": 4,
  "avg_creative_score": 0.68,
  "latency_sec": 8.41,
  "token_usage": { "input_tokens": 545, "output_tokens": 368, "total_tokens": 913 }
}
