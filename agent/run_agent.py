"""
RadicAI Take-home — LLM-based Ad Campaign Agent
------------------------------------------------
Usage examples:
  ## Mock mode (no API key needed):
    command:
      python agent/run_agent.py examples/brief1.json --out examples/out_mock.json

  ## Live mode (requires OpenAI API key):
    env setup:
      export OPENAI_API_KEY="sk-xxxx"
      export OPENAI_MODEL="gpt-4o-mini"
    command:
      python agent/run_agent.py examples/brief1.json --out examples/out_live.json --live

Options:
  --live   : Call OpenAI API (requires OPENAI_API_KEY)
  --no-kb  : Disable knowledge base grounding
  --out    : Output JSON file path
"""

import os
import json
import re
import argparse
from pathlib import Path
from datetime import datetime
from jsonschema import validate, ValidationError
from dotenv import load_dotenv

load_dotenv()

#Paths & constants ---------------------
BASE = Path(__file__).resolve().parents[1]
PROMPTS_PATH = BASE / "agent" / "prompts.md"
SCHEMA_PATH = BASE / "schema" / "campaign_schema.json"
KB_PATH = BASE / "kb" / "products.json"
OUTPUT_DIR = BASE / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

#Utilities -------------------------
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_prompts():
    if not PROMPTS_PATH.exists():
        raise FileNotFoundError("Missing prompts.md file.")
    raw = PROMPTS_PATH.read_text(encoding="utf-8")
    try:
        system, user = raw.split("---", 1)
    except ValueError:
        system, user = raw, "Convert this brief into structured JSON: {brief}"
    return system.strip(), user.strip()

def extract_json_from_text(text: str):

    """Extracts the last valid JSON object from the text output."""

    matches = list(re.finditer(r"\{[\s\S]*\}", text))
    for match in reversed(matches):
        try:
            return json.loads(match.group(0))
        except Exception:
            continue
    raise ValueError("No valid JSON found in model output.")

# Mock deterministic generator --------------------------
def mock_generate_plan(brief):
    product = brief.get("product", {})
    channels = brief.get("channels") or ["social"]
    total_budget = int(brief.get("budget") or 1000)
    budget_breakdown = {}
    for i, ch in enumerate(channels):
        if i == 0:
            budget_breakdown[ch] = int(total_budget * 0.6)
        else:
            budget_breakdown[ch] = int(total_budget * 0.4 // max(1, len(channels)-1))
    s = sum(budget_breakdown.values())
    if s != total_budget:
        budget_breakdown[channels[0]] += total_budget - s

    ad_groups = []
    aud_hints = brief.get("audience_hints") or ["general"]
    for i, hint in enumerate(aud_hints):
        creatives = []
        for j in range(2):
            cid = f"c_{i+1}{chr(97+j)}"
            pname = product.get("name", "Product")
            price = product.get("price", "").lower()
            cta = "Start Free Trial" if "trial" in price else "Learn More"
            headline = f"{pname} — Try Free" if "trial" in price else f"{pname} — Learn More"
            feature = product.get("key_features", ["Key benefit"])[0]
            body = f"{feature} — concise benefit-driven copy."
            creatives.append({
                "id": cid,
                "headline": headline,
                "body": body[:140],
                "cta": cta,
                "justification": "Mock creative: deterministic rule-based output; conservative wording."
            })
        ad_groups.append({
            "id": f"ag_{i+1}",
            "target": {"segment_hint": hint, "age": "25-45"},
            "creatives": creatives
        })

    return {
        "campaign_id": brief.get("campaign_id", "cmp_mock_001"),
        "campaign_name": f"{product.get('name','Campaign')} Plan",
        "objective": brief.get("goal", "unknown"),
        "total_budget": total_budget,
        "budget_breakdown": budget_breakdown,
        "ad_groups": ad_groups,
        "checks": {
            "budget_sum_ok": sum(budget_breakdown.values()) == total_budget,
            "required_fields_present": all(k in brief for k in ["campaign_id","goal","product","budget"]),
            "grounding_source": None
        }
    }

#Build prompt (inject KB facts) -----------------------------
def build_prompt(brief, kb=None):
    system, user = load_prompts()
    product_name = brief.get("product", {}).get("name")
    grounding_source = None
    facts = []
    if kb and product_name and product_name in kb:
        facts = kb[product_name].get("facts", [])
        grounding_source = product_name
    user_prompt = user.replace("{brief}", json.dumps(brief, ensure_ascii=False))
    if facts:
        user_prompt += "\n\nGrounding facts: " + ", ".join(facts)
    return system, user_prompt, grounding_source


# Live OpenAI call (with metrics) ----------------
def call_openai_chat(system_prompt, user_prompt, model=None, temperature=0.2, max_tokens=1200):
    """
    Calls OpenAI Chat Completions using the modern API and returns both the
    model reply and metadata (token usage, latency).
    """
    import time
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment")

    client = OpenAI(api_key=api_key)
    model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    start = time.time()
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    end = time.time()

    # Extract metrics safely
    usage = getattr(response, "usage", None)
    token_metrics = {
        "input_tokens": getattr(usage, "prompt_tokens", None),
        "output_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }

    latency = round(end - start, 2)
    model_reply = response.choices[0].message.content

    return model_reply, token_metrics, latency


#Validation -------------------------
def validate_output(plan):
    schema = load_json(SCHEMA_PATH)
    validate(instance=plan, schema=schema)


#Creative Scoring Helper --------
def score_creative(creative):
    """Simple CTR-style scoring heuristic for creative quality."""
    headline_len = len(creative.get("headline", ""))
    body_len = len(creative.get("body", ""))
    cta = creative.get("cta", "").lower()

    score = 0
    # Headline sweet spot
    if 25 <= headline_len <= 60:
        score += 0.4
    elif headline_len > 60:
        score += 0.2

    # CTA strength
    strong_ctas = ["buy", "try", "learn", "get", "sign", "shop"]
    if any(word in cta for word in strong_ctas):
        score += 0.4

    # Body conciseness
    if body_len <= 100:
        score += 0.2

    return round(score, 2)


#Run Agent --------------------
def run_agent(brief_path, out_path, live=False, use_kb=True):

    from pathlib import Path

    out_path = Path(out_path)
    brief_path = Path(brief_path)
    brief = load_json(brief_path)
    kb = load_json(KB_PATH) if (use_kb and KB_PATH.exists()) else None
    system_prompt, user_prompt, grounding_source = build_prompt(brief, kb)

    result_plan = None
    model_reply = None
    error_msg = None
    latency = 0
    token_metrics = {} 

    if live:
        try:
            print("[INFO] Calling OpenAI API...")
            model_reply, token_metrics, latency = call_openai_chat(system_prompt, user_prompt)
            result_plan = extract_json_from_text(model_reply)
            if grounding_source:
                result_plan.setdefault("checks", {})["grounding_source"] = grounding_source
            validate_output(result_plan)
            print("[SUCCESS] Valid JSON received and validated.")
        except Exception as e:
            error_msg = str(e)
            print("[WARN] Live call failed:", error_msg)
            print("[INFO] Falling back to mock mode.")

    if result_plan is None:
        result_plan = mock_generate_plan(brief)
        if grounding_source:
            result_plan["checks"]["grounding_source"] = grounding_source

    #Apply creative scoring
    for ag in result_plan.get("ad_groups", []):
        for creative in ag.get("creatives", []):
            creative["score"] = score_creative(creative)

    #Collect metrics
    total_creatives = sum(len(a["creatives"]) for a in result_plan.get("ad_groups", []))
    avg_score = (
        sum(score_creative(c) for a in result_plan.get("ad_groups", []) for c in a["creatives"])
        / max(1, total_creatives)
    )

    metrics = {
        "mode": "live" if live else "mock",
        "model": os.getenv("OPENAI_MODEL", "mock"),
        "latency_sec": latency if live else 0,
        "token_usage": token_metrics if live else {},
        "num_ad_groups": len(result_plan.get("ad_groups", [])),
        "num_creatives": total_creatives,
        "avg_creative_score": round(avg_score, 2),
    }

    #Save outputs and logs
    save_json(out_path, result_plan)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = OUTPUT_DIR / f"run_{timestamp}"
    log_dir.mkdir(exist_ok=True)
    save_json(log_dir / "brief.json", brief)
    if model_reply:
        (log_dir / "model_reply.txt").write_text(model_reply, encoding="utf-8")
    save_json(log_dir / "output.json", result_plan)
    if error_msg:
        (log_dir / "error.txt").write_text(error_msg, encoding="utf-8")
    save_json(log_dir / "metrics.json", metrics)

    print(f"[DONE] Output written to: {out_path}")
    print(f"[LOGS] Run artifacts saved under: {log_dir}")
    print(f"[SCORES] Avg creative score: {metrics['avg_creative_score']}")




#CLI entry point -----------------------------------
def main():
    ##Add argparse for CLI options
    parser = argparse.ArgumentParser(description="RadicAI LLM Ad Campaign Agent")
    parser.add_argument("brief", help="Path to campaign brief JSON")
    parser.add_argument("--out", required=True, help="Path to save output JSON")
    parser.add_argument("--live", action="store_true", help="Enable live OpenAI mode")
    parser.add_argument("--no-kb", action="store_true", help="Disable KB grounding")
    args = parser.parse_args()

    run_agent(
        brief_path=Path(args.brief),
        out_path=Path(args.out),
        live=args.live,
        use_kb=not args.no_kb
    )

if __name__ == "__main__":
    main()
