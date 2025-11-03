"""
Basic validation tests for RadicAI Take-home Assignment
Run with:  pytest -q [CLI COMMAND]. this is for the offline mock test without apikey or llm
"""

import json
from pathlib import Path
from jsonschema import validate
import pytest
import os

BASE = Path(__file__).resolve().parents[1]
AGENT_SCRIPT = BASE / "agent" / "run_agent.py"
SCHEMA_PATH = BASE / "schema" / "campaign_schema.json"
EXAMPLES_DIR = BASE / "examples"

# Helpers ---------
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# Tests ---------
def test_schema_exists():
    """Ensure JSON schema file is present and valid JSON"""
    assert SCHEMA_PATH.exists(), "Schema file missing"
    data = load_json(SCHEMA_PATH)
    assert "properties" in data

def test_mock_output_valid(tmp_path):
    """Run mock mode and validate output against schema"""
    out_file = tmp_path / "out_mock.json"
    cmd = f"python {AGENT_SCRIPT} {EXAMPLES_DIR / 'brief1.json'} --out {out_file}"
    result = os.system(cmd)
    assert result == 0, "Agent execution failed"
    output = load_json(out_file)

    # Validate schema
    schema = load_json(SCHEMA_PATH)
    validate(instance=output, schema=schema)

    # Validate budget check
    total = output["total_budget"]
    assert sum(output["budget_breakdown"].values()) == total, "Budget mismatch"

def test_required_fields_present(tmp_path):
    """Ensure required fields exist in mock output"""
    out_file = tmp_path / "out_mock2.json"
    cmd = f"python {AGENT_SCRIPT} {EXAMPLES_DIR / 'brief2.json'} --out {out_file}"
    result = os.system(cmd)
    assert result == 0
    data = load_json(out_file)

    required_keys = ["campaign_id","campaign_name","objective","total_budget","budget_breakdown","ad_groups","checks"]
    for k in required_keys:
        assert k in data, f"Missing key: {k}"

    assert isinstance(data["ad_groups"], list)
    assert len(data["ad_groups"]) >= 1
