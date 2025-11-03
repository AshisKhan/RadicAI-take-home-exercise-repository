"""
Optional integration test for live OpenAI mode.
Run with:  pytest -q -m live [CLI COMMAND], this is for the live test with openai api key. So, it requires OPENAI_API_KEY in environment variables.
"""

import os
import json
import pytest
from pathlib import Path
from jsonschema import validate
from dotenv import load_dotenv

load_dotenv()

BASE = Path(__file__).resolve().parents[1]
AGENT_SCRIPT = BASE / "agent" / "run_agent.py"
SCHEMA_PATH = BASE / "schema" / "campaign_schema.json"
EXAMPLES_DIR = BASE / "examples"


@pytest.mark.live
@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ,
    reason="No OPENAI_API_KEY in environment"
)
def test_live_openai_schema(tmp_path):
    """Integration: ensure live OpenAI response produces valid JSON output."""
    out_file = tmp_path / "out_live.json"
    cmd = (
        f"python {AGENT_SCRIPT} {EXAMPLES_DIR / 'brief1.json'} "
        f"--out {out_file} --live"
    )
    result = os.system(cmd)
    assert result == 0, "Live agent run failed"

    output = json.loads(out_file.read_text(encoding="utf-8"))

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    validate(instance=output, schema=schema)

    assert output["checks"]["budget_sum_ok"], "Budget check failed"
    assert output["checks"]["required_fields_present"], "Missing fields"
