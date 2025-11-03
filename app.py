import json
import streamlit as st
import os
import tempfile
from agent.run_agent import run_agent

st.set_page_config(page_title="RadicAI Ad Agent", layout="wide")

st.title("RadicAI LLM-based Ad Campaign Agent")
st.write("Generate ad campaigns using your uploaded brief or sample JSON.")

# Sidebar Controls ---
st.sidebar.header("Configuration")
mode = st.sidebar.radio("Select Mode", ["Mock", "Live (OpenAI)"])
use_kb = st.sidebar.checkbox("Use Knowledge Base", value=True)

# Show environment check if live mode selected
if mode == "Live (OpenAI)":
    key_set = bool(os.getenv("OPENAI_API_KEY"))
    if not key_set:
        st.warning("No OPENAI_API_KEY found in environment. Live mode may fail.")
    else:
        st.success("OpenAI key detected. Ready for live mode.")

# Upload Section ---
st.subheader("Upload Brief")
uploaded_file = st.file_uploader("Upload your brief JSON", type=["json"])

# Load a sample brief if user doesn’t upload
examples_dir = "examples"
example_files = [f for f in os.listdir(examples_dir) if f.endswith(".json")]
sample = st.selectbox("Or choose a sample brief:", example_files if example_files else ["No samples found"])

# Run Button ---
if st.button("Generate Campaign"):
    # Determine which brief to use
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
            tmp.write(uploaded_file.read())
            brief_path = tmp.name
    else:
        brief_path = os.path.join(examples_dir, sample)

    out_path = "outputs/ui_output.json"

    with st.spinner("Generating campaign... please wait"):
        run_agent(brief_path, out_path, live=(mode.startswith("Live")), use_kb=use_kb)

        # Display Results ---
        st.success("Campaign generated successfully!")
        with open(out_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        st.markdown("## Generated Ad Campaign")
        for i, ag in enumerate(data.get("ad_groups", []), start=1):
            st.markdown(f"### Ad Group {i}: {ag.get('objective', 'N/A')}")

            # Calculate average group score
            creatives = ag.get("creatives", [])
            avg_group_score = round(sum(c.get("score", 0) for c in creatives) / max(1, len(creatives)), 2)
            st.markdown(f"**Average Group Score:** `{avg_group_score}`")
            st.progress(min(int(avg_group_score * 100), 100))

            for j, creative in enumerate(creatives, start=1):
                with st.container():
                    st.markdown(f"#### Creative {j} — Score: `{creative.get('score', 0)}`")
                    st.progress(min(int(creative.get('score', 0) * 100), 100))
                    st.markdown(f"**Headline:** {creative.get('headline', '')}")
                    st.markdown(f"**Body:** {creative.get('body', '')}")
                    st.markdown(f"**CTA:** {creative.get('cta', '')}")
                    st.divider()

        # Metrics and Insights ---
        metrics_path = None
        for root, _, files in os.walk("outputs"):
            for f in files:
                if f == "metrics.json":
                    metrics_path = os.path.join(root, f)

        if metrics_path and os.path.exists(metrics_path):
            st.markdown("## Metrics Summary")
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            st.json(metrics)

            # Latency
            lat = metrics.get("latency_sec", 0)
            if lat:
                if lat < 3:
                    speed_note = "Fast response"
                elif lat < 10:
                    speed_note = "Moderate response"
                else:
                    speed_note = "Slow response (consider smaller prompts)"
                st.markdown(f"**Latency:** {lat} sec — {speed_note}")

            # Cost estimation
            if "token_usage" in metrics:
                total_tokens = metrics["token_usage"].get("total_tokens", 0)
                estimated_cost = round(total_tokens * 0.000002, 4)
                st.markdown(f" **Estimated API Cost:** `${estimated_cost}`")

            # Timestamp and mode
            from datetime import datetime
            st.caption(f" Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
                    f"Mode: {metrics.get('mode', 'mock').capitalize()} | "
                    f"Model: {metrics.get('model', '')}")

        # --- Download Button ---
        st.download_button(
            label="**Download Output JSON**",
            data=json.dumps(data, indent=2),
            file_name="generated_campaign.json",
            mime="application/json"
        )


    # Optional metrics file preview
    metrics_path = None
    for root, _, files in os.walk("outputs"):
        for f in files:
            if f == "metrics.json":
                metrics_path = os.path.join(root, f)
    if metrics_path and os.path.exists(metrics_path):
        st.markdown("### Metrics Summary")
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        st.json(metrics)
