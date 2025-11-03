# RadicAI Take-home Assignment Dockerfile ---
# Lightweight Python image
FROM python:3.11-slim

# Prevent Python from buffering logs
ENV PYTHONUNBUFFERED=1

# Set workdir
WORKDIR /workspace

# Copy project files
COPY . .

# Install dependencies (including Streamlit ui)
RUN pip install --no-cache-dir -r requirements.txt

# Default environment variables (can be overridden at runtime)
ENV OPENAI_MODEL=gpt-4o-mini

# Expose optional Streamlit port (if added later)
EXPOSE 8501

# Default command: launch Streamlit UI
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# To run the agent script instead, use:
# CMD ["python", "agent/run_agent.py", "path/to/brief.json", "path/to/output.json", "--live"]
