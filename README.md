# Multi-Horizon Crude Oil Price Forecasting with Transformers and Agentic AI

See ROADMAP.md for the research plan, timeline and evaluation protocol.

Setup (Python 3.12):
    python -m venv .venv && .venv/Scripts/activate
    pip install -r requirements.txt            # core
    pip install -r requirements-dl.txt         # + torch, transformers, TSFMs
    pip install -r requirements-agents.txt     # + agent stack (vllm: Linux/Colab only)
    # Moirai needs its own venv: pip install -r requirements-moirai.txt
    pytest tests -q

Protocol freeze: tag `protocol-frozen` (week 6). Test-split access is gated on it.
