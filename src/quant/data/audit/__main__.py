"""
Run log-returns audit via: ``python -m quant.data.audit`` (from project root with ``src`` on PYTHONPATH).
"""

from quant.data.audit.log_returns import cli

if __name__ == "__main__":
    cli()
