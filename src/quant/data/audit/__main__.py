"""
Run log-returns audit via: ``python -m quant.data.audit`` (with ``src`` on PYTHONPATH).
"""

from quant.data.audit.returns import cli

if __name__ == "__main__":
    cli()
