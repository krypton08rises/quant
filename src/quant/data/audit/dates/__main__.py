"""
Run date / volume audits: ``python -m quant.data.audit.dates missing-dates`` or ``null-volume``.
"""

from __future__ import annotations

import argparse

from quant.data.audit.dates.missing_dates import main as missing_dates_main
from quant.data.audit.dates.null_volume import main as null_volume_main


def cli() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m quant.data.audit.dates",
        description="Bronze data date and volume sanity audits",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser(
        "missing-dates",
        help="Per-symbol missing business dates vs holidays (day bars only)",
    )
    sub.add_parser(
        "null-volume",
        help="Zero-volume rows and symbols with any zero-volume day (day bars only)",
    )
    args = parser.parse_args()
    if args.command == "missing-dates":
        missing_dates_main()
    else:
        null_volume_main()


if __name__ == "__main__":
    cli()
