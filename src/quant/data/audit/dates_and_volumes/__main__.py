"""
Run date / volume audits:
    python -m quant.data.audit.dates_and_volumes missing-dates [--interval day]
    python -m quant.data.audit.dates_and_volumes null-volume   [--interval day]
"""

from __future__ import annotations

import argparse

from quant.data._common import Interval
from quant.data.audit.dates_and_volumes.missing_dates import main as missing_dates_main
from quant.data.audit.dates_and_volumes.null_volume import main as null_volume_main

_INTERVAL_CHOICES = [i.value for i in Interval]


def cli() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m quant.data.audit.dates_and_volumes",
        description="Bronze data bar-coverage and volume sanity audits",
    )
    parser.add_argument(
        "--interval",
        choices=_INTERVAL_CHOICES,
        default=Interval.DAY.value,
        help="Bar interval to audit (default: day)",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("missing-dates", help="Per-symbol missing bars vs holidays")
    sub.add_parser("null-volume", help="Zero-volume bars and symbols with any zero-volume bar")

    args = parser.parse_args()
    interval = Interval(args.interval)

    if args.command == "missing-dates":
        missing_dates_main(interval)
    else:
        null_volume_main(interval)


if __name__ == "__main__":
    cli()
