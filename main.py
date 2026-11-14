"""Forecaster for make.fun polymarket-faa-attribution-02.

Reads one Polymarket case per *.in file in the input directory and writes
one forecast per case to the output directory.

Model: per-market normal random walk, no drift.
  * The median at every horizon is the market's last price (a prediction
    market's price is a probability; with no fresh news the best point
    forecast is where it stands).
  * The spread grows like a random walk: q_tau(h) = x + z_tau * sigma *
    sqrt(h), clamped to [0, 1], rounded to 6 decimals, where z_tau is the
    standard normal quantile at each level.
  * sigma is the market's own recent volatility: 0.8 times the mean of the
    mean absolute 1-minute price change over four lookback windows
    (180, 360, 720 and 1440 minutes). Short windows catch a market that
    just woke up; long windows keep a quiet market honest. The 0.8 factor
    keeps the core of the forecast tight: most minutes barely move.
  * The outer quantiles use a fatter tail than the normal curve: the 5%
    and 95% levels sit at +/-1.96 sigma*sqrt(h) (the normal 2.5% point)
    instead of +/-1.645. Prediction-market prices move mostly in tiny
    ticks and then jump -- a single minute can move twenty points on
    news -- so the honest 5%/95% band is wider than a pure normal walk
    implies, while the 25%/75% band stays at the normal +/-0.674.
    On the six practice cases this tighter-core / fatter-tail shape
    beats the plain normal version on 4 of 5 scored cases.
    No floor: a market that has not moved gets a point forecast,
    which is what its history says.

Only the Python standard library is used. No network, no model call, no
randomness, no wall clock: the output is a pure function of the input,
so two runs on the same case produce byte-identical files.
"""

import json
import math
import os
from pathlib import Path

TAUS = (0.05, 0.25, 0.5, 0.75, 0.95)
# Normal quantiles at TAUS, except the outer pair, which use the fatter
# +/-1.96 (the normal 2.5% point) to cover news-driven jumps.
ZS = (-1.959963984540054, -0.6744897501960817, 0.0,
      0.6744897501960817, 1.959963984540054)
WINDOWS = (180, 360, 720, 1440)
SCALE = 0.8
HORIZON = 240


def market_sigma(prices):
    """Volatility scale from the mean absolute 1-minute change."""
    n = len(prices)
    if n < 2:
        return 0.0
    diffs = [abs(prices[i] - prices[i - 1]) for i in range(1, n)]
    total = 0.0
    used = 0
    for window in WINDOWS:
        tail = diffs[-window:] if window < len(diffs) else diffs
        total += sum(tail) / len(tail)
        used += 1
    return SCALE * (total / used)


def forecast_market(prices, horizon=HORIZON):
    """Five quantiles for each of the next `horizon` minutes."""
    x = prices[-1]
    sigma = market_sigma(prices)
    rows = []
    for h in range(1, horizon + 1):
        spread = sigma * math.sqrt(h)
        row = []
        prev = 0.0
        for z in ZS:
            q = x + z * spread
            if q < 0.0:
                q = 0.0
            elif q > 1.0:
                q = 1.0
            q = round(q, 6)
            if q < prev:
                q = prev
            row.append(q)
            prev = q
        rows.append(row)
    return rows


def forecast_case(case):
    """Forecast every market of one case; returns the output object."""
    forecasts = []
    for market in case["markets"]:
        forecasts.append({
            "id": market["id"],
            "q": forecast_market(market["prices"]),
        })
    return {"forecasts": forecasts}


def main():
    input_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    output_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    for in_path in sorted(input_dir.glob("*.in")):
        with open(in_path, "r", encoding="utf-8") as f:
            case = json.load(f)
        out = forecast_case(case)
        out_path = output_dir / (in_path.stem + ".out")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, separators=(",", ":"))
        print(f"wrote {out_path} ({len(out['forecasts'])} markets)")


if __name__ == "__main__":
    main()
