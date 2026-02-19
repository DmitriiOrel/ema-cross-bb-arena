# TradeSavvy Sandbox Starter

Minimal training project to run a trading bot in T-Invest Sandbox.

## Quick Start (Windows PowerShell)

```powershell
git clone https://github.com/DmitriiOrel/winter_school_project.git
cd .\winter_school_project
.\quickstart.ps1 -Token "t.YOUR_API_TOKEN" -Run
```

Token format: use the raw token only (`t.xxxxx`).
Do not wrap token with `< >`.

## What quickstart does

- Creates `.venv`
- Installs dependencies
- Installs T-Invest SDK (`tinkoff.invest`)
- Creates/finds sandbox account
- Writes `.env` with `TOKEN`, `ACCOUNT_ID`, `SANDBOX=True`

## Run later

```powershell
.\run_sandbox.ps1
```

Stop: `Ctrl+C`.

## Backtest + EMA sweep + chart + run sandbox

One command:

```powershell
.\run_backtest_and_sandbox.ps1
```

What it does:
- loads 2 years of candles (`--days-back 730`);
- sweeps EMA short/long ranges;
- picks the best EMA pair by CAGR (with drawdown and trade count tie-breakers);
- builds chart with price + buy/sell entry/exit points;
- saves metrics:
  - max drawdown;
  - number of trades;
  - average annual return (CAGR);
- writes selected EMA params into `instruments_config_scalpel.json`;
- starts sandbox bot with selected EMA.

Artifacts are saved into `reports/`:
- `reports/scalpel_backtest_plot.png`
- `reports/ema_grid_results.csv`
- `reports/best_trades.csv`
- `reports/backtest_summary.json`

## If script execution is blocked

```powershell
powershell -ExecutionPolicy Bypass -File .\quickstart.ps1 -Token "t.YOUR_API_TOKEN" -Run
```

## Notes

- Sandbox only (`SANDBOX=True`): virtual trades.
- Do not commit `.env`, `stats.db`, `market_data_cache`, `reports`.
