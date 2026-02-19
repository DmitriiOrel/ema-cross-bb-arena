import argparse
import json
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import matplotlib
import pandas as pd
from ta.trend import EMAIndicator
from ta.volatility import BollingerBands
from tinkoff.invest import CandleInterval, Client
from tinkoff.invest.constants import INVEST_GRPC_API, INVEST_GRPC_API_SANDBOX
from tinkoff.invest.utils import now

from app.config import settings
from app.utils.quotation import quotation_to_float

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


SECONDS_IN_YEAR = 365.25 * 24 * 60 * 60


@dataclass
class DetailedBacktest:
    df: pd.DataFrame
    equity: pd.Series
    trades: pd.DataFrame
    entries: list[tuple[pd.Timestamp, float]]
    exits: list[tuple[pd.Timestamp, float]]
    metrics: dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run 2-year scalpel backtest with EMA parameter sweep, "
            "build chart with entry/exit points and export metrics."
        )
    )
    parser.add_argument("--days-back", type=int, default=730)
    parser.add_argument("--fast-start", type=int, default=10)
    parser.add_argument("--fast-end", type=int, default=40)
    parser.add_argument("--fast-step", type=int, default=5)
    parser.add_argument("--slow-start", type=int, default=45)
    parser.add_argument("--slow-end", type=int, default=120)
    parser.add_argument("--slow-step", type=int, default=5)
    parser.add_argument("--backcandles", type=int, default=15)
    parser.add_argument("--initial-capital", type=float, default=100000.0)
    parser.add_argument("--interval", choices=["5min"], default="5min")
    parser.add_argument("--figi", type=str, default="")
    parser.add_argument("--stop-loss-percent", type=float, default=None)
    parser.add_argument(
        "--config-path", type=Path, default=Path("instruments_config_scalpel.json")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--write-live-config", action="store_true")
    return parser.parse_args()


def load_instrument_params(config_path: Path) -> tuple[str, dict]:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    instruments = payload.get("instruments", [])
    if not instruments:
        raise ValueError("No instruments found in instruments_config_scalpel.json")
    first = instruments[0]
    figi = first.get("figi")
    if not figi:
        raise ValueError("Missing figi in instruments_config_scalpel.json")
    params = first.get("strategy", {}).get("parameters", {})
    return figi, params


def fetch_candles(figi: str, days_back: int) -> pd.DataFrame:
    target = INVEST_GRPC_API_SANDBOX if settings.sandbox else INVEST_GRPC_API
    candles = []
    print(
        f"Fetching candles from API for figi={figi}, days_back={days_back}, interval=5min...",
        flush=True,
    )
    with Client(settings.token, target=target) as client:
        for candle in client.get_all_candles(
            figi=figi,
            from_=now() - timedelta(days=days_back),
            to=now(),
            interval=CandleInterval.CANDLE_INTERVAL_5_MIN,
        ):
            candles.append(
                {
                    "Time": candle.time,
                    "Open": quotation_to_float(candle.open),
                    "High": quotation_to_float(candle.high),
                    "Low": quotation_to_float(candle.low),
                    "Close": quotation_to_float(candle.close),
                    "Volume": candle.volume,
                }
            )
            if len(candles) % 1000 == 0:
                print(f"Fetched candles: {len(candles)}", flush=True)
    if not candles:
        raise ValueError("No candles returned from API")
    df = pd.DataFrame(candles)
    df = df[df["High"] != df["Low"]].copy()
    df.sort_values("Time", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def add_signals(
    df: pd.DataFrame, ema_fast_window: int, ema_slow_window: int, backcandles: int
) -> pd.DataFrame:
    bbands = BollingerBands(close=df["Close"], window=14, window_dev=2)
    df = df.copy()
    df["bbihband"] = bbands.bollinger_hband_indicator()
    df["bbilband"] = bbands.bollinger_lband_indicator()
    df["EMA_slow"] = EMAIndicator(
        close=df["Close"], window=ema_slow_window
    ).ema_indicator()
    df["EMA_fast"] = EMAIndicator(
        close=df["Close"], window=ema_fast_window
    ).ema_indicator()

    above = df["EMA_fast"] > df["EMA_slow"]
    below = df["EMA_fast"] < df["EMA_slow"]
    above_all = (
        above.rolling(window=backcandles)
        .apply(lambda x: x.all(), raw=True)
        .fillna(0)
        .astype(bool)
    )
    below_all = (
        below.rolling(window=backcandles)
        .apply(lambda x: x.all(), raw=True)
        .fillna(0)
        .astype(bool)
    )
    df["EMASignal"] = 0
    df.loc[above_all, "EMASignal"] = 2
    df.loc[below_all, "EMASignal"] = 1

    buy = (df["EMASignal"] == 2) & (df["bbilband"] == 1)
    sell = (df["EMASignal"] == 1) & (df["bbihband"] == 1)
    df["TotalSignal"] = 0
    df.loc[buy, "TotalSignal"] = 2
    df.loc[sell, "TotalSignal"] = 1
    return df


def run_backtest(
    df: pd.DataFrame, initial_capital: float, stop_loss_percent: float
) -> DetailedBacktest:
    cash = float(initial_capital)
    quantity = 0.0
    entry_price = 0.0
    entry_time = None
    entry_value = 0.0

    entries: list[tuple[pd.Timestamp, float]] = []
    exits: list[tuple[pd.Timestamp, float]] = []
    trades: list[dict] = []
    equity_points: list[float] = []
    equity_times: list[pd.Timestamp] = []

    for row in df.itertuples(index=False):
        ts = pd.Timestamp(row.Time)
        price = float(row.Close)
        signal = int(row.TotalSignal)

        if quantity == 0.0 and signal == 2 and price > 0:
            quantity = cash / price
            entry_value = cash
            cash = 0.0
            entry_price = price
            entry_time = ts
            entries.append((ts, price))
        elif quantity > 0.0:
            stop_hit = price <= entry_price * (1.0 - stop_loss_percent)
            if signal == 1 or stop_hit:
                cash = quantity * price
                pnl_abs = cash - entry_value
                pnl_pct = pnl_abs / entry_value if entry_value else 0.0
                exits.append((ts, price))
                trades.append(
                    {
                        "entry_time": entry_time,
                        "entry_price": entry_price,
                        "exit_time": ts,
                        "exit_price": price,
                        "pnl_abs": pnl_abs,
                        "pnl_pct": pnl_pct,
                        "reason": "stop_loss" if stop_hit else "signal",
                    }
                )
                quantity = 0.0
                entry_price = 0.0
                entry_time = None
                entry_value = 0.0

        equity_times.append(ts)
        equity_points.append(cash + quantity * price)

    if quantity > 0.0 and not df.empty:
        ts = pd.Timestamp(df.iloc[-1]["Time"])
        price = float(df.iloc[-1]["Close"])
        cash = quantity * price
        pnl_abs = cash - entry_value
        pnl_pct = pnl_abs / entry_value if entry_value else 0.0
        exits.append((ts, price))
        trades.append(
            {
                "entry_time": entry_time,
                "entry_price": entry_price,
                "exit_time": ts,
                "exit_price": price,
                "pnl_abs": pnl_abs,
                "pnl_pct": pnl_pct,
                "reason": "eod",
            }
        )
        equity_points[-1] = cash

    equity = pd.Series(equity_points, index=pd.to_datetime(equity_times), dtype=float)
    if equity.empty:
        raise ValueError("Backtest equity series is empty")

    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    max_drawdown = float(drawdown.min())
    total_return = float(equity.iloc[-1] / initial_capital - 1.0)

    total_seconds = max(
        (equity.index[-1] - equity.index[0]).total_seconds(), 24.0 * 60.0 * 60.0
    )
    years = total_seconds / SECONDS_IN_YEAR
    cagr = float((equity.iloc[-1] / initial_capital) ** (1.0 / years) - 1.0)

    trades_df = pd.DataFrame(trades)
    number_of_trades = int(len(trades_df))
    win_rate = (
        float((trades_df["pnl_pct"] > 0).mean()) if number_of_trades > 0 else 0.0
    )

    metrics = {
        "initial_capital": initial_capital,
        "final_equity": float(equity.iloc[-1]),
        "total_return": total_return,
        "cagr": cagr,
        "average_annual_return": cagr,
        "max_drawdown": max_drawdown,
        "number_of_trades": number_of_trades,
        "win_rate": win_rate,
    }
    return DetailedBacktest(
        df=df,
        equity=equity,
        trades=trades_df,
        entries=entries,
        exits=exits,
        metrics=metrics,
    )


def evaluate_combo(
    base_df: pd.DataFrame,
    ema_fast_window: int,
    ema_slow_window: int,
    backcandles: int,
    initial_capital: float,
    stop_loss_percent: float,
) -> dict | None:
    if ema_fast_window >= ema_slow_window:
        return None
    signaled_df = add_signals(
        base_df,
        ema_fast_window=ema_fast_window,
        ema_slow_window=ema_slow_window,
        backcandles=backcandles,
    )
    detailed = run_backtest(
        signaled_df,
        initial_capital=initial_capital,
        stop_loss_percent=stop_loss_percent,
    )
    result = dict(detailed.metrics)
    result["ema_fast_window"] = ema_fast_window
    result["ema_slow_window"] = ema_slow_window
    return result


def pick_best(results_df: pd.DataFrame) -> pd.Series:
    ranked = results_df.sort_values(
        by=["cagr", "max_drawdown", "number_of_trades"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    return ranked.iloc[0]


def build_plot(
    detailed: DetailedBacktest,
    figi: str,
    output_path: Path,
    ema_fast_window: int,
    ema_slow_window: int,
) -> None:
    fig, (ax_price, ax_equity) = plt.subplots(
        2, 1, figsize=(18, 10), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    ax_price.plot(detailed.df["Time"], detailed.df["Close"], label=f"{figi} Close")
    if detailed.entries:
        buy_t, buy_p = zip(*detailed.entries)
        ax_price.scatter(buy_t, buy_p, marker="^", color="green", s=80, label="Buy")
    if detailed.exits:
        sell_t, sell_p = zip(*detailed.exits)
        ax_price.scatter(
            sell_t, sell_p, marker="v", color="red", s=80, label="Sell/Exit"
        )
    ax_price.set_title(
        (
            f"Scalpel Backtest | EMA {ema_fast_window}/{ema_slow_window} | "
            f"CAGR {detailed.metrics['cagr']*100:.2f}% | "
            f"Max DD {detailed.metrics['max_drawdown']*100:.2f}% | "
            f"Trades {detailed.metrics['number_of_trades']}"
        )
    )
    ax_price.set_ylabel("Price")
    ax_price.grid(True, alpha=0.2)
    ax_price.legend(loc="upper left")

    normalized_equity = detailed.equity / detailed.equity.iloc[0]
    ax_equity.plot(normalized_equity.index, normalized_equity.values, color="#1f77b4")
    ax_equity.set_ylabel("Equity (x)")
    ax_equity.set_xlabel("Time")
    ax_equity.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def update_live_config(
    config_path: Path, ema_fast_window: int, ema_slow_window: int
) -> None:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["instruments"][0]["strategy"]["parameters"][
        "ema_fast_window"
    ] = ema_fast_window
    payload["instruments"][0]["strategy"]["parameters"][
        "ema_slow_window"
    ] = ema_slow_window
    config_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    figi_from_cfg, params = load_instrument_params(args.config_path)
    figi = args.figi or figi_from_cfg
    stop_loss_percent = (
        args.stop_loss_percent
        if args.stop_loss_percent is not None
        else float(params.get("stop_loss_percent", 0.05))
    )

    fast_values = list(range(args.fast_start, args.fast_end + 1, args.fast_step))
    slow_values = list(range(args.slow_start, args.slow_end + 1, args.slow_step))

    print("Starting backtest pipeline...", flush=True)
    print(
        (
            f"EMA fast range: {args.fast_start}..{args.fast_end} step {args.fast_step}; "
            f"EMA slow range: {args.slow_start}..{args.slow_end} step {args.slow_step}; "
            f"backcandles={args.backcandles}"
        ),
        flush=True,
    )

    base_df = fetch_candles(figi=figi, days_back=args.days_back)
    print(f"Total candles fetched: {len(base_df)}", flush=True)
    required_len = max(slow_values) + args.backcandles + 5
    if len(base_df) < required_len:
        raise ValueError(
            f"Not enough candles for sweep: got {len(base_df)}, required at least {required_len}"
        )

    results = []
    total = sum(1 for f in fast_values for s in slow_values if f < s)
    processed = 0
    for fast in fast_values:
        for slow in slow_values:
            if fast >= slow:
                continue
            processed += 1
            item = evaluate_combo(
                base_df=base_df,
                ema_fast_window=fast,
                ema_slow_window=slow,
                backcandles=args.backcandles,
                initial_capital=args.initial_capital,
                stop_loss_percent=stop_loss_percent,
            )
            if item is not None:
                results.append(item)
            print(f"[{processed}/{total}] checked EMA {fast}/{slow}", flush=True)

    if not results:
        raise ValueError("EMA sweep returned no valid combinations")

    print("Sweep complete. Selecting best EMA pair...", flush=True)
    results_df = pd.DataFrame(results)
    results_df.sort_values(by="cagr", ascending=False, inplace=True)
    grid_path = output_dir / "ema_grid_results.csv"
    results_df.to_csv(grid_path, index=False)

    best_row = pick_best(results_df)
    best_fast = int(best_row["ema_fast_window"])
    best_slow = int(best_row["ema_slow_window"])
    best_df = add_signals(
        base_df, best_fast, best_slow, backcandles=args.backcandles
    )
    best = run_backtest(
        best_df,
        initial_capital=args.initial_capital,
        stop_loss_percent=stop_loss_percent,
    )

    trades_path = output_dir / "best_trades.csv"
    best.trades.to_csv(trades_path, index=False)

    plot_path = output_dir / "scalpel_backtest_plot.png"
    build_plot(
        detailed=best,
        figi=figi,
        output_path=plot_path,
        ema_fast_window=best_fast,
        ema_slow_window=best_slow,
    )

    summary = {
        "figi": figi,
        "days_back": args.days_back,
        "best_ema_fast_window": best_fast,
        "best_ema_slow_window": best_slow,
        "cagr": best.metrics["cagr"],
        "average_annual_return": best.metrics["average_annual_return"],
        "max_drawdown": best.metrics["max_drawdown"],
        "number_of_trades": best.metrics["number_of_trades"],
        "win_rate": best.metrics["win_rate"],
        "grid_results_csv": str(grid_path),
        "trades_csv": str(trades_path),
        "plot_png": str(plot_path),
    }
    summary_path = output_dir / "backtest_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    if args.write_live_config:
        update_live_config(
            config_path=args.config_path,
            ema_fast_window=best_fast,
            ema_slow_window=best_slow,
        )

    print("\n=== Backtest Summary ===")
    print(f"FIGI: {figi}")
    print(f"Best EMA: {best_fast}/{best_slow}")
    print(f"CAGR: {best.metrics['cagr']*100:.2f}%")
    print(f"Average annual return: {best.metrics['average_annual_return']*100:.2f}%")
    print(f"Max drawdown: {best.metrics['max_drawdown']*100:.2f}%")
    print(f"Number of trades: {best.metrics['number_of_trades']}")
    print(f"Win rate: {best.metrics['win_rate']*100:.2f}%")
    print(f"Grid results: {grid_path}")
    print(f"Trades: {trades_path}")
    print(f"Plot: {plot_path}")
    print(f"Summary JSON: {summary_path}")


if __name__ == "__main__":
    main()
