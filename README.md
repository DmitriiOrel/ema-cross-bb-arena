# EMA Cross BB Arena

Учебный репозиторий для бэктеста и запуска стратегии в T-Invest Sandbox.
Сигнал построен строго на пересечении EMA и выходе цены за полосы Боллинджера.

<!-- LEADERBOARD:START -->
## Актуальный Лидерборд

Автоматически обновляется после каждого бэктеста. Последнее обновление: `20260220T123500Z` UTC.

| Место | Участник | CAGR % | Макс. просадка % | Сделки | EMA Fast | EMA Slow | BB Window | BB Dev | ТФ (мин) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | dmitrii | 39.50 | -11.83 | 81 | 30 | 40 | 20 | 1.0 | 60 |

<!-- LEADERBOARD:END -->

## Быстрый старт (Windows PowerShell)

```powershell
git clone https://github.com/DmitriiOrel/ema-cross-bb-arena.git
cd .\ema-cross-bb-arena
.\quickstart.ps1 -Token "t.ВАШ_TINVEST_TOKEN"
```

## Запуск эксперимента

```powershell
$env:GITHUB_TOKEN="github_pat_ВАШ_GITHUB_PAT"
.\run_backtest_manual.ps1 30 40 20 1.0 60 -Name ivan
```

## Логика сигнала (EMA crossover + Bollinger breakout)

Параметры `30 40 20 1.0 60` означают:
- `EMA_fast = 30`
- `EMA_slow = 40`
- `BB window = 20`
- `BB dev = 1.0`
- `timeframe = 60` минут

Правила:
- `BUY`: быстрая EMA пересекает медленную **снизу вверх** на текущей свече, и цена закрытия **выше верхней** полосы Боллинджера.
- `SELL`: быстрая EMA пересекает медленную **сверху вниз** на текущей свече, и цена закрытия **ниже нижней** полосы Боллинджера.

Стратегия рассматривает только случаи, когда цена вышла за границы полос Боллинджера с заданным стандартным отклонением (`bb_dev`).

## Масштабирование до 100 участников

- Клиентский скрипт не редактирует `README.md`/`reports/leaderboard.json` напрямую.
- Клиент отправляет `repository_dispatch` в GitHub.
- Лидерборд обновляет GitHub Action в одном потоке (`concurrency`), без гонок записи.
- В `run_backtest_manual.ps1` есть стартовый джиттер (`StartJitterSec`) и `DaysBack` (по умолчанию `365`) для снижения нагрузки на API.

## Ограничение

Для нагрузки в 100 человек каждому участнику нужен **свой T-Invest токен**.
Один общий токен упрется в лимиты API.

## Основные файлы

- `run_backtest_manual.ps1` — запуск бэктеста, отправка результата в GitHub, запуск sandbox-бота.
- `tools/manual_backtest_leaderboard.py` — расчет стратегии и отправка `repository_dispatch`.
- `.github/workflows/leaderboard-dispatch.yml` — серверное обновление лидерборда.
- `tools/apply_submission_event.py` — пересчет `reports/leaderboard.json` и блока в `README.md`.
- `reports/leaderboard.json` — источник данных лидерборда.
