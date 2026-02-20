# MarketLab Arena

Учебный репозиторий для бэктеста и запуска торговой стратегии в T-Invest Sandbox с общим лидербордом.

<!-- LEADERBOARD:START -->
## Актуальный Лидерборд

Автоматически обновляется после каждого бэктеста. Последнее обновление: `20260220T123500Z` UTC.

| Место | Участник | CAGR % | Макс. просадка % | Сделки | EMA Fast | EMA Slow | BB Window | BB Dev | ТФ (мин) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | dmitrii | 39.50 | -11.83 | 81 | 30 | 40 | 20 | 1.0 | 60 |
| 2 | dispatch_smoke | 11.11 | -7.77 | 33 | 20 | 50 | 20 | 2 | 60 |

<!-- LEADERBOARD:END -->

## Быстрый старт (Windows PowerShell)

```powershell
git clone https://github.com/DmitriiOrel/marketlab-arena.git
cd .\marketlab-arena
.\quickstart.ps1 -Token "t.ВАШ_TINVEST_TOKEN"
```

## Запуск эксперимента

```powershell
$env:GITHUB_TOKEN="github_pat_ВАШ_GITHUB_PAT"
.\run_backtest_manual.ps1 20 50 20 2.0 60 -Name ivan
```

## Что изменено для нагрузки до 100 участников

- Клиентский скрипт больше не редактирует `README.md` и `reports/leaderboard.json` напрямую.
- Клиент отправляет `repository_dispatch` событие в GitHub.
- Обновление лидерборда выполняет GitHub Action в одном потоке (`concurrency`), поэтому нет гонок записи.
- В `run_backtest_manual.ps1` добавлен случайный стартовый джиттер (`StartJitterSec`) для сглаживания пиков к API.
- По умолчанию бэктест запускается на `DaysBack=365` (можно увеличить параметром).

## Важное ограничение

Для масштабирования до 100 человек каждый участник должен использовать **свой T-Invest токен**.
Один общий токен T-Invest при массовом параллельном запуске упирается в лимиты API.

## Основные файлы

- `run_backtest_manual.ps1` — запуск бэктеста, отправка результата в GitHub, запуск sandbox-бота.
- `tools/manual_backtest_leaderboard.py` — расчет стратегии и отправка `repository_dispatch`.
- `.github/workflows/leaderboard-dispatch.yml` — серверное обновление лидерборда.
- `tools/apply_submission_event.py` — пересчет `reports/leaderboard.json` и блока в `README.md`.
- `reports/leaderboard.json` — источник данных лидерборда.
