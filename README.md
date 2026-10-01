# Football Transfer Value Predictor

![CI](https://github.com/MrThakar/transfer-value-predictor/actions/workflows/ci.yml/badge.svg)

A model that estimates what a footballer is worth from one season of stats,
with a small web app for trying it out. It covers the top five European
leagues and uses Transfermarkt market values as the target.

Football is my favourite sport, and I wanted a project that combined the
software skills I already had with something new to me: machine learning.
Player values seemed like a good fit, because everyone who watches has an
opinion on what a player is worth, and I wanted to see how far stats alone
could get.

![Actual vs. predicted market value](outputs/actual_vs_predicted.png)

## How well it works

I trained on the 2021/22 to 2024/25 seasons (7,800 player-seasons) and held
back 2025/26 (1,964 player-seasons) as a test set the models never saw.

| Model | CV error | Test error | Test R² |
| --- | --- | --- | --- |
| Median baseline | €10.90m | €12.25m | -0.19 |
| Random Forest, basic features | €8.46m | €9.52m | 0.27 |
| Ridge regression | €5.72m | €6.10m | 0.63 |
| Random Forest | €5.39m | €6.05m | 0.64 |
| Gradient Boosting | €4.84m | €5.41m | 0.73 |

Errors are mean absolute error in euros. I picked Gradient Boosting from the
cross-validation column and only looked at the test season after that.

The first version was a Random Forest on goals, assists, minutes, age and
position ("basic features" above). Adding club strength, last season's stats,
detailed position and league helped far more than changing the algorithm did:
with those features even plain Ridge regression beats the original forest.

The Premier League is the hardest league to price (€8.6m error vs €3.8m in
Serie A), since its values are the highest and the most spread out.

![Feature importances](outputs/feature_importance.png)

### What it gets wrong

- It predicts Transfermarkt's estimated value, not what a club would actually
  pay.
- It undervalues the biggest names. Haaland is worth €200m on Transfermarkt
  and the model says €91m. Reputation and contract length aren't in the data.
- Most players show up in more than one season, so the model has usually seen
  an earlier season of a test player. I think that's fair, since in practice
  you would know a player's history, but it will do worse on players who are
  new to these leagues.
- I left out a player's previous market value on purpose. With it the model
  would mostly just repeat last year's number.

## Data

Everything comes from the
[transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets)
project, downloaded with `requests` (about 60 MB, cached after the first run).
I use four tables: match appearances, player profiles, historical valuations
and match results.

I started with football-data.org, but it has no market values and no minutes
played, so I switched.

## How it works

1. Add up each player's league appearances into one row per season. If someone
   moved mid-season they count for the club where they played most.
2. Drop anyone with under 450 league minutes.
3. Use the Transfermarkt valuation closest to the end of the season (15 June,
   within 90 days) as the target. I model `log(1 + value)` because a handful
   of €100m+ players would otherwise dominate.
4. Build 38 features: season totals and per-90 rates, last season's minutes,
   goals and assists, the club's points and goal difference per game, plus
   age, height, position and league.
5. Compare models with expanding-window cross-validation (train on earlier
   seasons, validate on the next one). A random split would let the model
   train on seasons that come after the ones it is tested on.
6. Score every model once on the held-out season.

## Running it

```bash
git clone https://github.com/MrThakar/transfer-value-predictor.git
cd transfer-value-predictor
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
python -m transfer_value.train
```

That downloads the data, trains the models and writes the charts and metrics
to `outputs/` and the model to `models/`. There are a few options:

```bash
python -m transfer_value.train --leagues GB1 --seasons 2022 2023 2024 2025
python -m transfer_value.train --min-minutes 900 --refresh
```

### Web app

```bash
cd frontend && npm install && npm run build && cd ..
uvicorn transfer_value.api:app
```

Then open `http://127.0.0.1:8000`. Change any number and the estimate updates,
along with the real players valued closest to it. FastAPI serves the built
React page and the API from the same process.

When working on the front end, run `uvicorn transfer_value.api:app` in one
terminal and `npm run dev` inside `frontend/` in another, and use
`http://localhost:5173`.

### API

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 24.5, "position": "Attack", "sub_position": "Centre-Forward",
       "league": "GB1", "appearances": 34, "minutes_played": 2900,
       "goals": 18, "assists": 6, "prev_minutes": 2500, "prev_goals": 12,
       "prev_assists": 4, "club_points_per_game": 2.0,
       "club_goal_diff_per_game": 0.9}'
```

```json
{"predicted_value_eur": 84671000.0, "model": "Gradient Boosting"}
```

Only age, position, appearances and minutes are required. The other endpoints
are `/comparables`, `/metadata` and `/health`; `/docs` has the full list.

### Docker

```bash
docker build -t transfer-value .
docker run -p 8000:8000 transfer-value
```

The image builds the front end, trains the model and serves everything on one
port.

### Tests

```bash
python -m pytest
```

The tests run on small made-up tables, so they don't need the real data or a
network connection. GitHub Actions runs them and builds the front end on every
push.

## Layout

```
transfer_value/    Python package
  data.py          downloads and loads the tables
  features.py      builds the player-season rows and features
  model.py         models, cross-validation, scoring
  train.py         runs the whole pipeline
  plots.py         charts
  api.py           FastAPI app
frontend/          React + TypeScript + Tailwind
tests/             pytest
outputs/           charts and metrics from the last training run
```

## To do

- Add xG and xA from FBref or Understat
- Add contract length
- Report error separately for players who are new to the dataset
- Prediction ranges, not just a single number
