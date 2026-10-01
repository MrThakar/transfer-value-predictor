# Football Transfer Value Predictor

![CI](https://github.com/MrThakar/transfer-value-predictor/actions/workflows/ci.yml/badge.svg)
[![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-20232A?style=flat&logo=react&logoColor=61DAFB)](https://react.dev/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)

**Live demo:** https://transfer-value-predictor.onrender.com (free hosting, so
it can take about a minute to wake up)

A model that estimates what a footballer is worth from one season of stats,
with a small web app for trying it out. It covers the top five European
leagues and uses Transfermarkt market values as the target.

Football is my favourite sport, and I wanted a project that combined the
software skills I already had with something new to me: machine learning.
Player values seemed like a good fit, because everyone who watches has an
opinion on what a player is worth, and I wanted to see how far stats alone
could get.

**Built with:** Python, pandas, scikit-learn, Matplotlib, FastAPI, React,
TypeScript, Tailwind, pytest, GitHub Actions, Docker

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

## The app and the engineering around it

- A FastAPI service wraps the model. `POST /predict` takes a player's season
  and returns a value, with the inputs validated by Pydantic.
- A React + TypeScript + Tailwind page calls that API. Change any number and
  the estimate updates, along with the real players valued closest to it.
- 20 pytest tests cover the feature building, the model code and the API. They
  run on small made-up tables, so they don't need the real data.
- GitHub Actions runs the tests and builds the front end on every push.
- A Dockerfile builds the front end, trains the model and serves everything
  from one container.

## To do

- Add xG and xA from FBref or Understat
- Add contract length
- Report error separately for players who are new to the dataset
- Prediction ranges, not just a single number

## Running it

```bash
pip install -r requirements-dev.txt
python -m transfer_value.train
cd frontend && npm install && npm run build && cd ..
uvicorn transfer_value.api:app
```

Then open `http://127.0.0.1:8000`.
