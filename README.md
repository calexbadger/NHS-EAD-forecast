# nhs_ead_forecast

Faculty AI entry for the [SPHERE-PPL NHS Acute Patient Harm Forecasting Competition](https://github.com/SPHERE-PPL/NHS-EAD-forecast).

## Background

This project develops a forecasting algorithm to predict daily estimated avoidable deaths resulting from ED admission delays in the Bristol NHS healthcare system. The goal is to predict avoidable deaths over a 10-day horizon using near-real-time data, evaluated on MSE across 1–5 day and 6–10 day forecast windows.

Submission deadline: **5 June 2026**. Assessment dataset released: **6 June 2026**.

## Project Structure
```
nhs_ead_forecast/
├── data/                  # Competition data (not committed — see below)
├── docs/                  # Project documentation
├── notebooks/             # Exploration and analysis
├── report/                # Competition report (max 1000 words)
├── src/nhs_ead_forecast/  # Source code
├── submission/            # Final forecasts — submitted to competition
└── tests/                 # Unit tests
```

## Getting Started

This project uses `uv` for Python version and dependency management. See [Faculty's uv guide on Notion](https://www.notion.so/facultyai/uv-25c296bcfe388052821be36e24c8297e) if you need to install it.
```bash
git clone git@gitlab.com:facultyai/data-science/nhs_ead_forecast.git
cd nhs_ead_forecast
uv sync
uv run pre-commit install
```

## Data Setup

The competition dataset is not committed to this repo. To get it:

1. Clone the competition repository:
```bash
git clone https://github.com/SPHERE-PPL/NHS-EAD-forecast.git
cd NHS-EAD-forecast
git lfs pull
```

2. Copy the data file into this project:
```bash
cp data/turingAI_forecasting_challenge_dataset.csv.zip /path/to/nhs_ead_forecast/data/
```

The `data/` folder is gitignored — do not commit data files.

## Local Development

- Linting & formatting: `uv run pre-commit run --all-files`
- Tests: `uv run pytest tests/`

## Submission

Forecasts and the accompanying report must be placed in the `submission/` folder following the competition template format. The repo must be public at submission time.

## Notes

- This project was scaffolded using Faculty's [consultancy-cookie](https://gitlab.com/facultyai/faculty-tools/consultancy-cookie).
- Competition rules, data glossary, and evaluation criteria are in the [competition repo](https://github.com/SPHERE-PPL/NHS-EAD-forecast).
