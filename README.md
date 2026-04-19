# ForecastLab-AI

ForecastLab-AI is a small CLI forecasting workflow built around pandas, Prophet, and XGBoost. It prepares time-series data, trains from local CSV or Parquet inputs, writes a model artifact, and exports prediction results.

## Requirements

- Python 3.11 or 3.12
- Poetry

## Installation

```bash
poetry install --no-root
```

Or use the Make target:

```bash
make install
```

## Project Layout

```text
input/
	config/
		hyperparameters.json
	data/
		train/
			train.csv
		test/
			test.csv
output/
	model.pkl
	model.pkl.meta.json
	predictions.csv
	predictions.csv.meta.json
src/
	core/
		prepare.py
		trainer.py
		predictor.py
```

Default paths are defined in `src/utils/settings.py`.

## Data Format

The sample dataset is multi-series and uses these columns:

- `TSId`: time-series identifier
- `ds`: timestamp column
- `y`: target value
- `TestIndicator`: `0` for train/history rows and `1` for holdout/test rows

Training data is expected to contain both historical rows and holdout rows in a single file. The test file contains only the holdout portion.

Example training rows:

```csv
TSId,ds,y,TestIndicator
series_0,2022-01-01,118,0
series_0,2022-02-01,124,0
series_1,2022-01-01,92,0
series_0,2025-01-01,149,1
series_1,2025-01-01,114,1
```

Example test rows:

```csv
TSId,ds,y,TestIndicator
series_0,2025-01-01,149,1
series_0,2025-02-01,156,1
series_1,2025-01-01,114,1
```

Supported input formats are `.csv` and `.parquet`.

## Configuration

The main configuration file is `input/config/hyperparameters.json`.

Key fields:

- `frequency`: `daily`, `weekly`, or `monthly`
- `algorithm.name`: `xgboost`, `prophet`, or `auto`
- `prophet`: Prophet-specific settings
- `xgboost.model_parameters`: raw XGBoost regressor parameters
- `xgboost.level_method`: currently `mean` or `median`
- `xgboost.exogenous`: optional numerical and categorical columns
- `probabilistic_forecast.quantiles`: optional quantiles for quantile loss
- `seasonality`: `auto` or a custom seasonality definition

Current sample config defaults to monthly frequency with XGBoost.

## Usage

### Train

```bash
poetry run python main.py train
```

Or:

```bash
make train
```

Optional arguments:

```bash
poetry run python main.py train \
	--hyperparameters input/config/hyperparameters.json \
	--data input/data/train/train.csv \
	--output output/model.pkl
```

### Predict

```bash
poetry run python main.py predict
```

Or:

```bash
make predict
```

Optional arguments:

```bash
poetry run python main.py predict \
	--artifact output/model.pkl \
	--data input/data/test/test.csv \
	--output output/predictions.csv
```

## Current Prediction Behavior

The current workflow generates predictions during training and stores them inside the saved artifact.

- `trainer.py` creates the predictions dataframe and saves it in the artifact under `predictions`.
- `predictor.py` performs inference by loading those stored predictions and optionally filtering them to the rows requested in the input file, typically by `TSId` and `ds`.
- The CLI still accepts `--periods` and `--include-history`, but they are currently ignored because predictor execution does not run the underlying forecasting model again.

This means `predict` is best understood as artifact-backed inference/export, not model re-scoring from scratch.

## Outputs

By default the workflow writes:

- `output/model.pkl`: serialized training artifact
- `output/model.pkl.meta.json`: metadata about the artifact
- `output/predictions.csv`: exported predictions
- `output/predictions.csv.meta.json`: prediction export metadata

Current sample output metadata shows:

- Training artifact input header: `TSId`, `ds`, `y`, `TestIndicator`
- Prediction output columns: `TSId`, `ds`, `y`, `yhat`

## Development

Run tests:

```bash
make test
```

Run lint:

```bash
make lint
```

## Notes

- Column names are normalized by trimming whitespace.
- `ds` is parsed as datetime.
- Empty input files are rejected.
- The sample repository currently includes two series: `series_0` and `series_1`.