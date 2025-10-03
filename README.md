# Song-Popularity-Prediction

UMC301 Kaggle Competition 1 Repository for Song Popularity Prediction.

We are tasked to predict the song popularity of a song based on the given dataset containing various features, for the tabulated csv dataset given to us.
My final submission for the code is present in [main.py](./main.py) or [final_submission.ipynb](./final_submission.ipynb).

## Dataset

The dataset given is present in `./data/` with [train.csv](./data/train.csv) and [test.csv](./data/train.csv).

## Methodology

To predict the song popularity probabilities, I used the following-

- Feature Engineering
- XGBoost
- CatBoost
- Ensemble Averaging
- Optuna(to train Hyperparameters)

I created 2 XGBoost and 1 Catboost model with its hyperparameters, ran them on the dataset and took their ensemble average. I also used Optuna to find better hyperparameters to get good hyperparameters for all the models.

## Reproduce results

The final code is present in [main.py](./main.py) or [final_submission.ipynb](./final_submission.ipynb). To run these, follow the below steps:

1. Clone the repository

```bash
git clone https://github.com/AnirudhG07/Song-Popularity-Prediction
cd Song-Popularity-Prediction
```

2. Create the Environment

You can create the environment using `pyenv` using `requirements.txt` and `pyproject.toml`. The below uses [uv](https://docs.astral.sh/uv/) package manager.

```bash
uv venv --python 3.12 # Create the env
uv sync # to install all the packages from uv.lock and pyproject.toml
uv add --dev ipykernel # To run the jupyter notebook
```

(Optional) You can source the environment with `source .venv/bin/activate`

3. Run the code

You can directly run the code with -

```bash
uv run main.py # without sourcing the env
python3 main.py # After sourcing the env
```

You can also select the kernel of current virtual environment in VS code to run the Jupyter Notebook [final_submission.ipynb](./final_submission.ipynb).

## Other Results and codes

Based on previous attempts and trials, my codes and respective submissions are stored in [submissions/](./submissions/) and [codes](./codes/). Final submission corresponds to the 10th test code.

## Acknowledgments

This was done as a part of Kaggle 1 Competition, UMC301 2025 Course, IISc Bengaluru 