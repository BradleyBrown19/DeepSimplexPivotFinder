# DeepSimplexPivotFinder

## Code Structure

```
.
├── spinningup // PPO algorithm, training and testing
├── data_generation
│   ├── TSP2LP.ipynb // Generation of the data
│   └── data // TSP data 
├── spicy_env_package // Simplex solving gym
├── PolicyExploration.ipynb // Policy interpretation notebook
├── visualization // Notebook to generate figures
```

## Installation

### Go through the setup of spinning up

https://spinningup.openai.com/en/latest/user/installation.html

### Set up Gym

Go to spicy_env_package, 'run pip install -e .'

## Running Experiments

python -m spinup.run ppo --env Walker2d-v2 --simplex True --exp_name NAME --simplex_data PATH TO DATA WITH PICKLES --heuristic True/False --full_tableau True/False [any other command flags (ex. lr), see ppo.py method for options]

ex. 

python -m spinup.run ppo --env Walker2d-v2 --exp_name exp1 --simplex True --simplex_data ./data_generation/data/4_euclid

## Testing

python -m spinup.run test_policy PATH_TO_EXPERIMENT_SAVE --data_dir DATA_DIR

ex. python -m spinup.run test_policy spinningup/data/exp1/exp1_s0 --data_dir ./data_generation/data/4_no_euclid/
