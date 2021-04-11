# DeepSimplexPivotFinder

Donwload a bunch of shiz
- Spinup, torch, the usual

## Set up Gym

Go to spicy_env_package, 'run pip install -e .'

## Running Experiments

PPO file is in spinup/algos/pytorch/ppo

python -m spinup.run ppo --env Walker2d-v2 --simplex True --exp_name NAME --simplex_data PATH TO DATA WITH PICKLES --heuristic True/False --full_tableau True/False [any other command flags (ex. lr), see ppo.py method for options]

ex. 

python -m spinup.run ppo --env Walker2d-v2 --exp_name sunday_euc4 --simplex True --simplex_data /Users/bradleybrown/Desktop/Waterloo/Courses/3A/CO255/DeepSimplexPivotFinder/data_generation/data/4_euclid

## Testing

python -m spinup.run test_policy spinningup/data/euc_4 --data_dir /Users/bradleybrown/Desktop/Waterloo/Courses/3A/CO255/DeepSimplexPivotFinder/data_generation/data/4_euclid/

PATH_TO_EXPERIMENT_SAVE --data_dir DATA_DIR

ex. python -m spinup.run test_policy spinningup/data/euc_4_sat_noverfit/euc_4_sat_noverfit_s0/ --data_dir /Users/bradleybrown/Desktop/Waterloo/Courses/3A/CO255/DeepSimplexPivotFinder/data_generation/data/4_no_euclid/