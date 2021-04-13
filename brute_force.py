"""
Runs a branch and bound brute force on the simplex to find the minimum possible
shortest path.
"""

import heapq
import copy
import gym
import json
from tqdm import tqdm
import argparse
from spicy_env.env_v2 import SpicyGym, available_pivots
from infer import BASELINES

steep_predictor = BASELINES["steepest_edge"]


def carry_through_path(env: SpicyGym, path):

    for p in path[:-1]:
        output = env.step(p)
        assert not output[2] # not done
    
    return env.step(path[-1])


def brute_force_env(env):

    steep_length = 0
    state = env.reset(same_file = True)
    done = False
    while not done:
        action = steep_predictor.predict(state)
        (state, reward, done, info) = env.step(action)
        steep_length += 1
    
    best_length = steep_length

    configurations = [(0, [])]
    configs_tried = 0

    while len(configurations) > 0:
        configs_tried += 1
        config = heapq.heappop(configurations)
        pri, path = config

        if len(path) >= best_length:
            continue

        state = env.reset(same_file = True)
        done = False
        if len(path) > 0:
            (state, reward, done, info) = carry_through_path(env, path)

        if done:
            best_length = min(len(path), best_length)
        else:
            if len(path) <= best_length - 2:
                new_last_elems = available_pivots(state)
                for n in new_last_elems:
                    new_path = copy.deepcopy(path)
                    new_path.append(n)
                    to_insert = (len(new_path), new_path)
                    heapq.heappush(configurations, to_insert)

    
    return (best_length, configs_tried)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--out", required=True)
    parser.add_argument("--data_dir", type=str, required=True)

    args = parser.parse_args()

    env = gym.make(
        "spicy-v0", 
        return_raw_state = True, 
        data_dir = args.data_dir, 
        heuristic = False,  
        full_tableau = True,
        sort_files = True
    )

    brute_force_results = []
    for i in tqdm(range(len(env.data_files))):
    # for i in tqdm(range(10)):

        # Tick the file counter
        env.reset()

        res = brute_force_env(env)
        brute_force_results.append(res)

    
    with open(args.out, "w") as file:
        json.dump(brute_force_results, file)


if __name__ == "__main__":
    main()

