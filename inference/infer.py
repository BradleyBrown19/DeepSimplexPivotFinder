import gym
from spicy_env.env_v2 import dantzigs_rule, steepest_edge_rule
from tqdm import tqdm
import argparse
import json
from pathlib import Path

def do_inference(model, data_dir):

    env = gym.make('spicy-v0', data_dir=data_dir)

    test_set_size = len(env.data_files)

    all_rewards = []

    for i in tqdm(range(test_set_size)):
        total_reward = 0
        initial_state = env.reset()
        action = model.predict(initial_state)
        done = False

        while not done:
            (state, reward, done, info) = env.state(action)
            total_reward += reward
            action = model.predict(state)

        all_rewards.append(total_reward)
    
    return all_rewards



class DantzigBaseline:

    def predict(state):
        return dantzigs_rule(state)


class SteeptestEdgeBaseline:

    def predict(state):
        return steepest_edge_rule(state)


BASELINES = {
    "dantzip" : DantzigBaseline,
    "steepest_edge" : SteeptestEdgeBaseline
}


def main():

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--baseline", choices=BASELINES.keys(), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--data_dir", type=Path, required=True)

    args = parser.parse_args()
    model = BASELINES[args.baseline]()

    results = do_inference(model, args.data_dir)
    print("Mean result: ", sum(results) / len(results))


    with open(args.out, "w") as file:
        json.dump(results, file)


if __name__ == "__main__":
    main()