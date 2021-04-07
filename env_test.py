import gym
import sys 
sys.path.append("/Users/bradleybrown/Desktop/Waterloo/Courses/3A/CO255/DeepSimplexPivotFinder/simplex_env")
# import simplex_env

if __name__ == "__main__":
    print("TESTING:")
    env = gym.make("simplex_env:simplex-v0", data_dir="/foo")