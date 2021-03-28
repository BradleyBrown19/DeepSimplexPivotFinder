from pathlib import Path
import pickle
from copy import deepcopy
import gym


from solver import SimplexSolver

class SimplexGym(gym.Env):

    def __init__(self):
        self.data_dir = None
        self.data_files = []
        self.data_index = 0
        self.solver = SimplexSolver()

    def load_data(self, data_dir):
        self.data_dir = Path(data_dir)
        self.data_files = list(data_dir.glob("*"))

    def step(self, action):
        self.solver.step(enter_index=action)

        tableau = deepcopy(self.solver.tableau)
        reward = -1
        done = self.solver.is_done()
        info = {}

        return (tableau, reward, done, info)

    def reset(self):
        fname = self.data_dir / self.data_files[self.data_index]
        self.data_index = (self.data_index + 1) % len(self.data_files)

        with open(fname, "rb") as file:
            data = pickle.load(file)
        
        A = data["A"]
        b = data["b"]
        c = data["c"]
        prob = data["prob"]

        self.solver.init_simplex(A, b, c, prob=prob)

        assert not self.solver.is_done(), "Done right after starting, hmmmmm...."

        tableau = deepcopy(self.solver.tableau)

        return tableau 

    def render(self, mode='human'):
        raise NotImplementedError("render THIS <waves arms wildly>")

    def close(self):
        pass
