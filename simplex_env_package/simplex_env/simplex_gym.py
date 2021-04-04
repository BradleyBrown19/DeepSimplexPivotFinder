from pathlib import Path
import pickle
from copy import deepcopy
import gym
from gym import spaces
import numpy as np
import torch
from pandas import *
from scipy.optimize import *
import sys
sys.path.append('/Users/bradleybrown/Desktop/Waterloo/Courses/3A/CO255/DeepSimplexPivotFinder/simplex_env_package/simplex_env')

from solver import SimplexSolver

class SimplexGym(gym.Env):

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data_files = []
        self.data_index = 0
        self.idx = 0
        self.solver = SimplexSolver()
        self.load_data(data_dir)

        fname = self.data_dir / self.data_files[self.data_index]
        print(fname)
        self.data_index = (self.data_index + 1) % len(self.data_files)

        with open(fname, "rb") as file:
            data = pickle.load(file)

        Aarr = np.array(data['A'])
        barr = np.array(data['b'])
        carr = np.array(data['c'])

        num_vertices = int(data_dir.split("_")[-2].split("/")[-1])

        m = num_vertices**2+num_vertices+2
        n = num_vertices**2

        self.action_space = spaces.Discrete(n-1+m)
        self.observation_space = spaces.Discrete((m+n)*n)

    def load_data(self, data_dir):
        self.data_dir = Path(data_dir)
        self.data_files = list(self.data_dir.glob("*"))
    
    def get_feasible_idxs(self, tab):
        """
        Mask for columns where 1 means it is part of the basis or has a postive objective value
        """
        idxs = []
        bottom_row = self.solver.tableau[len(self.solver.tableau) - 1]
        for index, value in enumerate(bottom_row[:-1]):
            if value < 0:
                idxs.append(0)
            else:
                idxs.append(1)
        return idxs

    
    def tab_to_obs(self, tab):
        ret = np.array([tab[i][j] for j in range(len(tab[0])) for i in range(len(tab))], dtype=np.float32)
        return (ret, self.get_feasible_idxs(tab))

    def step(self, action):
        self.solver.step(enter_index=action)

        tableau = deepcopy(self.solver.tableau)
        reward = -1
        done = self.solver.is_done()
        info = {}

        if done:
            print("NUM STEPS: ", self.solver.num_steps)
            print("OBJ VAL: ", round(tableau[-1][-1].numerator / tableau[-1][-1].denominator, 2))
            print("SCIPY VAL: ", round(self.res_val, 2))

            if abs(self.res_val - tableau[-1][-1].numerator / tableau[-1][-1].denominator) > 1e-3:
                print("ERRORR")

        return (self.tab_to_obs(tableau), reward, done, info)

    def reset(self):
        self.idx += 1

        # if (self.idx == 8 or self.idx == 16 or self.idx == 17):
        #     self.data_index += 1

        self.solver = SimplexSolver()

        # if self.data_index == 83:
        #     self.data_index+=1

        fname = self.data_dir / self.data_files[self.data_index]

        self.data_index = (self.data_index + 1) % len(self.data_files)

        with open(fname, "rb") as file:
            data = pickle.load(file)
        
        print("="*100)
        print(fname)
        print(self.idx)
        
        A = data["A"]
        b = data["b"]
        c = data["c"]
        prob = data["prob"]


        res = linprog(c=c, A_ub=A, b_ub=b, method="simplex")
        self.res_val = res['fun']

        A = [[x*-1 for x in row] for row in data["A"]]
        b = [x*-1 for x in data["b"]]
        c = [x for x in data["c"]]
        prob = data["prob"]

        # for idx in range(len(A[0])):
        #     new_row = []
        #     for j in range(len(A[0])):
        #         if j == idx:
        #             new_row.append(1)
        #         else:
        #             new_row.append(0)
        #     A.append(new_row)
        #     b.append(0)

        # print (DataFrame( [[round(r,2) for r in row] for row in A] ))

        # import pdb; pdb.set_trace()

        self.solver.init_simplex(A, b, c, prob="min", do_stop=self.idx == 83)

        assert not self.solver.is_done(), "Done right after starting, hmmmmm...."

        tableau = deepcopy(self.solver.tableau)

        return self.tab_to_obs(tableau)

    def render(self, mode='human'):
        raise NotImplementedError("render THIS <waves arms wildly>")

    def close(self):
        pass
