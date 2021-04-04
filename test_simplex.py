from scipy.optimize import *
import pickle

if __name__ == "__main__":
    id = 4862
    fname = "/Users/bradleybrown/Desktop/Waterloo/Courses/3A/CO255/DeepSimplexPivotFinder/data_generation/data/4_euclid/"+str(id)+".pickle"

    with open(fname, "rb") as file:
        data = pickle.load(file)
    
    A = data["A"]
    b = data["b"]
    # c = data['c']
    c = [-1*d for d in data["c"]]
    prob = data["prob"]

    res = linprog(c=c, A_ub=A, b_ub=b, method="simplex")

    print(res)