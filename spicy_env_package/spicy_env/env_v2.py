"""
Modified from Scipy repo:

https://github.com/scipy/scipy/blob/20642e52fb3b158d41e4f7fea0c800ffd42b6955/scipy/optimize/_linprog.py
https://github.com/scipy/scipy/blob/20642e52fb3b158d41e4f7fea0c800ffd42b6955/scipy/optimize/_linprog_simplex.py

"""


import numpy as np
import gym
from pathlib import Path
import pickle
from gym import spaces
from copy import deepcopy
from collections import namedtuple
import scipy.sparse as sps
import sys
import copy
# from pandas import *
sys.path.append('/Users/jacklu/Documents/GitHub/DeepSimplexPivotFinder/spicy_env_package/spicy_env')

from scipy_utils import *
import scipy_utils

messages = {0: "Optimization terminated successfully.",
                1: "Iteration limit reached.",
                2: "Optimization failed. Unable to find a feasible"
                   " starting point.",
                3: "Optimization failed. The problem appears to be unbounded.",
                4: "Optimization failed. Singular matrix encountered."}


def dantzigs_rule(state):
    """
    Picks index with minimum cost.

    NOTE: assumes a valid pivot index exists, else will raise an exception.
    """

    tableau, tol, _ = state

    last_row_vars = tableau[-1, :-1]
    index = np.argmin(last_row_vars)

    assert last_row_vars[index] < -tol, "Gosh darnit, whom'st'd've called this pivot rule where there ain't no valid pivots"
    return index


def norm(x):
    return np.sqrt(np.sum(x * x))


def steepest_edge_rule(state):
    """
    Picks index with minimum cost, scaled by norm of corresponding column. 

    NOTE: assumes a valid pivot index exists, else will raise an exception.
    """

    tableau, tol, _ = state

    num_vars = len(tableau[0]) - 1
    scores = np.ones(num_vars) * np.inf
    for i in range(num_vars):
        raw_cost = tableau[-1][i]

        if raw_cost >= -tol:
            continue

        column_norm = norm(tableau[:-1, i])
        scores[i] = raw_cost / column_norm
    
    index =  np.argmin(scores)

    assert index[scores] < -tol, "AnGErY - no valid pivot exists"
    return index


HEURISTICS = [dantzigs_rule, steepest_edge_rule]


class SpicyGym(gym.Env):
    def __init__(self, data_dir, direct_column_selection = True):
        self.direct_column_selection = direct_column_selection
        self.data_dir = data_dir
        self.cur_state = None
        self.data_files = []
        self.data_index = 0
        self.generator = None
        self.output = None
        self.pivot = None
        self.load_data(data_dir)

        fname = self.data_dir / self.data_files[self.data_index]
        self.data_index = (self.data_index + 1) % len(self.data_files)

        with open(fname, "rb") as file:
            data = pickle.load(file)

        Aarr = np.array(data['A'])
        barr = np.array(data['b'])
        carr = np.array(data['c'])

        try:
            num_vertices = int(data_dir.split("_")[-2].split("/")[-1])
        except:
            num_vertices = int(data_dir.split("_")[-3].split("/")[-1])

        m = num_vertices**2+num_vertices+2
        n = num_vertices**2

        self.action_space = spaces.Discrete(n-1+m)
        self.observation_space = spaces.Discrete((m+1)*(m+n))


    def load_data(self, data_dir):
        self.data_dir = Path(data_dir)
        self.data_files = list(self.data_dir.glob("*"))


    def scipy_to_brad(self, state):
        T, tol, bland = state
        ma = np.ma.masked_where(T[-1, :-1] >= -tol, T[-1, :-1], copy=True)
        
        # non negative is true, else false 
        mult_by_valid = ma.mask

        return (copy.deepcopy(T).flatten(), mult_by_valid)


    def reset(self):
        fname = self.data_dir / self.data_files[self.data_index]
        self.data_index = (self.data_index + 1) % len(self.data_files)

        with open(fname, "rb") as file:
            data = pickle.load(file)
        
        A_ub = data["A"]
        b_ub = data["b"]
        c = data["c"]
        
        self.generator = self.simplex_generator(c, A_ub, b_ub, None, None)
        self.done = False
        self.pivot = None

        try:
            state = next(self.generator)
            self.cur_state = state
        except StopIteration:
            raise Exception("Generator terminated without producing any elements - maybe did everything in phase 1?")

        return self.scipy_to_brad(state)


    def step(self, action):

        if self.direct_column_selection:
            self.pivot = action
        else:
            heuristic = HEURISTICS[action]
            self.pivot = heuristic(self.cur_tableau)


        done = False
        state = None

        try:
            state = next(self.generator)
            self.cur_state = state
        except StopIteration:
            done = True

        reward = -1
        info = {}

        if not done:
            state = self.scipy_to_brad(state)
        else:
            state = None

        return (state, reward, done, info)


    def simplex_generator(self, c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None, options=None, 
                            x0=None, maxiter=1000, bland=False):
        
        meth = "simplex"
        callback = None
          
        ### From linprog() in _linprog.py

        lp = _LPProblem(c, A_ub, b_ub, A_eq, b_eq, bounds, x0)
        lp, solver_options = _parse_linprog(lp, options, meth)
        tol = solver_options.get('tol', 1e-9)

        # Give unmodified problem to HiGHS
        if meth.startswith('highs'):
            if callback is not None:
                raise NotImplementedError("HiGHS solvers do not support the "
                                        "callback interface.")
            highs_solvers = {'highs-ipm': 'ipm', 'highs-ds': 'simplex',
                            'highs': None}

            sol = _linprog_highs(lp, solver=highs_solvers[meth],
                                **solver_options)
            sol['status'], sol['message'] = (
                _check_result(sol['x'], sol['fun'], sol['status'], sol['slack'],
                            sol['con'], lp.bounds, tol, sol['message']))
            sol['success'] = sol['status'] == 0
            return OptimizeResult(sol)

        iteration = 0
        complete = False  # will become True if solved in presolve
        undo = []

        # Keep the original arrays to calculate slack/residuals for original
        # problem.
        lp_o = deepcopy(lp)

        # Solve trivial problem, eliminate variables, tighten bounds, etc.
        rr_method = solver_options.pop('rr_method', None)  # need to pop these;
        rr = solver_options.pop('rr', True)  # they're not passed to methods
        c0 = 0  # we might get a constant term in the objective
        if solver_options.pop('presolve', True):
            (lp, c0, x, undo, complete, status, message) = scipy_utils._presolve(lp, rr,
                                                                    rr_method,
                                                                    tol)

        C, b_scale = 1, 1  # for trivial unscaling if autoscale is not used
        postsolve_args = (lp_o._replace(bounds=lp.bounds), undo, C, b_scale)
            
        if complete:
            x, fun, slack, con = scipy_utils._postsolve(x, postsolve_args, complete)

            status, message = scipy_utils._check_result(x, fun, status, slack, con, lp_o.bounds, tol, message)

            sol = {
                'x': x,
                'fun': fun,
                'slack': slack,
                'con': con,
                'status': status,
                'message': message,
                'nit': iteration,
                'success': status == 0}

            self.output = sol
            return

        if not complete:
            A, b, c, c0, x0 = scipy_utils._get_Abc(lp, c0)
            if solver_options.pop('autoscale', False):
                A, b, c, x0, C, b_scale = scipy_utils._autoscale(A, b, c, x0)
                postsolve_args = postsolve_args[:-2] + (C, b_scale)

            if meth == 'simplex':
                x, status, message, iteration = scipy_utils._linprog_simplex(
                    c, c0=c0, A=A, b=b, callback=callback,
                    postsolve_args=postsolve_args, **solver_options)
            elif meth == 'interior-point':
                x, status, message, iteration = scipy_utils._linprog_ip(
                    c, c0=c0, A=A, b=b, callback=callback,
                    postsolve_args=postsolve_args, **solver_options)
            elif meth == 'revised simplex':
                x, status, message, iteration =scipy_utils. _linprog_rs(
                    c, c0=c0, A=A, b=b, x0=x0, callback=callback,
                    postsolve_args=postsolve_args, **solver_options)


        A, b, c, c0, x0 = _get_Abc(lp, c0)
        if solver_options.pop('autoscale', False):
            A, b, c, x0, C, b_scale = _autoscale(A, b, c, x0)
            postsolve_args = postsolve_args[:-2] + (C, b_scale)


        ### from _linprog_simplex in _linprog_simplex.py

        n, m = A.shape

        # All constraints must have b >= 0.
        is_negative_constraint = np.less(b, 0)
        A[is_negative_constraint] *= -1
        b[is_negative_constraint] *= -1

        # As all constraints are equality constraints the artificial variables
        # will also be basic variables.
        av = np.arange(n) + m
        basis = av.copy()

        # Format the phase one tableau by adding artificial variables and stacking
        # the constraints, the objective row and pseudo-objective row.
        row_constraints = np.hstack((A, np.eye(n), b[:, np.newaxis]))
        row_objective = np.hstack((c, np.zeros(n), c0))
        row_pseudo_objective = -row_constraints.sum(axis=0)
        row_pseudo_objective[av] = 0
        T = np.vstack((row_constraints, row_objective, row_pseudo_objective))

        nit1, status = scipy_utils._solve_simplex(T, n, basis, callback=callback,
                                    postsolve_args=postsolve_args,
                                    maxiter=maxiter, tol=tol, phase=1,
                                    bland=bland
                                    )
        # if pseudo objective is zero, remove the last row from the tableau and
        # proceed to phase 2
        nit2 = nit1
        if abs(T[-1, -1]) < tol:
            # Remove the pseudo-objective row from the tableau
            T = T[:-1, :]
            # Remove the artificial variable columns from the tableau
            T = np.delete(T, av, 1)
        else:
            # Failure to find a feasible starting point
            status = 2
            messages[status] = (
                "Phase 1 of the simplex method failed to find a feasible "
                "solution. The pseudo-objective function evaluates to {0:.1e} "
                "which exceeds the required tolerance of {1} for a solution to be "
                "considered 'close enough' to zero to be a basic solution. "
                "Consider increasing the tolerance to be greater than {0:.1e}. "
                "If this tolerance is unacceptably  large the problem may be "
                "infeasible.".format(abs(T[-1, -1]), tol)
            )

        if status == 0:
            # Phase 2
            for thing in self._solve_simplex_generator(T, n, basis, callback=callback,
                                        postsolve_args=postsolve_args,
                                        maxiter=maxiter, tol=tol, phase=2,
                                        bland=bland, nit0=nit1
                                        ):
                yield thing



            # nit2, status = _solve_simplex_generator(T, n, basis, callback=callback,
            #                             postsolve_args=postsolve_args,
            #                             maxiter=maxiter, tol=tol, phase=2,
            #                             bland=bland, nit0=nit1
            #                             )

        solution = np.zeros(n + m)
        solution[basis[:n]] = T[:n, -1]
        x = solution[:m]

        final_output = (x, status, messages[status], int(nit2))
        self.output = final_output


    def _solve_simplex_generator(self, T, n, basis, callback, postsolve_args,
                    maxiter=1000, tol=1e-9, phase=2, bland=False, nit0=0,
                    ):
        """
        Solve a linear programming problem in "standard form" using the Simplex
        Method. Linear Programming is intended to solve the following problem form:

        Minimize::

            c @ x

        Subject to::

            A @ x == b
                x >= 0

        Parameters
        ----------
        T : 2-D array
            A 2-D array representing the simplex tableau, T, corresponding to the
            linear programming problem. It should have the form:

            [[A[0, 0], A[0, 1], ..., A[0, n_total], b[0]],
            [A[1, 0], A[1, 1], ..., A[1, n_total], b[1]],
            .
            .
            .
            [A[m, 0], A[m, 1], ..., A[m, n_total], b[m]],
            [c[0],   c[1], ...,   c[n_total],    0]]

            for a Phase 2 problem, or the form:

            [[A[0, 0], A[0, 1], ..., A[0, n_total], b[0]],
            [A[1, 0], A[1, 1], ..., A[1, n_total], b[1]],
            .
            .
            .
            [A[m, 0], A[m, 1], ..., A[m, n_total], b[m]],
            [c[0],   c[1], ...,   c[n_total],   0],
            [c'[0],  c'[1], ...,  c'[n_total],  0]]

            for a Phase 1 problem (a problem in which a basic feasible solution is
            sought prior to maximizing the actual objective. ``T`` is modified in
            place by ``_solve_simplex``.
        n : int
            The number of true variables in the problem.
        basis : 1-D array
            An array of the indices of the basic variables, such that basis[i]
            contains the column corresponding to the basic variable for row i.
            Basis is modified in place by _solve_simplex
        callback : callable, optional
            If a callback function is provided, it will be called within each
            iteration of the algorithm. The callback must accept a
            `scipy.optimize.OptimizeResult` consisting of the following fields:

                x : 1-D array
                    Current solution vector
                fun : float
                    Current value of the objective function
                success : bool
                    True only when a phase has completed successfully. This
                    will be False for most iterations.
                slack : 1-D array
                    The values of the slack variables. Each slack variable
                    corresponds to an inequality constraint. If the slack is zero,
                    the corresponding constraint is active.
                con : 1-D array
                    The (nominally zero) residuals of the equality constraints,
                    that is, ``b - A_eq @ x``
                phase : int
                    The phase of the optimization being executed. In phase 1 a basic
                    feasible solution is sought and the T has an additional row
                    representing an alternate objective function.
                status : int
                    An integer representing the exit status of the optimization::

                        0 : Optimization terminated successfully
                        1 : Iteration limit reached
                        2 : Problem appears to be infeasible
                        3 : Problem appears to be unbounded
                        4 : Serious numerical difficulties encountered

                nit : int
                    The number of iterations performed.
                message : str
                    A string descriptor of the exit status of the optimization.
        postsolve_args : tuple
            Data needed by _postsolve to convert the solution to the standard-form
            problem into the solution to the original problem.
        maxiter : int
            The maximum number of iterations to perform before aborting the
            optimization.
        tol : float
            The tolerance which determines when a solution is "close enough" to
            zero in Phase 1 to be considered a basic feasible solution or close
            enough to positive to serve as an optimal solution.
        phase : int
            The phase of the optimization being executed. In phase 1 a basic
            feasible solution is sought and the T has an additional row
            representing an alternate objective function.
        bland : bool
            If True, choose pivots using Bland's rule [3]_. In problems which
            fail to converge due to cycling, using Bland's rule can provide
            convergence at the expense of a less optimal path about the simplex.
        nit0 : int
            The initial iteration number used to keep an accurate iteration total
            in a two-phase problem.

        Returns
        -------
        nit : int
            The number of iterations. Used to keep an accurate iteration total
            in the two-phase problem.
        status : int
            An integer representing the exit status of the optimization::

            0 : Optimization terminated successfully
            1 : Iteration limit reached
            2 : Problem appears to be infeasible
            3 : Problem appears to be unbounded
            4 : Serious numerical difficulties encountered

        """
        nit = nit0
        status = 0
        message = ''
        complete = False

        if phase == 1:
            m = T.shape[1]-2
        elif phase == 2:
            m = T.shape[1]-1
        else:
            raise ValueError("Argument 'phase' to _solve_simplex must be 1 or 2")

        if phase == 2:
            # Check if any artificial variables are still in the basis.
            # If yes, check if any coefficients from this row and a column
            # corresponding to one of the non-artificial variable is non-zero.
            # If found, pivot at this term. If not, start phase 2.
            # Do this for all artificial variables in the basis.
            # Ref: "An Introduction to Linear Programming and Game Theory"
            # by Paul R. Thie, Gerard E. Keough, 3rd Ed,
            # Chapter 3.7 Redundant Systems (pag 102)
            for pivrow in [row for row in range(basis.size)
                        if basis[row] > T.shape[1] - 2]:
                non_zero_row = [col for col in range(T.shape[1] - 1)
                                if abs(T[pivrow, col]) > tol]
                if len(non_zero_row) > 0:
                    pivcol = non_zero_row[0]
                    scipy_utils._apply_pivot(T, basis, pivrow, pivcol, tol)
                    nit += 1

        if len(basis[:m]) == 0:
            solution = np.empty(T.shape[1] - 1, dtype=np.float64)
        else:
            solution = np.empty(max(T.shape[1] - 1, max(basis[:m]) + 1),
                                dtype=np.float64)

        while not complete:
            # Find the pivot column
            pivcol_found, pivcol = scipy_utils._pivot_col(T, tol, bland)
            # import pdb; pdb.set_trace()
            if not pivcol_found:
                pivcol = np.nan
                pivrow = np.nan
                status = 0
                complete = True
            else:

                # ADDED BY US: IGNORE the default pivot chosen and select our own
                yield (T, tol, bland)
                pivcol = self.pivot

                # Find the pivot row
                pivrow_found, pivrow = scipy_utils._pivot_row(T, basis, pivcol, phase, tol, bland)
                if not pivrow_found:
                    status = 3
                    complete = True

            # if callback is not None:
            #     solution[:] = 0
            #     solution[basis[:n]] = T[:n, -1]
            #     x = solution[:m]
            #     x, fun, slack, con = _postsolve(
            #         x, postsolve_args
            #     )
            #     res = OptimizeResult({
            #         'x': x,
            #         'fun': fun,
            #         'slack': slack,
            #         'con': con,
            #         'status': status,
            #         'message': message,
            #         'nit': nit,
            #         'success': status == 0 and complete,
            #         'phase': phase,
            #         'complete': complete,
            #         })
            #     callback(res)

            if not complete:
                if nit >= maxiter:
                    # Iteration limit exceeded
                    status = 1
                    complete = True
                else:
                    scipy_utils._apply_pivot(T, basis, pivrow, pivcol, tol)
                    nit += 1
        
        # return nit, status
