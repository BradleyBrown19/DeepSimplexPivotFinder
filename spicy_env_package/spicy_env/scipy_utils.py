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


# _LPProblem = namedtuple('_LPProblem', 'c A_ub b_ub A_eq b_eq bounds x0')


def _get_Abc(lp, c0):
    """
    Given a linear programming problem of the form:

    Minimize::

        c @ x

    Subject to::

        A_ub @ x <= b_ub
        A_eq @ x == b_eq
         lb <= x <= ub

    where ``lb = 0`` and ``ub = None`` unless set in ``bounds``.

    Return the problem in standard form:

    Minimize::

        c @ x

    Subject to::

        A @ x == b
            x >= 0

    by adding slack variables and making variable substitutions as necessary.

    Parameters
    ----------
    lp : A `scipy.optimize._linprog_util._LPProblem` consisting of the following fields:

        c : 1D array
            The coefficients of the linear objective function to be minimized.
        A_ub : 2D array, optional
            The inequality constraint matrix. Each row of ``A_ub`` specifies the
            coefficients of a linear inequality constraint on ``x``.
        b_ub : 1D array, optional
            The inequality constraint vector. Each element represents an
            upper bound on the corresponding value of ``A_ub @ x``.
        A_eq : 2D array, optional
            The equality constraint matrix. Each row of ``A_eq`` specifies the
            coefficients of a linear equality constraint on ``x``.
        b_eq : 1D array, optional
            The equality constraint vector. Each element of ``A_eq @ x`` must equal
            the corresponding element of ``b_eq``.
        bounds : 2D array
            The bounds of ``x``, lower bounds in the 1st column, upper
            bounds in the 2nd column. The bounds are possibly tightened
            by the presolve procedure.
        x0 : 1D array, optional
            Guess values of the decision variables, which will be refined by
            the optimization algorithm. This argument is currently used only by the
            'revised simplex' method, and can only be used if `x0` represents a
            basic feasible solution.

    c0 : float
        Constant term in objective function due to fixed (and eliminated)
        variables.

    Returns
    -------
    A : 2-D array
        2-D array such that ``A`` @ ``x``, gives the values of the equality
        constraints at ``x``.
    b : 1-D array
        1-D array of values representing the RHS of each equality constraint
        (row) in A (for standard form problem).
    c : 1-D array
        Coefficients of the linear objective function to be minimized (for
        standard form problem).
    c0 : float
        Constant term in objective function due to fixed (and eliminated)
        variables.
    x0 : 1-D array
        Starting values of the independent variables, which will be refined by
        the optimization algorithm

    References
    ----------
    .. [9] Bertsimas, Dimitris, and J. Tsitsiklis. "Introduction to linear
           programming." Athena Scientific 1 (1997): 997.

    """
    c, A_ub, b_ub, A_eq, b_eq, bounds, x0 = lp

    if sps.issparse(A_eq):
        sparse = True
        A_eq = sps.csr_matrix(A_eq)
        A_ub = sps.csr_matrix(A_ub)

        def hstack(blocks):
            return sps.hstack(blocks, format="csr")

        def vstack(blocks):
            return sps.vstack(blocks, format="csr")

        zeros = sps.csr_matrix
        eye = sps.eye
    else:
        sparse = False
        hstack = np.hstack
        vstack = np.vstack
        zeros = np.zeros
        eye = np.eye

    # Variables lbs and ubs (see below) may be changed, which feeds back into
    # bounds, so copy.
    bounds = np.array(bounds, copy=True)

    # modify problem such that all variables have only non-negativity bounds
    lbs = bounds[:, 0]
    ubs = bounds[:, 1]
    m_ub, n_ub = A_ub.shape

    lb_none = np.equal(lbs, -np.inf)
    ub_none = np.equal(ubs, np.inf)
    lb_some = np.logical_not(lb_none)
    ub_some = np.logical_not(ub_none)

    # unbounded below: substitute xi = -xi' (unbounded above)
    # if -inf <= xi <= ub, then -ub <= -xi <= inf, so swap and invert bounds
    l_nolb_someub = np.logical_and(lb_none, ub_some)
    i_nolb = np.nonzero(l_nolb_someub)[0]
    lbs[l_nolb_someub], ubs[l_nolb_someub] = (
        -ubs[l_nolb_someub], -lbs[l_nolb_someub])
    lb_none = np.equal(lbs, -np.inf)
    ub_none = np.equal(ubs, np.inf)
    lb_some = np.logical_not(lb_none)
    ub_some = np.logical_not(ub_none)
    c[i_nolb] *= -1
    if x0 is not None:
        x0[i_nolb] *= -1
    if len(i_nolb) > 0:
        if A_ub.shape[0] > 0:  # sometimes needed for sparse arrays... weird
            A_ub[:, i_nolb] *= -1
        if A_eq.shape[0] > 0:
            A_eq[:, i_nolb] *= -1

    # upper bound: add inequality constraint
    i_newub, = ub_some.nonzero()
    ub_newub = ubs[ub_some]
    n_bounds = len(i_newub)
    if n_bounds > 0:
        shape = (n_bounds, A_ub.shape[1])
        if sparse:
            idxs = (np.arange(n_bounds), i_newub)
            A_ub = vstack((A_ub, sps.csr_matrix((np.ones(n_bounds), idxs),
                                                shape=shape)))
        else:
            A_ub = vstack((A_ub, np.zeros(shape)))
            A_ub[np.arange(m_ub, A_ub.shape[0]), i_newub] = 1
        b_ub = np.concatenate((b_ub, np.zeros(n_bounds)))
        b_ub[m_ub:] = ub_newub

    A1 = vstack((A_ub, A_eq))
    b = np.concatenate((b_ub, b_eq))
    c = np.concatenate((c, np.zeros((A_ub.shape[0],))))
    if x0 is not None:
        x0 = np.concatenate((x0, np.zeros((A_ub.shape[0],))))
    # unbounded: substitute xi = xi+ + xi-
    l_free = np.logical_and(lb_none, ub_none)
    i_free = np.nonzero(l_free)[0]
    n_free = len(i_free)
    c = np.concatenate((c, np.zeros(n_free)))
    if x0 is not None:
        x0 = np.concatenate((x0, np.zeros(n_free)))
    A1 = hstack((A1[:, :n_ub], -A1[:, i_free]))
    c[n_ub:n_ub+n_free] = -c[i_free]
    if x0 is not None:
        i_free_neg = x0[i_free] < 0
        x0[np.arange(n_ub, A1.shape[1])[i_free_neg]] = -x0[i_free[i_free_neg]]
        x0[i_free[i_free_neg]] = 0

    # add slack variables
    A2 = vstack([eye(A_ub.shape[0]), zeros((A_eq.shape[0], A_ub.shape[0]))])

    A = hstack([A1, A2])

    # lower bound: substitute xi = xi' + lb
    # now there is a constant term in objective
    i_shift = np.nonzero(lb_some)[0]
    lb_shift = lbs[lb_some].astype(float)
    c0 += np.sum(lb_shift * c[i_shift])
    if sparse:
        b = b.reshape(-1, 1)
        A = A.tocsc()
        b -= (A[:, i_shift] * sps.diags(lb_shift)).sum(axis=1)
        b = b.ravel()
    else:
        b -= (A[:, i_shift] * lb_shift).sum(axis=1)
    if x0 is not None:
        x0[i_shift] -= lb_shift

    return A, b, c, c0, x0



def _linprog_simplex(c, c0, A, b, callback, postsolve_args,
                     maxiter=1000, tol=1e-9, disp=False, bland=False,
                     **unknown_options):
    """
    Minimize a linear objective function subject to linear equality and
    non-negativity constraints using the two phase simplex method.
    Linear programming is intended to solve problems of the following form:

    Minimize::

        c @ x

    Subject to::

        A @ x == b
            x >= 0

    User-facing documentation is in _linprog_doc.py.

    Parameters
    ----------
    c : 1-D array
        Coefficients of the linear objective function to be minimized.
    c0 : float
        Constant term in objective function due to fixed (and eliminated)
        variables. (Purely for display.)
    A : 2-D array
        2-D array such that ``A @ x``, gives the values of the equality
        constraints at ``x``.
    b : 1-D array
        1-D array of values representing the right hand side of each equality
        constraint (row) in ``A``.
    callback : callable, optional
        If a callback function is provided, it will be called within each
        iteration of the algorithm. The callback function must accept a single
        `scipy.optimize.OptimizeResult` consisting of the following fields:

            x : 1-D array
                Current solution vector
            fun : float
                Current value of the objective function
            success : bool
                True when an algorithm has completed successfully.
            slack : 1-D array
                The values of the slack variables. Each slack variable
                corresponds to an inequality constraint. If the slack is zero,
                the corresponding constraint is active.
            con : 1-D array
                The (nominally zero) residuals of the equality constraints,
                that is, ``b - A_eq @ x``
            phase : int
                The phase of the algorithm being executed.
            status : int
                An integer representing the status of the optimization::

                     0 : Algorithm proceeding nominally
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

    Options
    -------
    maxiter : int
       The maximum number of iterations to perform.
    disp : bool
        If True, print exit status message to sys.stdout
    tol : float
        The tolerance which determines when a solution is "close enough" to
        zero in Phase 1 to be considered a basic feasible solution or close
        enough to positive to serve as an optimal solution.
    bland : bool
        If True, use Bland's anti-cycling rule [3]_ to choose pivots to
        prevent cycling. If False, choose pivots which should lead to a
        converged solution more quickly. The latter method is subject to
        cycling (non-convergence) in rare instances.
    unknown_options : dict
        Optional arguments not used by this particular solver. If
        `unknown_options` is non-empty a warning is issued listing all
        unused options.

    Returns
    -------
    x : 1-D array
        Solution vector.
    status : int
        An integer representing the exit status of the optimization::

         0 : Optimization terminated successfully
         1 : Iteration limit reached
         2 : Problem appears to be infeasible
         3 : Problem appears to be unbounded
         4 : Serious numerical difficulties encountered

    message : str
        A string descriptor of the exit status of the optimization.
    iteration : int
        The number of iterations taken to solve the problem.

    References
    ----------
    .. [1] Dantzig, George B., Linear programming and extensions. Rand
           Corporation Research Study Princeton Univ. Press, Princeton, NJ,
           1963
    .. [2] Hillier, S.H. and Lieberman, G.J. (1995), "Introduction to
           Mathematical Programming", McGraw-Hill, Chapter 4.
    .. [3] Bland, Robert G. New finite pivoting rules for the simplex method.
           Mathematics of Operations Research (2), 1977: pp. 103-107.


    Notes
    -----
    The expected problem formulation differs between the top level ``linprog``
    module and the method specific solvers. The method specific solvers expect a
    problem in standard form:

    Minimize::

        c @ x

    Subject to::

        A @ x == b
            x >= 0

    Whereas the top level ``linprog`` module expects a problem of form:

    Minimize::

        c @ x

    Subject to::

        A_ub @ x <= b_ub
        A_eq @ x == b_eq
         lb <= x <= ub

    where ``lb = 0`` and ``ub = None`` unless set in ``bounds``.

    The original problem contains equality, upper-bound and variable constraints
    whereas the method specific solver requires equality constraints and
    variable non-negativity.

    ``linprog`` module converts the original problem to standard form by
    converting the simple bounds to upper bound constraints, introducing
    non-negative slack variables for inequality constraints, and expressing
    unbounded variables as the difference between two non-negative variables.
    """
    _check_unknown_options(unknown_options)

    status = 0
    messages = {0: "Optimization terminated successfully.",
                1: "Iteration limit reached.",
                2: "Optimization failed. Unable to find a feasible"
                   " starting point.",
                3: "Optimization failed. The problem appears to be unbounded.",
                4: "Optimization failed. Singular matrix encountered."}

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

    nit1, status = _solve_simplex(T, n, basis, callback=callback,
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
        nit2, status = _solve_simplex(T, n, basis, callback=callback,
                                      postsolve_args=postsolve_args,
                                      maxiter=maxiter, tol=tol, phase=2,
                                      bland=bland, nit0=nit1
                                      )

    solution = np.zeros(n + m)
    solution[basis[:n]] = T[:n, -1]
    x = solution[:m]

    return x, status, messages[status], int(nit2)



def _solve_simplex(T, n, basis, callback, postsolve_args,
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
                _apply_pivot(T, basis, pivrow, pivcol, tol)
                nit += 1

    if len(basis[:m]) == 0:
        solution = np.empty(T.shape[1] - 1, dtype=np.float64)
    else:
        solution = np.empty(max(T.shape[1] - 1, max(basis[:m]) + 1),
                            dtype=np.float64)

    while not complete:
        # Find the pivot column
        pivcol_found, pivcol = _pivot_col(T, tol, bland)
        if not pivcol_found:
            pivcol = np.nan
            pivrow = np.nan
            status = 0
            complete = True
        else:
            # Find the pivot row
            pivrow_found, pivrow = _pivot_row(T, basis, pivcol, phase, tol, bland)
            if not pivrow_found:
                status = 3
                complete = True

        if callback is not None:
            solution[:] = 0
            solution[basis[:n]] = T[:n, -1]
            x = solution[:m]
            x, fun, slack, con = _postsolve(
                x, postsolve_args
            )
            res = OptimizeResult({
                'x': x,
                'fun': fun,
                'slack': slack,
                'con': con,
                'status': status,
                'message': message,
                'nit': nit,
                'success': status == 0 and complete,
                'phase': phase,
                'complete': complete,
                })
            callback(res)

        if not complete:
            if nit >= maxiter:
                # Iteration limit exceeded
                status = 1
                complete = True
            else:
                _apply_pivot(T, basis, pivrow, pivcol, tol)
                nit += 1
    return nit, status


def _pivot_col(T, tol=1e-9, bland=False):

    """
    Given a linear programming simplex tableau, determine the column
    of the variable to enter the basis.

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
    tol : float
        Elements in the objective row larger than -tol will not be considered
        for pivoting. Nominally this value is zero, but numerical issues
        cause a tolerance about zero to be necessary.
    bland : bool
        If True, use Bland's rule for selection of the column (select the
        first column with a negative coefficient in the objective row,
        regardless of magnitude).

    Returns
    -------
    status: bool
        True if a suitable pivot column was found, otherwise False.
        A return of False indicates that the linear programming simplex
        algorithm is complete.
    col: int
        The index of the column of the pivot element.
        If status is False, col will be returned as nan.
    """
    ma = np.ma.masked_where(T[-1, :-1] >= -tol, T[-1, :-1], copy=False)
    if ma.count() == 0:
        return False, np.nan
    if bland:
        # ma.mask is sometimes 0d
        return True, np.nonzero(np.logical_not(np.atleast_1d(ma.mask)))[0][0]
    return True, np.ma.nonzero(ma == ma.min())[0][0]


def _pivot_row(T, basis, pivcol, phase, tol=1e-9, bland=False):
    """
    Given a linear programming simplex tableau, determine the row for the
    pivot operation.

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

         for a Phase 1 problem (a Problem in which a basic feasible solution is
         sought prior to maximizing the actual objective. ``T`` is modified in
         place by ``_solve_simplex``.
    basis : array
        A list of the current basic variables.
    pivcol : int
        The index of the pivot column.
    phase : int
        The phase of the simplex algorithm (1 or 2).
    tol : float
        Elements in the pivot column smaller than tol will not be considered
        for pivoting. Nominally this value is zero, but numerical issues
        cause a tolerance about zero to be necessary.
    bland : bool
        If True, use Bland's rule for selection of the row (if more than one
        row can be used, choose the one with the lowest variable index).

    Returns
    -------
    status: bool
        True if a suitable pivot row was found, otherwise False. A return
        of False indicates that the linear programming problem is unbounded.
    row: int
        The index of the row of the pivot element. If status is False, row
        will be returned as nan.
    """
    if phase == 1:
        k = 2
    else:
        k = 1
    ma = np.ma.masked_where(T[:-k, pivcol] <= tol, T[:-k, pivcol], copy=False)
    if ma.count() == 0:
        return False, np.nan
    mb = np.ma.masked_where(T[:-k, pivcol] <= tol, T[:-k, -1], copy=False)
    q = mb / ma
    min_rows = np.ma.nonzero(q == q.min())[0]
    if bland:
        return True, min_rows[np.argmin(np.take(basis, min_rows))]
    return True, min_rows[0]


def _apply_pivot(T, basis, pivrow, pivcol, tol=1e-9):
    """
    Pivot the simplex tableau inplace on the element given by (pivrow, pivol).
    The entering variable corresponds to the column given by pivcol forcing
    the variable basis[pivrow] to leave the basis.

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
    basis : 1-D array
        An array of the indices of the basic variables, such that basis[i]
        contains the column corresponding to the basic variable for row i.
        Basis is modified in place by _apply_pivot.
    pivrow : int
        Row index of the pivot.
    pivcol : int
        Column index of the pivot.
    """
    basis[pivrow] = pivcol
    pivval = T[pivrow, pivcol]
    T[pivrow] = T[pivrow] / pivval
    for irow in range(T.shape[0]):
        if irow != pivrow:
            T[irow] = T[irow] - T[pivrow] * T[irow, pivcol]

    # The selected pivot should never lead to a pivot value less than the tol.
    if np.isclose(pivval, tol, atol=0, rtol=1e4):
        message = (
            "The pivot operation produces a pivot value of:{0: .1e}, "
            "which is only slightly greater than the specified "
            "tolerance{1: .1e}. This may lead to issues regarding the "
            "numerical stability of the simplex method. "
            "Removing redundant constraints, changing the pivot strategy "
            "via Bland's rule or increasing the tolerance may "
            "help reduce the issue.".format(pivval, tol))
        warn(message, OptimizeWarning, stacklevel=5)


def _presolve(lp, rr, rr_method, tol=1e-9):
    """
    Given inputs for a linear programming problem in preferred format,
    presolve the problem: identify trivial infeasibilities, redundancies,
    and unboundedness, tighten bounds where possible, and eliminate fixed
    variables.

    Parameters
    ----------
    lp : A `scipy.optimize._linprog_util._LPProblem` consisting of the following fields:

        c : 1D array
            The coefficients of the linear objective function to be minimized.
        A_ub : 2D array, optional
            The inequality constraint matrix. Each row of ``A_ub`` specifies the
            coefficients of a linear inequality constraint on ``x``.
        b_ub : 1D array, optional
            The inequality constraint vector. Each element represents an
            upper bound on the corresponding value of ``A_ub @ x``.
        A_eq : 2D array, optional
            The equality constraint matrix. Each row of ``A_eq`` specifies the
            coefficients of a linear equality constraint on ``x``.
        b_eq : 1D array, optional
            The equality constraint vector. Each element of ``A_eq @ x`` must equal
            the corresponding element of ``b_eq``.
        bounds : 2D array
            The bounds of ``x``, as ``min`` and ``max`` pairs, one for each of the N
            elements of ``x``. The N x 2 array contains lower bounds in the first
            column and upper bounds in the 2nd. Unbounded variables have lower
            bound -np.inf and/or upper bound np.inf.
        x0 : 1D array, optional
            Guess values of the decision variables, which will be refined by
            the optimization algorithm. This argument is currently used only by the
            'revised simplex' method, and can only be used if `x0` represents a
            basic feasible solution.

    rr : bool
        If ``True`` attempts to eliminate any redundant rows in ``A_eq``.
        Set False if ``A_eq`` is known to be of full row rank, or if you are
        looking for a potential speedup (at the expense of reliability).
    rr_method : string
        Method used to identify and remove redundant rows from the
        equality constraint matrix after presolve.
    tol : float
        The tolerance which determines when a solution is "close enough" to
        zero in Phase 1 to be considered a basic feasible solution or close
        enough to positive to serve as an optimal solution.

    Returns
    -------
    lp : A `scipy.optimize._linprog_util._LPProblem` consisting of the following fields:

        c : 1D array
            The coefficients of the linear objective function to be minimized.
        A_ub : 2D array, optional
            The inequality constraint matrix. Each row of ``A_ub`` specifies the
            coefficients of a linear inequality constraint on ``x``.
        b_ub : 1D array, optional
            The inequality constraint vector. Each element represents an
            upper bound on the corresponding value of ``A_ub @ x``.
        A_eq : 2D array, optional
            The equality constraint matrix. Each row of ``A_eq`` specifies the
            coefficients of a linear equality constraint on ``x``.
        b_eq : 1D array, optional
            The equality constraint vector. Each element of ``A_eq @ x`` must equal
            the corresponding element of ``b_eq``.
        bounds : 2D array
            The bounds of ``x``, as ``min`` and ``max`` pairs, possibly tightened.
        x0 : 1D array, optional
            Guess values of the decision variables, which will be refined by
            the optimization algorithm. This argument is currently used only by the
            'revised simplex' method, and can only be used if `x0` represents a
            basic feasible solution.

    c0 : 1D array
        Constant term in objective function due to fixed (and eliminated)
        variables.
    x : 1D array
        Solution vector (when the solution is trivial and can be determined
        in presolve)
    revstack: list of functions
        the functions in the list reverse the operations of _presolve()
        the function signature is x_org = f(x_mod), where x_mod is the result
        of a presolve step and x_org the value at the start of the step
        (currently, the revstack contains only one function)
    complete: bool
        Whether the solution is complete (solved or determined to be infeasible
        or unbounded in presolve)
    status : int
        An integer representing the exit status of the optimization::

         0 : Optimization terminated successfully
         1 : Iteration limit reached
         2 : Problem appears to be infeasible
         3 : Problem appears to be unbounded
         4 : Serious numerical difficulties encountered

    message : str
        A string descriptor of the exit status of the optimization.

    References
    ----------
    .. [5] Andersen, Erling D. "Finding all linearly dependent rows in
           large-scale linear programming." Optimization Methods and Software
           6.3 (1995): 219-227.
    .. [8] Andersen, Erling D., and Knud D. Andersen. "Presolving in linear
           programming." Mathematical Programming 71.2 (1995): 221-245.

    """
    # ideas from Reference [5] by Andersen and Andersen
    # however, unlike the reference, this is performed before converting
    # problem to standard form
    # There are a few advantages:
    #  * artificial variables have not been added, so matrices are smaller
    #  * bounds have not been converted to constraints yet. (It is better to
    #    do that after presolve because presolve may adjust the simple bounds.)
    # There are many improvements that can be made, namely:
    #  * implement remaining checks from [5]
    #  * loop presolve until no additional changes are made
    #  * implement additional efficiency improvements in redundancy removal [2]

    c, A_ub, b_ub, A_eq, b_eq, bounds, x0 = lp

    revstack = []               # record of variables eliminated from problem
    # constant term in cost function may be added if variables are eliminated
    c0 = 0
    complete = False        # complete is True if detected infeasible/unbounded
    x = np.zeros(c.shape)   # this is solution vector if completed in presolve

    status = 0              # all OK unless determined otherwise
    message = ""

    # Lower and upper bounds. Copy to prevent feedback.
    lb = bounds[:, 0].copy()
    ub = bounds[:, 1].copy()

    m_eq, n = A_eq.shape
    m_ub, n = A_ub.shape

    if (rr_method is not None
            and rr_method.lower() not in {"svd", "pivot", "id"}):
        message = ("'" + str(rr_method) + "' is not a valid option "
                   "for redundancy removal. Valid options are 'SVD', "
                   "'pivot', and 'ID'.")
        raise ValueError(message)

    if sps.issparse(A_eq):
        A_eq = A_eq.tocsr()
        A_ub = A_ub.tocsr()

        def where(A):
            return A.nonzero()

        vstack = sps.vstack
    else:
        where = np.where
        vstack = np.vstack

    # upper bounds > lower bounds
    if np.any(ub < lb) or np.any(lb == np.inf) or np.any(ub == -np.inf):
        status = 2
        message = ("The problem is (trivially) infeasible since one "
                   "or more upper bounds are smaller than the corresponding "
                   "lower bounds, a lower bound is np.inf or an upper bound "
                   "is -np.inf.")
        complete = True
        return (_LPProblem(c, A_ub, b_ub, A_eq, b_eq, bounds, x0),
                c0, x, revstack, complete, status, message)

    # zero row in equality constraints
    zero_row = np.array(np.sum(A_eq != 0, axis=1) == 0).flatten()
    if np.any(zero_row):
        if np.any(
            np.logical_and(
                zero_row,
                np.abs(b_eq) > tol)):  # test_zero_row_1
            # infeasible if RHS is not zero
            status = 2
            message = ("The problem is (trivially) infeasible due to a row "
                       "of zeros in the equality constraint matrix with a "
                       "nonzero corresponding constraint value.")
            complete = True
            return (_LPProblem(c, A_ub, b_ub, A_eq, b_eq, bounds, x0),
                    c0, x, revstack, complete, status, message)
        else:  # test_zero_row_2
            # if RHS is zero, we can eliminate this equation entirely
            A_eq = A_eq[np.logical_not(zero_row), :]
            b_eq = b_eq[np.logical_not(zero_row)]

    # zero row in inequality constraints
    zero_row = np.array(np.sum(A_ub != 0, axis=1) == 0).flatten()
    if np.any(zero_row):
        if np.any(np.logical_and(zero_row, b_ub < -tol)):  # test_zero_row_1
            # infeasible if RHS is less than zero (because LHS is zero)
            status = 2
            message = ("The problem is (trivially) infeasible due to a row "
                       "of zeros in the equality constraint matrix with a "
                       "nonzero corresponding  constraint value.")
            complete = True
            return (_LPProblem(c, A_ub, b_ub, A_eq, b_eq, bounds, x0),
                    c0, x, revstack, complete, status, message)
        else:  # test_zero_row_2
            # if LHS is >= 0, we can eliminate this constraint entirely
            A_ub = A_ub[np.logical_not(zero_row), :]
            b_ub = b_ub[np.logical_not(zero_row)]

    # zero column in (both) constraints
    # this indicates that a variable isn't constrained and can be removed
    A = vstack((A_eq, A_ub))
    if A.shape[0] > 0:
        zero_col = np.array(np.sum(A != 0, axis=0) == 0).flatten()
        # variable will be at upper or lower bound, depending on objective
        x[np.logical_and(zero_col, c < 0)] = ub[
            np.logical_and(zero_col, c < 0)]
        x[np.logical_and(zero_col, c > 0)] = lb[
            np.logical_and(zero_col, c > 0)]
        if np.any(np.isinf(x)):  # if an unconstrained variable has no bound
            status = 3
            message = ("If feasible, the problem is (trivially) unbounded "
                       "due  to a zero column in the constraint matrices. If "
                       "you wish to check whether the problem is infeasible, "
                       "turn presolve off.")
            complete = True
            return (_LPProblem(c, A_ub, b_ub, A_eq, b_eq, bounds, x0),
                    c0, x, revstack, complete, status, message)
        # variables will equal upper/lower bounds will be removed later
        lb[np.logical_and(zero_col, c < 0)] = ub[
            np.logical_and(zero_col, c < 0)]
        ub[np.logical_and(zero_col, c > 0)] = lb[
            np.logical_and(zero_col, c > 0)]

    # row singleton in equality constraints
    # this fixes a variable and removes the constraint
    singleton_row = np.array(np.sum(A_eq != 0, axis=1) == 1).flatten()
    rows = where(singleton_row)[0]
    cols = where(A_eq[rows, :])[1]
    if len(rows) > 0:
        for row, col in zip(rows, cols):
            val = b_eq[row] / A_eq[row, col]
            if not lb[col] - tol <= val <= ub[col] + tol:
                # infeasible if fixed value is not within bounds
                status = 2
                message = ("The problem is (trivially) infeasible because a "
                           "singleton row in the equality constraints is "
                           "inconsistent with the bounds.")
                complete = True
                return (_LPProblem(c, A_ub, b_ub, A_eq, b_eq, bounds, x0),
                        c0, x, revstack, complete, status, message)
            else:
                # sets upper and lower bounds at that fixed value - variable
                # will be removed later
                lb[col] = val
                ub[col] = val
        A_eq = A_eq[np.logical_not(singleton_row), :]
        b_eq = b_eq[np.logical_not(singleton_row)]

    # row singleton in inequality constraints
    # this indicates a simple bound and the constraint can be removed
    # simple bounds may be adjusted here
    # After all of the simple bound information is combined here, get_Abc will
    # turn the simple bounds into constraints
    singleton_row = np.array(np.sum(A_ub != 0, axis=1) == 1).flatten()
    cols = where(A_ub[singleton_row, :])[1]
    rows = where(singleton_row)[0]
    if len(rows) > 0:
        for row, col in zip(rows, cols):
            val = b_ub[row] / A_ub[row, col]
            if A_ub[row, col] > 0:  # upper bound
                if val < lb[col] - tol:  # infeasible
                    complete = True
                elif val < ub[col]:  # new upper bound
                    ub[col] = val
            else:  # lower bound
                if val > ub[col] + tol:  # infeasible
                    complete = True
                elif val > lb[col]:  # new lower bound
                    lb[col] = val
            if complete:
                status = 2
                message = ("The problem is (trivially) infeasible because a "
                           "singleton row in the upper bound constraints is "
                           "inconsistent with the bounds.")
                return (_LPProblem(c, A_ub, b_ub, A_eq, b_eq, bounds, x0),
                        c0, x, revstack, complete, status, message)
        A_ub = A_ub[np.logical_not(singleton_row), :]
        b_ub = b_ub[np.logical_not(singleton_row)]

    # identical bounds indicate that variable can be removed
    i_f = np.abs(lb - ub) < tol   # indices of "fixed" variables
    i_nf = np.logical_not(i_f)  # indices of "not fixed" variables

    # test_bounds_equal_but_infeasible
    if np.all(i_f):  # if bounds define solution, check for consistency
        residual = b_eq - A_eq.dot(lb)
        slack = b_ub - A_ub.dot(lb)
        if ((A_ub.size > 0 and np.any(slack < 0)) or
                (A_eq.size > 0 and not np.allclose(residual, 0))):
            status = 2
            message = ("The problem is (trivially) infeasible because the "
                       "bounds fix all variables to values inconsistent with "
                       "the constraints")
            complete = True
            return (_LPProblem(c, A_ub, b_ub, A_eq, b_eq, bounds, x0),
                    c0, x, revstack, complete, status, message)

    ub_mod = ub
    lb_mod = lb
    if np.any(i_f):
        c0 += c[i_f].dot(lb[i_f])
        b_eq = b_eq - A_eq[:, i_f].dot(lb[i_f])
        b_ub = b_ub - A_ub[:, i_f].dot(lb[i_f])
        c = c[i_nf]
        x_undo = lb[i_f]  # not x[i_f], x is just zeroes
        x = x[i_nf]
        # user guess x0 stays separate from presolve solution x
        if x0 is not None:
            x0 = x0[i_nf]
        A_eq = A_eq[:, i_nf]
        A_ub = A_ub[:, i_nf]
        # modify bounds
        lb_mod = lb[i_nf]
        ub_mod = ub[i_nf]

        def rev(x_mod):
            # Function to restore x: insert x_undo into x_mod.
            # When elements have been removed at positions k1, k2, k3, ...
            # then these must be replaced at (after) positions k1-1, k2-2,
            # k3-3, ... in the modified array to recreate the original
            i = np.flatnonzero(i_f)
            # Number of variables to restore
            N = len(i)
            index_offset = np.arange(N)
            # Create insert indices
            insert_indices = i - index_offset
            x_rev = np.insert(x_mod.astype(float), insert_indices, x_undo)
            return x_rev

        # Use revstack as a list of functions, currently just this one.
        revstack.append(rev)

    # no constraints indicates that problem is trivial
    if A_eq.size == 0 and A_ub.size == 0:
        b_eq = np.array([])
        b_ub = np.array([])
        # test_empty_constraint_1
        if c.size == 0:
            status = 0
            message = ("The solution was determined in presolve as there are "
                       "no non-trivial constraints.")
        elif (np.any(np.logical_and(c < 0, ub_mod == np.inf)) or
              np.any(np.logical_and(c > 0, lb_mod == -np.inf))):
            # test_no_constraints()
            # test_unbounded_no_nontrivial_constraints_1
            # test_unbounded_no_nontrivial_constraints_2
            status = 3
            message = ("The problem is (trivially) unbounded "
                       "because there are no non-trivial constraints and "
                       "a) at least one decision variable is unbounded "
                       "above and its corresponding cost is negative, or "
                       "b) at least one decision variable is unbounded below "
                       "and its corresponding cost is positive. ")
        else:  # test_empty_constraint_2
            status = 0
            message = ("The solution was determined in presolve as there are "
                       "no non-trivial constraints.")
        complete = True
        x[c < 0] = ub_mod[c < 0]
        x[c > 0] = lb_mod[c > 0]
        # where c is zero, set x to a finite bound or zero
        x_zero_c = ub_mod[c == 0]
        x_zero_c[np.isinf(x_zero_c)] = ub_mod[c == 0][np.isinf(x_zero_c)]
        x_zero_c[np.isinf(x_zero_c)] = 0
        x[c == 0] = x_zero_c
        # if this is not the last step of presolve, should convert bounds back
        # to array and return here

    # Convert modified lb and ub back into N x 2 bounds
    bounds = np.hstack((lb_mod[:, np.newaxis], ub_mod[:, np.newaxis]))

    # remove redundant (linearly dependent) rows from equality constraints
    n_rows_A = A_eq.shape[0]
    redundancy_warning = ("A_eq does not appear to be of full row rank. To "
                          "improve performance, check the problem formulation "
                          "for redundant equality constraints.")
    if (sps.issparse(A_eq)):
        if rr and A_eq.size > 0:  # TODO: Fast sparse rank check?
            rr_res = _remove_redundancy_pivot_sparse(A_eq, b_eq)
            A_eq, b_eq, status, message = rr_res
            if A_eq.shape[0] < n_rows_A:
                warn(redundancy_warning, OptimizeWarning, stacklevel=1)
            if status != 0:
                complete = True
        return (_LPProblem(c, A_ub, b_ub, A_eq, b_eq, bounds, x0),
                c0, x, revstack, complete, status, message)

    # This is a wild guess for which redundancy removal algorithm will be
    # faster. More testing would be good.
    small_nullspace = 5
    if rr and A_eq.size > 0:
        try:  # TODO: use results of first SVD in _remove_redundancy_svd
            rank = np.linalg.matrix_rank(A_eq)
        # oh well, we'll have to go with _remove_redundancy_pivot_dense
        except Exception:
            rank = 0
    if rr and A_eq.size > 0 and rank < A_eq.shape[0]:
        warn(redundancy_warning, OptimizeWarning, stacklevel=3)
        dim_row_nullspace = A_eq.shape[0]-rank
        if rr_method is None:
            if dim_row_nullspace <= small_nullspace:
                rr_res = _remove_redundancy_svd(A_eq, b_eq)
                A_eq, b_eq, status, message = rr_res
            if dim_row_nullspace > small_nullspace or status == 4:
                rr_res = _remove_redundancy_pivot_dense(A_eq, b_eq)
                A_eq, b_eq, status, message = rr_res

        else:
            rr_method = rr_method.lower()
            if rr_method == "svd":
                rr_res = _remove_redundancy_svd(A_eq, b_eq)
                A_eq, b_eq, status, message = rr_res
            elif rr_method == "pivot":
                rr_res = _remove_redundancy_pivot_dense(A_eq, b_eq)
                A_eq, b_eq, status, message = rr_res
            elif rr_method == "id":
                rr_res = _remove_redundancy_id(A_eq, b_eq, rank)
                A_eq, b_eq, status, message = rr_res
            else:  # shouldn't get here; option validity checked above
                pass
        if A_eq.shape[0] < rank:
            message = ("Due to numerical issues, redundant equality "
                       "constraints could not be removed automatically. "
                       "Try providing your constraint matrices as sparse "
                       "matrices to activate sparse presolve, try turning "
                       "off redundancy removal, or try turning off presolve "
                       "altogether.")
            status = 4
        if status != 0:
            complete = True
    return (_LPProblem(c, A_ub, b_ub, A_eq, b_eq, bounds, x0),
            c0, x, revstack, complete, status, message)

