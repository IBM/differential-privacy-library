import numbers
import warnings

import numpy as np
from scipy import optimize
from sklearn.exceptions import ConvergenceWarning
from sklearn import linear_model
from sklearn.externals.joblib import delayed, Parallel
from sklearn.linear_model.logistic import _logistic_loss_and_grad
from sklearn.utils import check_X_y, check_array, check_consistent_length
from sklearn.utils.fixes import _joblib_parallel_args
from sklearn.utils.multiclass import check_classification_targets

from diffprivlib.mechanisms import Vector
from diffprivlib.utils import PrivacyLeakWarning, DiffprivlibCompatibilityWarning


class LogisticRegression(linear_model.LogisticRegression):
    def __init__(self, epsilon=1.0, data_norm=1.0, penalty='l2', dual=False, tol=1e-4, C=1.0, fit_intercept=True,
                 intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=100,
                 multi_class='ovr', verbose=0, warm_start=False, n_jobs=None):
        super().__init__(penalty=penalty, dual=dual, tol=tol, C=C, fit_intercept=fit_intercept,
                         intercept_scaling=intercept_scaling, class_weight=class_weight, random_state=random_state,
                         solver=solver, max_iter=max_iter, multi_class=multi_class, verbose=verbose,
                         warm_start=warm_start, n_jobs=n_jobs)
        self.epsilon = epsilon
        self.data_norm = data_norm
        self.classes_ = None

    # noinspection PyAttributeOutsideInit
    def fit(self, X, y, sample_weight=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples,)
            Target vector relative to X.

        sample_weight : array-like, shape (n_samples,) optional
            Array of weights that are assigned to individual samples.
            If not provided, then each sample is given unit weight.

            .. versionadded:: 0.17
               *sample_weight* support to LogisticRegression.

        Returns
        -------
        self : object
        """
        if not isinstance(self.C, numbers.Real) or self.C < 0:
            raise ValueError("Penalty term must be positive; got (C=%r)" % self.C)
        if not isinstance(self.max_iter, numbers.Integral) or self.max_iter < 0:
            raise ValueError("Maximum number of iteration must be positive; got (max_iter=%r)" % self.max_iter)
        if not isinstance(self.tol, numbers.Real) or self.tol < 0:
            raise ValueError("Tolerance for stopping criteria must be positive; got (tol=%r)" % self.tol)

        max_norm = np.linalg.norm(X, axis=1).max()
        if max_norm > self.data_norm:
            warnings.warn("Differential privacy is only guaranteed for data whose rows have a 2-norm of at most %g. "
                          "Got %f\n"
                          "Translate and/or scale the data accordingly to ensure differential privacy is achieved."
                          % (self.data_norm, max_norm), PrivacyLeakWarning)

        solver = _check_solver(self.solver, self.penalty, self.dual)
        # todo: may be able to remove when docstring for __init__ is written
        self.max_iter = int(self.max_iter)
        self.tol = float(self.tol)

        _dtype = np.float64

        X, y = check_X_y(X, y, accept_sparse='csr', dtype=_dtype, order="C", accept_large_sparse=solver != 'liblinear')
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape

        multi_class = _check_multi_class(self.multi_class, solver, len(self.classes_))

        n_classes = len(self.classes_)
        classes_ = self.classes_
        if n_classes < 2:
            raise ValueError("This solver needs samples of at least 2 classes in the data, but the data contains only "
                             "one class: %r" % classes_[0])

        if len(self.classes_) == 2:
            n_classes = 1
            classes_ = classes_[1:]

        if self.warm_start:
            warm_start_coef = getattr(self, 'coef_', None)
        else:
            warm_start_coef = None
        if warm_start_coef is not None and self.fit_intercept:
            warm_start_coef = np.append(warm_start_coef, self.intercept_[:, np.newaxis], axis=1)

        self.coef_ = list()
        self.intercept_ = np.zeros(n_classes)

        if warm_start_coef is None:
            warm_start_coef = [None] * n_classes

        path_func = delayed(logistic_regression_path)

        fold_coefs_ = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                               **_joblib_parallel_args(prefer='processes'))(
            path_func(X, y, epsilon=self.epsilon / n_classes, data_norm=self.data_norm, pos_class=class_, Cs=[self.C],
                      fit_intercept=self.fit_intercept, tol=self.tol, verbose=self.verbose, solver=solver,
                      multi_class=multi_class, max_iter=self.max_iter, class_weight=self.class_weight,
                      check_input=False, random_state=self.random_state, coef=warm_start_coef_, penalty=self.penalty,
                      max_squared_sum=None, sample_weight=sample_weight)
            for class_, warm_start_coef_ in zip(classes_, warm_start_coef))

        fold_coefs_, _, n_iter_ = zip(*fold_coefs_)
        self.n_iter_ = np.asarray(n_iter_, dtype=np.int32)[:, 0]

        self.coef_ = np.asarray(fold_coefs_)
        self.coef_ = self.coef_.reshape(n_classes, n_features + int(self.fit_intercept))

        if self.fit_intercept:
            self.intercept_ = self.coef_[:, -1]
            self.coef_ = self.coef_[:, :-1]

        return self


def logistic_regression_path(X, y, epsilon=1.0, data_norm=1.0, pos_class=None, Cs=10, fit_intercept=True, max_iter=100,
                             tol=1e-4, verbose=0, solver='lbfgs', coef=None, class_weight=None, dual=False,
                             penalty='l2', intercept_scaling=1., multi_class='ovr', random_state=None,
                             check_input=True, max_squared_sum=None, sample_weight=None):
    """Compute a Logistic Regression model for a list of regularization
    parameters.

    This is an implementation that uses the result of the previous model
    to speed up computations along the set of solutions, making it faster
    than sequentially calling LogisticRegression for the different parameters.
    Note that there will be no speedup with liblinear solver, since it does
    not handle warm-starting.

    Read more in the :ref:`User Guide <logistic_regression>`.

    Parameters
    ----------
    X : array-like or sparse matrix, shape (n_samples, n_features)
        Input data.

    y : array-like, shape (n_samples,) or (n_samples, n_targets)
        Input data, target values.

    epsilon : float
        Privacy parameter for differential privacy.

    data_norm : float
        Max norm of the data for which differential privacy is satisfied.

    pos_class : int, None
        The class with respect to which we perform a one-vs-all fit.
        If None, then it is assumed that the given problem is binary.

    Cs : int | array-like, shape (n_cs,)
        List of values for the regularization parameter or integer specifying
        the number of regularization parameters that should be used. In this
        case, the parameters will be chosen in a logarithmic scale between
        1e-4 and 1e4.

    fit_intercept : bool
        Whether to fit an intercept for the model. In this case the shape of
        the returned array is (n_cs, n_features + 1).

    max_iter : int
        Maximum number of iterations for the solver.

    tol : float
        Stopping criterion. For the newton-cg and lbfgs solvers, the iteration
        will stop when ``max{|g_i | i = 1, ..., n} <= tol``
        where ``g_i`` is the i-th component of the gradient.

    verbose : int
        For the liblinear and lbfgs solvers set verbose to any positive
        number for verbosity.

    solver : {'lbfgs', 'newton-cg', 'liblinear', 'sag', 'saga'}
        Numerical solver to use.

    coef : array-like, shape (n_features,), default None
        Initialization value for coefficients of logistic regression.
        Useless for liblinear solver.

    class_weight : dict or 'balanced', optional
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one.

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``.

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

    dual : bool
        Dual or primal formulation. Dual formulation is only implemented for
        l2 penalty with liblinear solver. Prefer dual=False when
        n_samples > n_features.

    penalty : str, 'l1' or 'l2'
        Used to specify the norm used in the penalization. The 'newton-cg',
        'sag' and 'lbfgs' solvers support only l2 penalties.

    intercept_scaling : float, default 1.
        Useful only when the solver 'liblinear' is used
        and self.fit_intercept is set to True. In this case, x becomes
        [x, self.intercept_scaling],
        i.e. a "synthetic" feature with constant value equal to
        intercept_scaling is appended to the instance vector.
        The intercept becomes ``intercept_scaling * synthetic_feature_weight``.

        Note! the synthetic feature weight is subject to l1/l2 regularization
        as all other features.
        To lessen the effect of regularization on synthetic feature weight
        (and therefore on the intercept) intercept_scaling has to be increased.

    multi_class : str, {'ovr', 'multinomial', 'auto'}, default: 'ovr'
        If the option chosen is 'ovr', then a binary problem is fit for each
        label. For 'multinomial' the loss minimised is the multinomial loss fit
        across the entire probability distribution, *even when the data is
        binary*. 'multinomial' is unavailable when solver='liblinear'.
        'auto' selects 'ovr' if the data is binary, or if solver='liblinear',
        and otherwise selects 'multinomial'.

        .. versionadded:: 0.18
           Stochastic Average Gradient descent solver for 'multinomial' case.
        .. versionchanged:: 0.20
            Default will change from 'ovr' to 'auto' in 0.22.

    random_state : int, RandomState instance or None, optional, default None
        The seed of the pseudo random number generator to use when shuffling
        the data.  If int, random_state is the seed used by the random number
        generator; If RandomState instance, random_state is the random number
        generator; If None, the random number generator is the RandomState
        instance used by `np.random`. Used when ``solver`` == 'sag' or
        'liblinear'.

    check_input : bool, default True
        If False, the input arrays X and y will not be checked.

    max_squared_sum : float, default None
        Maximum squared sum of X over samples. Used only in SAG solver.
        If None, it will be computed, going through all the samples.
        The value should be precomputed to speed up cross validation.

    sample_weight : array-like, shape(n_samples,) optional
        Array of weights that are assigned to individual samples.
        If not provided, then each sample is given unit weight.

    Returns
    -------
    coefs : ndarray, shape (n_cs, n_features) or (n_cs, n_features + 1)
        List of coefficients for the Logistic Regression model. If
        fit_intercept is set to True then the second dimension will be
        n_features + 1, where the last item represents the intercept. For
        ``multiclass='multinomial'``, the shape is (n_classes, n_cs,
        n_features) or (n_classes, n_cs, n_features + 1).

    Cs : ndarray
        Grid of Cs used for cross-validation.

    n_iter : array, shape (n_cs,)
        Actual number of iteration for each Cs.

    Notes
    -----
    You might get slightly different results with the solver liblinear than
    with the others since this uses LIBLINEAR which penalizes the intercept.

    .. versionchanged:: 0.19
        The "copy" parameter was removed.
    """
    if class_weight is not None:
        warnings.warn("For diffprivlib, class_weight is not used. Set to None to suppress this warning.",
                      DiffprivlibCompatibilityWarning)
        del class_weight
    if sample_weight is not None:
        warnings.warn("For diffprivlib, sample_weight is not used. Set to None to suppress this warning.",
                      DiffprivlibCompatibilityWarning)
        del sample_weight
    if intercept_scaling != 1.:
        warnings.warn("For diffprivlib, intercept_scaling is not used. Set to 1.0 to suppress this warning.",
                      DiffprivlibCompatibilityWarning)
        del intercept_scaling
    if max_squared_sum is not None:
        warnings.warn("For diffprivlib, max_squared_sum is not used. Set to None to suppress this warning.",
                      DiffprivlibCompatibilityWarning)
        del max_squared_sum

    if isinstance(Cs, numbers.Integral):
        Cs = np.logspace(-4, 4, int(Cs))

    solver = _check_solver(solver, penalty, dual)

    # Data norm increases if intercept is included
    if fit_intercept:
        data_norm = np.sqrt(data_norm ** 2 + 1)

    # Pre-processing.
    if check_input:
        X = check_array(X, accept_sparse='csr', dtype=np.float64, accept_large_sparse=solver != 'liblinear')
        y = check_array(y, ensure_2d=False, dtype=None)
        check_consistent_length(X, y)
    n_samples, n_features = X.shape

    classes = np.unique(y)

    multi_class = _check_multi_class(multi_class, solver, len(classes))
    if pos_class is None and multi_class != 'multinomial':
        if classes.size > 2:
            raise ValueError('To fit OvR, use the pos_class argument')
        # np.unique(y) gives labels in sorted order.
        pos_class = classes[1]

    sample_weight = np.ones(X.shape[0], dtype=X.dtype)

    # For doing a ovr, we need to mask the labels first.
    w0 = np.zeros(n_features + int(fit_intercept), dtype=X.dtype)
    mask = (y == pos_class)
    y_bin = np.ones(y.shape, dtype=X.dtype)
    y_bin[~mask] = -1.
    # for compute_class_weight

    if coef is not None:
        # it must work both giving the bias term and not
        if coef.size not in (n_features, w0.size):
            raise ValueError('Initialization coef is of shape %d, expected shape %d or %d' % (coef.size, n_features,
                                                                                              w0.size))
        w0[:coef.size] = coef

    target = y_bin

    coefs = list()
    n_iter = np.zeros(len(Cs), dtype=np.int32)
    for i, C in enumerate(Cs):
        vector_mech = Vector()\
            .set_dimension(n_features + int(fit_intercept))\
            .set_epsilon(epsilon)\
            .set_alpha(1. / C)\
            .set_sensitivity(0.25, data_norm)
        noisy_logistic_loss = vector_mech.randomise(_logistic_loss_and_grad)

        iprint = [-1, 50, 1, 100, 101][np.searchsorted(np.array([0, 1, 2, 3]), verbose)]
        w0, loss, info = optimize.fmin_l_bfgs_b(noisy_logistic_loss, w0, fprime=None,
                                                args=(X, target, 1. / C, sample_weight), iprint=iprint, pgtol=tol,
                                                maxiter=max_iter)
        if info["warnflag"] == 1:
            warnings.warn("lbfgs failed to converge. Increase the number of iterations.", ConvergenceWarning)

        coefs.append(w0.copy())

        n_iter[i] = info['nit']

    return np.array(coefs), np.array(Cs), n_iter


def _check_solver(solver, penalty, dual):
    if solver != 'lbfgs':
        warnings.warn("For diffprivlib, solver must be 'lbfgs'.", DiffprivlibCompatibilityWarning)
        solver = 'lbfgs'

    if penalty != 'l2':
        raise ValueError("Solver %s supports only l2 penalties, got %s penalty." % (solver, penalty))
    if dual:
        raise ValueError("Solver %s supports only dual=False, got dual=%s" % (solver, dual))
    return solver


def _check_multi_class(multi_class, solver, n_classes):
    del solver, n_classes

    if multi_class != 'ovr':
        warnings.warn("For diffprivlib, multi_class must be 'ovr'.", DiffprivlibCompatibilityWarning)
        multi_class = 'ovr'

    return multi_class
