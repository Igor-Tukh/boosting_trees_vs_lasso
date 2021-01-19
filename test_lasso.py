import numpy as np
import warnings

from sklearn.linear_model import Lasso

warnings.filterwarnings('ignore')

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn import linear_model

from tqdm import tqdm

from config import RANDOM_STATE, get_plot_path, get_resource_or_build, get_resource
from datasets import load_california_housing, load_diabetes_dataset
from parse_monomials import parse_decision_tree, parse_monomials_from_decision_tree_ensemble, _sift_monomials
from utils import build_plot, save_to_pickle, get_regression_metrics


def train_gradient_boosting_classifier(X, y):
    return GradientBoostingClassifier(random_state=RANDOM_STATE).fit(X, y)


def train_gradient_boosting_regressor(X, y, params):
    # return GradientBoostingRegressor(random_state=RANDOM_STATE,
    #                                  criterion='friedman_mse',
    #                                  max_depth=20).fit(X, y)
    return GradientBoostingRegressor(**params).fit(X, y)


def train_lasso(X, y, n_iterations, alpha=1.):
    return linear_model.Lasso(max_iter=n_iterations, alpha=alpha).fit(X, y)


def transform_X(X, ms):
    X_mon = np.zeros(shape=(X.shape[0], len(ms)))
    for i in tqdm(range(X.shape[0])):
        for j, m in enumerate(ms):
            X_mon[i, j] = m(X[i])
    return X_mon


def test_lasso(regressor, X_train, X_test, y_train, y_test, experiment_key, max_iter=300, alpha=0.001,
               plot_rate=50, iter_step=25, plot_ind=0, build_plots=True):
    reg_mse = mean_squared_error(y_test, regressor.predict(X_test))
    reg_r2 = regressor.score(X_test, y_test)
    reg_train_mse = mean_squared_error(y_train, regressor.predict(X_train))
    reg_train_r2 = regressor.score(X_train, y_train)
    print(f'GB Regressor: R2 {reg_r2}, MSE: {reg_mse}')
    # monomials = parse_monomials_from_decision_tree_ensemble(regressor, debug_output=False)
    monomials = None
    X_train_mon = get_resource_or_build(f'{experiment_key}_X_train_mon', transform_X, (X_train, monomials))
    X_test_mon = get_resource_or_build(f'{experiment_key}_X_test_mon', transform_X, (X_test, monomials))
    mses = []
    r2s = []
    train_mses = []
    train_r2s = []
    for iter_number in np.arange(1, max_iter + 2, iter_step):
        lasso = train_lasso(X_train_mon, y_train, iter_number, alpha)
        train_mses.append(mean_squared_error(y_train, lasso.predict(X_train_mon)))
        train_r2s.append(lasso.score(X_train_mon, y_train))
        mses.append(mean_squared_error(y_test, lasso.predict(X_test_mon)))
        r2s.append(lasso.score(X_test_mon, y_test))
        # print(f'Iteration {iter_number}: R^2 {r2s[-1]}, MSE: {mses[-1]}')
        if iter_number % plot_rate == 1 and iter_number > 1 and build_plots:
            build_plot(np.arange(1, iter_number + 1, iter_step), [(r2s, 'Test R2'),
                                                                  (train_r2s, 'Train R2')],
                       get_plot_path(f'{plot_ind}_{alpha}_lasso_{iter_number}_iterations_r2', experiment_key),
                       x_label='Iterations', y_label='Score', title=f'Lasso scores, labda={alpha}',
                       x_lines=[(reg_r2, 'GB Test R2'),
                                (reg_train_r2, 'GB Train R2')],
                       y_lim=(0, 1))
            build_plot(np.arange(1, iter_number + 1, iter_step), [(mses, 'Test MSE'),
                                                                  (train_mses, 'Train MSE')],
                       get_plot_path(f'{alpha}_lasso_{iter_number}_iterations_mse', experiment_key),
                       x_label='Iterations', y_label='Score', title=f'Lasso scores, labda={alpha}',
                       x_lines=[(reg_mse, 'GB Test MSE'),
                                (reg_train_mse, 'GB Train MSE')])
    return r2s, mses, train_r2s, train_mses


def test_diabetes_lasso():
    reg, X_train, X_val, X_test, y_train, y_val, y_test = get_resource('diabetes')
    best_r2 = -1
    best_lambda = None
    for i, alpha in enumerate(np.arange(2.1, 3.91, 0.1)):
        r2s = test_lasso(reg, X_train, X_test, y_train, y_test, 'diabetes_01', max_iter=100, alpha=alpha, iter_step=1,
                         plot_rate=100, build_plots=False)[0]
        cur_r2 = np.max(r2s)
        print(f'Lambda {alpha}: best test score {cur_r2}, {np.argmax(r2s)} iterations')
        if best_lambda is None or cur_r2 > best_r2:
            best_r2 = cur_r2
            best_lambda = alpha
    print(f'Best r2: {best_r2}, best lambda: {best_lambda}')
    # test_lasso(reg, X_train, X_test, y_train, y_test, 'diabetes_01', max_iter=1000, alpha=0.1)


def test_diabetes_bootstrap():
    reg, X_train, X_val, X_test, y_train, y_val, y_test = get_resource('diabetes')
    experiment_key = 'diabetes_01'
    X_train_mon = get_resource_or_build(f'{experiment_key}_X_train_mon', transform_X, (X_train, None))
    X_test_mon = get_resource_or_build(f'{experiment_key}_X_test_mon', transform_X, (X_test, None))
    n = X_test.shape[0]
    diffs = []
    for _ in tqdm(range(500)):
        ids = np.random.choice(np.arange(n), n, replace=True)
        lasso = Lasso(alpha=2.8, max_iter=12).fit(X_train_mon[ids], y_train[ids])
        diffs.append(reg.score(X_test[ids], y_test[ids]) - lasso.score(X_test_mon[ids], y_test[ids]))
    print(np.percentile(diffs, 5), np.percentile(diffs, 95), np.mean(diffs))


if __name__ == '__main__':
    test_diabetes_bootstrap()
    # test_diabetes_lasso()
    # test_lasso(X_train, X_test, y_train, y_test)
    # reg = train_gradient_boosting_regressor(X_train, y_train)
    # print(get_regression_metrics(reg, X_train, y_train))
    # print(get_regression_metrics(reg, X_test, y_test))
