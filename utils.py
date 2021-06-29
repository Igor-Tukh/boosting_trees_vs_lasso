import matplotlib.pyplot as plt
import pickle

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

colors = ['r', 'b', 'g', 'y', 'black']


def load_from_pickle(filepath):
    with open(filepath, 'rb') as input_file:
        return pickle.load(input_file)


def save_to_pickle(obj, filepath):
    with open(filepath, 'wb') as output_file:
        pickle.dump(obj, output_file)


def build_plot(x, ys, output_path, title='', x_label='', y_label='', x_lines=None, x_lim=None,
               y_lim=None):
    plt.clf()
    for y, label in ys:
        plt.plot(x, y, label=label)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if x_lines is not None:
        for i, (x_line, label) in enumerate(x_lines):
            plt.axhline(y=x_line, linestyle='--', label=label, color=colors[i])
    if x_lim is not None:
        plt.xlim(*x_lim)
    if y_lim is not None:
        plt.ylim(*y_lim)
    plt.legend()
    plt.savefig(output_path)


def get_regression_metrics(reg, X, y):
    y_pred = reg.predict(X)
    return {
        'RMSE': mean_squared_error(y, y_pred, squared=False),
        'MSE': mean_squared_error(y, y_pred, reg.predict(X)),
        'MAE': mean_absolute_error(y, y_pred, reg.predict(X)),
        'R2': r2_score(y, y_pred)
    }
