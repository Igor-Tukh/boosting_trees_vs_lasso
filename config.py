import os

from utils import load_from_pickle, save_to_pickle

DATASET_DIR = './datasets'
RANDOM_STATE = 23923


def get_datasets_path(dataset_name, subset):
    return os.path.join(DATASET_DIR, f'{dataset_name}_{subset}.pkl')


def get_plot_path(plot_name, experiment_key):
    experiment_dir_path = os.path.join('plots', experiment_key)
    os.makedirs(experiment_dir_path, exist_ok=True)
    return os.path.join(experiment_dir_path, f'{plot_name}.png')


def get_resource(resource_name):
    resource_path = os.path.join('resources', f'{resource_name}.pkl')
    if os.path.exists(resource_path):
        return load_from_pickle(resource_path)
    return None


def get_resource_or_build(resource_name, function, args):
    resource = get_resource(resource_name)
    if resource is None:
        resource = function(*args)
        save_to_pickle(resource, os.path.join('resources', f'{resource_name}.pkl'))
    return resource
