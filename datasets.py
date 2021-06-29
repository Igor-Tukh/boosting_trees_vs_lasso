from config import get_datasets_path, RANDOM_STATE
from utils import save_to_pickle, load_from_pickle

from sklearn.datasets import fetch_20newsgroups, fetch_california_housing, load_diabetes
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

import os


def load_20_newsgroups(subset='train'):
    dataset_path = get_datasets_path('20_newsgroups', subset)
    if not os.path.exists(dataset_path):
        train = fetch_20newsgroups(subset=subset)
        vectorizer = TfidfVectorizer()
        data = vectorizer.fit_transform(train.data), train.target
        save_to_pickle(data, dataset_path)
        return data
    return load_from_pickle(dataset_path)


def load_california_housing(subset='train'):
    dataset_path = get_datasets_path('california_housing', subset)
    if not os.path.exists(dataset_path):
        X, y = fetch_california_housing(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=RANDOM_STATE, test_size=0.25)
        save_to_pickle((X_train, y_train), get_datasets_path('california_housing', 'train'))
        save_to_pickle((X_test, y_test), get_datasets_path('california_housing', 'test'))
    return load_from_pickle(dataset_path)


def load_diabetes_dataset(subset='train'):
    dataset_path = get_datasets_path('diabetes', subset)
    if not os.path.exists(dataset_path):
        diabetes = load_diabetes()
        X, y = diabetes['data'], diabetes['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=RANDOM_STATE, test_size=0.3)
        save_to_pickle((X_train, y_train), get_datasets_path('diabetes', 'train'))
        save_to_pickle((X_test, y_test), get_datasets_path('diabetes', 'test'))
    return load_from_pickle(dataset_path)


if __name__ == '__main__':
    california_housing = load_california_housing()
    print(california_housing[0].shape, california_housing[1].shape)
