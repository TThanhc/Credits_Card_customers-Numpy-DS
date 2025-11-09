import numpy as np


def normalize_data(data):
    mean = np.mean(data)
    std = np.std(data)
    normalized_data = (data - mean) / std
    return normalized_data


def load_csv(DATA_PATH):
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        header = f.readline().strip().split(',')
    data = np.genfromtxt(DATA_PATH, delimiter=',', skip_header=1, dtype=str)
    return data, header
