import numpy as np


def normalize_data(data):
    mean = np.mean(data)
    std = np.std(data)
    normalized_data = (data - mean) / std
    return normalized_data

def strip_quotes(s):
    return s.strip().strip('"').strip() if isinstance(s, str) else s

def convert_numeric(x):
    if not isinstance(x, str):
        return x
    
    try:
        # 1. Thử ép kiểu INT (số nguyên)
        # Không phải '1.335' hay '9.3e-05'
        if '.' not in x and 'e' not in x.lower():
            return int(x)
    except (ValueError, TypeError):
        pass 
    
    try:
        # 2. Thử ép kiểu FLOAT (số thực)
        # Ép kiểu '1.335' và '9.3448e-05'
        return float(x)
    except (ValueError, TypeError):
        # 3. Nếu không phải dạng số
        # Trả về chuỗi gốc
        return x

def load_csv(DATA_PATH):
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        header = f.readline().strip().split(',')
    data = np.genfromtxt(DATA_PATH, delimiter=',', skip_header=1, dtype=str)
    return data, header

def one_hot_encoding(col):
    unique_vals = np.unique(col)
    
    col_reshaped = col.reshape(-1, 1) # shape (n, 1)
    # Dùng broadcasting để so sánh (n, 1) với (k,)
    # NumPy sẽ so sánh mỗi phần tử trong 'col_reshaped'
    # với TẤT CẢ các phần tử trong 'unique_vals'.
    # Kết quả là một ma trận Boolean (True/False) kích thước (n, k)
    boolean_matrix = (col_reshaped == unique_vals)
    
    return boolean_matrix.astype(int), unique_vals

def ordinal_encoding(col_idx, col_dict, col):
    col_keys = np.array(list(col_dict.keys()))
    col_values = np.array(list(col_dict.values()))
    sort_idx = np.argsort(col_keys)
    mapped = col_values[sort_idx]
    col_uni, inv = np.unique(col, return_inverse=True)
    return mapped[inv] # fancy indexing