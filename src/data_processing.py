import numpy as np

# Chuẩn hóa dữ liệu
def normalize_data(data):
    mean = np.mean(data)
    std = np.std(data)
    normalized_data = (data - mean) / std
    return normalized_data

# Bỏ qua các kí tự (") khi đọc dữ liệu dạng str
def strip_quotes(s):
    return s.strip().strip('"').strip() if isinstance(s, str) else s

# Ép kiểu dữ liệu số INT, FLOAT
def convert_numeric(x):
    if not isinstance(x, str):
        return x
    
    try:
        # 1. Thử ép kiểu INT
        # Không phải '1.335' hay '9.3e-05'
        if '.' not in x and 'e' not in x.lower():
            return int(x)
    except (ValueError, TypeError):
        pass 
    
    try:
        # 2. Thử ép kiểu FLOAT
        # Ép kiểu '1.335' và '9.3448e-05'
        return float(x)
    except (ValueError, TypeError):
        # 3. Nếu không phải dạng số
        # Trả về chuỗi gốc
        return x

# Đọc dữ liệu từ file .csv
def load_csv(DATA_PATH):
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        header = f.readline().strip().split(',')
    data = np.genfromtxt(DATA_PATH, delimiter=',', skip_header=1, dtype=str)
    return data, header

# Mã hóa One-hot trả về ma trận gồm các cột là
# các giá trị duy nhất, và tên các cột đó
def one_hot_encoding(col):
    unique_vals = np.unique(col)
    col_reshaped = col.reshape(-1, 1) 
    boolean_matrix = (col_reshaped == unique_vals)
    
    return boolean_matrix.astype(int), unique_vals

# Mã hóa có thứ tự, theo từ điển định nghĩa trước
def ordinal_encoding(col_dict, col):
    col_keys = np.array(list(col_dict.keys()))
    col_values = np.array(list(col_dict.values()))
    sort_idx = np.argsort(col_keys)
    mapped = col_values[sort_idx]
    col_uni, inv = np.unique(col, return_inverse=True)
    return mapped[inv]