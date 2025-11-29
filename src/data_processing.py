import numpy as np

# HÀM CHUẨN HÓA DỮ LIỆU
def normalize_data(data):
    mean = np.mean(data)
    std = np.std(data)
    normalized_data = (data - mean) / std
    return normalized_data

# BỎ KHOẢNG TRẮNG VÀ DẤU NGOẶC KÉP (") Ở ĐẦU CUỐI CHUỖI
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

# ĐỌC DỮ LIỆU TỪ FILE CSV
def load_csv(DATA_PATH):
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        header = f.readline().strip().split(',')
    data = np.genfromtxt(DATA_PATH, delimiter=',', skip_header=1, dtype=str)
    return data, header

# MÃ HÓA ONE-HOT ENCODING
# TRẢ VỀ MA TRẬN GỒM CÁC CỘT LÀ GIÁ TRỊ DUY NHẤT VÀ SỐ HÀNG DỮ LIỆU BAN ĐẦU
def one_hot_encoding(col):
    unique_vals = np.unique(col)
    col_reshaped = col.reshape(-1, 1) 
    boolean_matrix = (col_reshaped == unique_vals)
    
    return boolean_matrix.astype(int), unique_vals

# MÃ HÓA ORDINAL ENCODING
# THEO THỨ TỰ ĐỊNH NGHĨA TRONG TỪ ĐIỂN TRUYỀN VÀO
def ordinal_encoding(col_dict, col):
    col_keys = np.array(list(col_dict.keys()))
    col_values = np.array(list(col_dict.values()))
    sort_idx = np.argsort(col_keys)
    mapped = col_values[sort_idx]
    col_uni, inv = np.unique(col, return_inverse=True)
    return mapped[inv]