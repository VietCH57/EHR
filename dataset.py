import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def extract_dataframe(obj):
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    
    if isinstance(obj, dict):
        # Thử tìm theo các key DataFrame phổ biến
        for key in ['dataframe', 'df', 'frame', 'train', 'test', 'table', 'data']:
            if key in obj and isinstance(obj[key], pd.DataFrame):
                return obj[key].copy()
        
        # Kiểm tra cấu trúc cặp X, y
        if 'X' in obj and 'y' in obj:
            X = np.asarray(obj['X'])
            y = np.asarray(obj['y'])
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            df = pd.DataFrame(X)
            df['target'] = y
            return df
        
        # Kiểm tra nếu dict chứa key là ID bệnh nhân và value là dict dữ liệu
        keys_list = list(obj.keys())
        if keys_list and isinstance(keys_list[0], str):
            records = []
            for key, value in obj.items():
                if isinstance(value, dict):
                    value_copy = value.copy()
                    value_copy['id'] = key
                    records.append(value_copy)
                elif isinstance(value, (list, np.ndarray)):
                    records.append({'id': key, 'data': value})
            
            if records:
                try:
                    return pd.DataFrame(records)
                except:
                    pass
        
        # Phương án cuối cùng: lấy DataFrame đầu tiên tìm thấy trong dict
        for value in obj.values():
            if isinstance(value, pd.DataFrame):
                return value.copy()
    
    return obj

def load_and_preprocess_data(train_path, test_path):
    # Đọc đối tượng pickle thô bằng pandas
    train_obj = pd.read_pickle(train_path)
    test_obj = pd.read_pickle(test_path)
    
    # Ép kiểu/Trích xuất sang DataFrame chuẩn
    train_df = extract_dataframe(train_obj)
    test_df = extract_dataframe(test_obj)
    
    # 1. Tự động xác định cột Target (nhãn) giống hệt logic code cũ của bạn
    target_candidates = [
        'mortality', 'in_hospital_mortality', 'hospital_mortality',
        'death', 'label', 'target', 'outcome'
    ]
    target_col = None
    for c in target_candidates:
        if c in train_df.columns:
            target_col = c
            break

    if target_col is None:
        binary_like = [c for c in train_df.columns if train_df[c].dropna().nunique() <= 2]
        diff_cols = [c for c in train_df.columns if c not in test_df.columns]
        guess_cols = [c for c in diff_cols if c in binary_like]
        if len(guess_cols) == 1:
            target_col = guess_cols[0]
        else:
            raise ValueError('Cannot confidently infer target column. Please set target_col manually.')

    # 2. Tự động xác định cột ID bệnh nhân
    id_candidates = ['id', 'patient_id', 'stay_id', 'hadm_id', 'subject_id']
    id_col = None
    for c in id_candidates:
        if c in test_df.columns:
            id_col = c
            break

    # Lấy danh sách các cột đặc trưng (loại bỏ id và target)
    all_cols = list(train_df.columns)
    if id_col and id_col in all_cols: 
        all_cols.remove(id_col)
    if target_col in all_cols: 
        all_cols.remove(target_col)
    
    # Phân loại biến danh mục (categorical) và biến số liên tục (numerical)
    cat_cols = [col for col in all_cols if train_df[col].dtype == 'object' or train_df[col].nunique() < 5]
    num_cols = [col for col in all_cols if col not in cat_cols]
    
    # Tiền xử lý biến danh mục (Categorical)
    cat_dims = []
    for col in cat_cols:
        le = LabelEncoder()
        train_df[col] = train_df[col].astype(str).fillna('missing')
        test_df[col] = test_df[col].astype(str).fillna('missing')
        
        train_df[col] = le.fit_transform(train_df[col])
        test_map = {label: idx for idx, label in enumerate(le.classes_)}
        test_df[col] = test_df[col].map(lambda s: test_map.get(s, 0))
        cat_dims.append(len(le.classes_))
        
    # Tiền xử lý biến số liên tục (Numerical)
    if len(num_cols) > 0:
        imputer = SimpleImputer(strategy='median')
        scaler = StandardScaler()
        
        train_df[num_cols] = imputer.fit_transform(train_df[num_cols])
        test_df[num_cols] = imputer.transform(test_df[num_cols])
        
        train_df[num_cols] = scaler.fit_transform(train_df[num_cols])
        test_df[num_cols] = scaler.transform(test_df[num_cols])
        
    return train_df, test_df, num_cols, cat_cols, cat_dims, target_col, id_col

class EHRDataset(Dataset):
    def __init__(self, df, num_cols, cat_cols, target_col=None, seq_len=24, is_test=False):
        self.num_data = df[num_cols].values.astype(np.float32) if len(num_cols) > 0 else np.zeros((len(df), 1), dtype=np.float32)
        self.cat_data = df[cat_cols].values.astype(np.int64) if len(cat_cols) > 0 else np.zeros((len(df), 1), dtype=np.int64)
        self.targets = df[target_col].values.astype(np.float32) if not is_test and target_col in df.columns else np.zeros(len(df), dtype=np.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.num_data)

    def __getitem__(self, idx):
        num_feat = self.num_data[idx]
        cat_feat = self.cat_data[idx]
        seq_num = np.tile(num_feat, (self.seq_len, 1))
        deltas = np.ones((self.seq_len, 1), dtype=np.float32)
        
        return {
            'num_feats': torch.tensor(seq_num),
            'cat_feats': torch.tensor(cat_feat),
            'deltas': torch.tensor(deltas),
            'target': torch.tensor(self.targets[idx])
        }