import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def load_and_preprocess_data(train_path, test_path):
    with open(train_path, 'rb') as f:
        train_df = pickle.load(f)
    with open(test_path, 'rb') as f:
        test_df = pickle.load(f)
        
    target_col = 'outcome' if 'outcome' in train_df.columns else 'target'
    id_col = 'id' if 'id' in train_df.columns else None
    
    all_cols = list(train_df.columns)
    if id_col: all_cols.remove(id_col)
    if target_col in all_cols: all_cols.remove(target_col)
    
    cat_cols = [col for col in all_cols if train_df[col].dtype == 'object' or train_df[col].nunique() < 5]
    num_cols = [col for col in all_cols if col not in cat_cols]
    
    cat_dims = []
    for col in cat_cols:
        le = LabelEncoder()
        train_df[col] = train_df[col].astype(str).fillna('missing')
        test_df[col] = test_df[col].astype(str).fillna('missing')
        train_df[col] = le.fit_transform(train_df[col])
        test_map = {label: idx for idx, label in enumerate(le.classes_)}
        test_df[col] = test_df[col].map(lambda s: test_map.get(s, 0))
        cat_dims.append(len(le.classes_))
        
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