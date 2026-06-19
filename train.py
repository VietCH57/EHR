import os
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from tqdm import tqdm  

from model import EHRMambaTransformer
from dataset import load_and_preprocess_data, EHRDataset

def parse_args():
    parser = argparse.ArgumentParser(description="Train EHR Deep Learning Model")
    parser.add_argument('--train_path', type=str, default='/kaggle/input/datasets/hongvitchugo/dl-mini-project-1/train.pkl')
    parser.add_argument('--test_path', type=str, default='/kaggle/input/datasets/hongvitchugo/dl-mini-project-1/test.pkl')
    parser.add_argument('--epochs', type=str, default='15')
    parser.add_argument('--batch_size', type=str, default='64')
    parser.add_argument('--lr', type=str, default='0.001')
    parser.add_argument('--d_model', type=str, default='64')
    parser.add_argument('--nhead', type=str, default='4')
    parser.add_argument('--num_layers', type=str, default='2')
    parser.add_argument('--seed', type=str, default='42')
    parser.add_argument('--save_dir', type=str, default='./models')
    return parser.parse_args()

def main():
    args = parse_args()
    epochs, batch_size, lr = int(args.epochs), int(args.batch_size), float(args.lr)
    d_model, nhead, num_layers, seed = int(args.d_model), int(args.nhead), int(args.num_layers), int(args.seed)
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Kiểm tra và hiển thị thiết bị phần cứng khi khởi tạo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 50)
    print(f" KHỞI TẠO THIẾT BỊ: Đang sử dụng [{device.upper()}] để huấn luyện mô hình.")
    print("=" * 50)
    
    print("Đang tải và tiền xử lý dữ liệu từ file pkl...")
    train_df, test_df, num_cols, cat_cols, cat_dims, target_col, id_col = load_and_preprocess_data(args.train_path, args.test_path)
    
    meta = {'num_cols': num_cols, 'cat_cols': cat_cols, 'cat_dims': cat_dims, 'target_col': target_col, 'id_col': id_col}
    with open(os.path.join(args.save_dir, 'metadata.pkl'), 'wb') as f:
        pickle.dump(meta, f)
        
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    X, y = train_df.copy(), train_df[target_col].values
    oof_predictions = np.zeros(len(train_df))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n⚡ Huấn luyện Fold {fold + 1}/5")
        train_loader = DataLoader(EHRDataset(X.iloc[train_idx].reset_index(drop=True), num_cols, cat_cols, target_col), batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(EHRDataset(X.iloc[val_idx].reset_index(drop=True), num_cols, cat_cols, target_col), batch_size=batch_size, shuffle=False)
        
        model = EHRMambaTransformer(max(len(num_cols), 1), max(len(cat_cols), 1), cat_dims, d_model, nhead, num_layers).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        best_auc = 0.0
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            
            # Sử dụng tqdm tạo progress bar chạy cho từng Batch
            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1:02d}/{epochs:02d}", leave=False)
            for batch in pbar:
                optimizer.zero_grad()
                logits = model(batch['num_feats'].to(device), batch['cat_feats'].to(device), batch['deltas'].to(device))
                loss = criterion(logits, batch['target'].to(device))
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                # Hiển thị trạng thái loss hiện tại lên thanh tiến trình
                pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Device": str(device).upper()})
                
            scheduler.step()
            
            # Đánh giá trên tập Validation cuối mỗi Epoch
            model.eval()
            val_preds, val_targets = [], []
            with torch.no_grad():
                for batch in val_loader:
                    logits = model(batch['num_feats'].to(device), batch['cat_feats'].to(device), batch['deltas'].to(device))
                    val_preds.extend(torch.sigmoid(logits).cpu().numpy())
                    val_targets.extend(batch['target'].cpu().numpy())
            
            val_auc = roc_auc_score(val_targets, val_preds)
            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(model.state_dict(), os.path.join(args.save_dir, f'model_fold_{fold}.pt'))
                
        print(f" Kết quả Fold {fold + 1} - Best Val ROC-AUC: {best_auc:.4f}")
        
        # Tạo dự đoán Out-of-Fold (OOF) từ checkpoint tốt nhất
        model.load_state_dict(torch.load(os.path.join(args.save_dir, f'model_fold_{fold}.pt')))
        model.eval()
        fold_preds = []
        with torch.no_grad():
            for batch in val_loader:
                logits = model(batch['num_feats'].to(device), batch['cat_feats'].to(device), batch['deltas'].to(device))
                fold_preds.extend(torch.sigmoid(logits).cpu().numpy())
        oof_predictions[val_idx] = fold_preds

    print("\n" + "=" * 50)
    print(f" CHỈ SỐ TOÀN BỘ MÔ HÌNH (5-Fold OOF ROC-AUC): {roc_auc_score(y, oof_predictions):.4f}")
    print("=" * 50)

if __name__ == '__main__':
    main()