import os
import argparse
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm  

from model import EHRMambaTransformer
from dataset import load_and_preprocess_data, EHRDataset

def parse_args():
    parser = argparse.ArgumentParser(description="Inference Module")
    parser.add_argument('--train_path', type=str, default='/kaggle/input/datasets/hongvitchugo/dl-mini-project-1/train.pkl')
    parser.add_argument('--test_path', type=str, default='/kaggle/input/datasets/hongvitchugo/dl-mini-project-1/test.pkl')
    parser.add_argument('--model_dir', type=str, default='./models')
    parser.add_argument('--batch_size', type=str, default='64')
    parser.add_argument('--d_model', type=str, default='64')
    parser.add_argument('--nhead', type=str, default='4')
    parser.add_argument('--num_layers', type=str, default='2')
    parser.add_argument('--output_csv', type=str, default='submission.csv')
    return parser.parse_args()

def main():
    args = parse_args()
    batch_size, d_model, nhead, num_layers = int(args.batch_size), int(args.d_model), int(args.nhead), int(args.num_layers)
    
    # Kiểm tra và hiển thị thiết bị phần cứng khi khởi tạo suy diễn
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 50)
    print(f" Thiết bị: [{str(device).upper()}]")
    print("=" * 50)
    
    with open(os.path.join(args.model_dir, 'metadata.pkl'), 'rb') as f:
        meta = pickle.load(f)
        
    print("Đang nạp tập dữ liệu kiểm thử (test.pkl)...")
    _, test_df, num_cols, cat_cols, cat_dims, target_col, id_col = load_and_preprocess_data(args.train_path, args.test_path)
    test_loader = DataLoader(EHRDataset(test_df, num_cols, cat_cols, is_test=True), batch_size=batch_size, shuffle=False)
    
    ensemble_preds = np.zeros((len(test_df), 5))
    
    for fold in range(5):
        model_path = os.path.join(args.model_dir, f'model_fold_{fold}.pt')
        if not os.path.exists(model_path):
            print(f" Cảnh báo: Không tìm thấy trọng số cho fold {fold}. Bỏ qua.")
            continue
            
        model = EHRMambaTransformer(max(len(num_cols), 1), max(len(cat_cols), 1), cat_dims, d_model, nhead, num_layers).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        fold_preds = []
        # Tạo progress bar chạy tiến trình dự đoán cho tập dữ liệu Test
        pbar = tqdm(test_loader, desc=f"Đang dự đoán [Fold {fold + 1}/5]", leave=False)
        with torch.no_grad():
            for batch in pbar:
                logits = model(batch['num_feats'].to(device), batch['cat_feats'].to(device), batch['deltas'].to(device))
                fold_preds.extend(torch.sigmoid(logits).cpu().numpy())
                
        ensemble_preds[:, fold] = fold_preds
        
    # Tính toán xác suất trung bình từ 5 mô hình (Ensemble)
    final_prob = ensemble_preds.mean(axis=1)
    pred_id = test_df[id_col].values if id_col and id_col in test_df.columns else np.arange(len(test_df))
    
    print("Đang đóng gói và lưu tệp kết quả...")
    pd.DataFrame({
        'id': pred_id,
        'probability': final_prob,
        'prediction': (final_prob >= 0.5).astype(int)
    }).to_csv(args.output_csv, index=False)
    
    print(f" Kết quả xuất thành công ra file: {args.output_csv}")

if __name__ == '__main__':
    main()