import torch
import pandas as pd
from typing import Union, List
import gradio as gr
from transformers import BertTokenizer, BertForSequenceClassification
import sys
import os
import json 

current_dir = os.path.dirname(os.path.abspath(__file__)) 
parent_dir = os.path.dirname(current_dir) 
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from script.run_clean import clean_text

# ==========================================
# 1. CẤU HÌNH MODEL VÀ THRESHOLDS
# ==========================================
MODEL_NAME = "tinhuynh79/bert-toxic-comment-classifier"

# Danh sách các nhãn gốc (Key trong file json)
LABEL_COLS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

print("⏳ Đang tải Model và Tokenizer (lần đầu sẽ mất vài phút)...")
try:
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME)
except Exception as e:
    print(f"Lỗi tải model: {e}")
    print("Vui lòng kiểm tra lại tên MODEL_NAME hoặc kết nối mạng.")
    exit()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()
print(f"✅ Đã tải xong model! Đang chạy trên: {device}")

# Đọc file best_thresholds.json
try:
    # Đảm bảo file best_thresholds.json nằm cùng thư mục với file code này
    thresholds_path = os.path.join(current_dir, 'best_thresholds.json')
    with open(thresholds_path, 'r', encoding='utf-8') as f:
        best_thresholds = json.load(f)
    print(f"✅ Đã load best_thresholds: {best_thresholds}")
except FileNotFoundError:
    print("⚠️ Không tìm thấy file best_thresholds.json, sẽ dùng mặc định 0.5")
    best_thresholds = {col: 0.5 for col in LABEL_COLS}


# ==========================================
# 2. HÀM DỰ ĐOÁN & GIAO DIỆN GRADIO
# ==========================================

def predict_toxicity(text):
    # 1. Làm sạch văn bản bằng pipeline của bạn
    text_clean = clean_text(text)
    
    # 2. Tokenize và mã hóa
    inputs = tokenizer(
        text_clean, 
        return_tensors="pt", 
        truncation=True, 
        max_length=128, 
        padding='max_length'
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 3. Inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        # Sigmoid cho multi-label
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    
    # 4. Trả kết quả format dictionary kèm Threshold
    labels_vn = [
        'Độc hại (Toxic)', 'Rất độc hại (Severe)', 'Tục tĩu (Obscene)', 
        'Đe dọa (Threat)', 'Xúc phạm (Insult)', 'Ghét bỏ sắc tộc (Identity Hate)'
    ]
    
    results = {}
    for i, col in enumerate(LABEL_COLS):
        thresh = best_thresholds.get(col, 0.5)
        prob = float(probs[i])
        
        # Đánh dấu CÓ/KHÔNG tùy thuộc vào việc prob có vượt ngưỡng hay không
        if prob >= thresh:
            display_name = f"🔴 CÓ: {labels_vn[i]} (Ngưỡng: {thresh})"
        else:
            display_name = f"🟢 KHÔNG: {labels_vn[i]} (Ngưỡng: {thresh})"
            
        results[display_name] = prob
        
    return results

# Cấu hình giao diện Gradio
interface = gr.Interface(
    fn=predict_toxicity,
    inputs=gr.Textbox(
        lines=5, 
        placeholder="Nhập bình luận tại đây...", 
        label="Nội dung bình luận (Tiếng Anh)"
    ),
    outputs=gr.Label(
        num_top_classes=6, 
        label="Dự đoán mức độ độc hại (Thanh ngang thể hiện xác suất thực tế)"
    ),
    title="🛡️ Hệ thống phát hiện bình luận độc hại (BERT)",
    description="Nhập một đoạn văn bản tiếng Anh. Hệ thống sẽ so sánh xác suất dự đoán với Threshold tối ưu để đưa ra kết luận cuối cùng.",
    examples=[
        ["I love you so much!"],
        ["You are an idiot and I hate you!"],
        ["I will find you and kill you!"]
    ],
    theme="default" 
)

if __name__ == "__main__":
    # Khởi chạy server. (share=False khi chạy ở máy local, nếu muốn tạo link public dùng share=True)
    interface.launch(share=False, debug=True)