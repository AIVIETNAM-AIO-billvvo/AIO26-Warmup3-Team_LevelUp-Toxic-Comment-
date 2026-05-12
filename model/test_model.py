import torch
import pandas as pd
from typing import Union, List
import gradio as gr
from transformers import BertTokenizer, BertForSequenceClassification
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

from script.run_clean import clean_text
# ==========================================
# 1. CẤU HÌNH MODEL
# ==========================================
# Đặt tên Model Hugging Face của bạn vào đây
# Ví dụ: "voquangthua/toxic-comment-bert"
# Nếu bạn muốn chạy từ thư mục có sẵn ở máy, thay bằng: "./my_model"
MODEL_NAME = "tinhuynh79/toxic-comment-bert-v1"

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


# ==========================================
# 3. HÀM DỰ ĐOÁN & GIAO DIỆN GRADIO
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
    
    # 4. Trả kết quả format dictionary
    labels = [
        'Độc hại (Toxic)', 'Rất độc hại (Severe)', 'Tục tĩu (Obscene)', 
        'Đe dọa (Threat)', 'Xúc phạm (Insult)', 'Ghét bỏ sắc tộc (Identity Hate)'
    ]
    
    results = {labels[i]: float(probs[i]) for i in range(len(labels))}
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
        label="Dự đoán mức độ độc hại"
    ),
    title="🛡️ Hệ thống phát hiện bình luận độc hại (BERT)",
    description="Nhập một đoạn văn bản tiếng Anh để kiểm tra khả năng nhận diện các loại độc hại khác nhau.",
    examples=[
        ["I love you so much!"],
        ["You are an idiot and I hate you!"],
        ["I will find you and kill you!"]
    ],
    theme="default" # Giao diện mặc định chuẩn
)

if __name__ == "__main__":
    # Khởi chạy server. (share=False khi chạy ở máy local, nếu muốn tạo link public dùng share=True)
    interface.launch(share=False, debug=True)