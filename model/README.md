# 🛡️ Hệ thống Phát hiện Bình luận Độc hại (Toxic Comment Detection)

Dự án này sử dụng mô hình ngôn ngữ tiên tiến **BERT (bert-base-uncased)** để tự động nhận diện và phân loại các bình luận độc hại. Đây là một bài toán phân loại đa nhãn (Multi-label Classification), cho phép một câu bình luận có thể thuộc về nhiều nhóm độc hại cùng lúc.

## 📌 Tổng quan dự án
Dựa trên tập dữ liệu từ cuộc thi **Jigsaw Toxic Comment Classification Challenge**, mô hình được huấn luyện để nhận diện 6 loại hành vi tiêu cực:
- **Toxic**: Độc hại nói chung.
- **Severe Toxic**: Rất độc hại/Cực đoan.
- **Obscene**: Tục tĩu, thô tục.
- **Threat**: Đe dọa bạo lực.
- **Insult**: Xúc phạm người khác.
- **Identity Hate**: Ghét bỏ sắc tộc, tôn giáo, giới tính...

## 🛠️ Công nghệ và Thư viện sử dụng
- **Ngôn ngữ:** Python 3.11
- **Framework Deep Learning:** PyTorch
- **Thư viện NLP:** Hugging Face Transformers & Tokenizers
- **Giao diện người dùng:** Gradio
- **Xử lý dữ liệu:** Pandas, Numpy
- **Đánh giá:** Scikit-learn (ROC-AUC, F1-score)
- **Công cụ theo dõi:** Matplotlib
## Thông số đánh giá mô hình trong quá trình training
### Train với 2 epochs 
<img width="1132" height="754" alt="image" src="https://github.com/user-attachments/assets/ef471659-8e14-4cbf-a339-ce3d743e05ab" />

### Train với 4 epochs 
<img width="579" height="432" alt="image" src="https://github.com/user-attachments/assets/0b99c0cb-f3b5-4300-953d-3feff8ea44cc" />
<img width="578" height="311" alt="image" src="https://github.com/user-attachments/assets/cddfdd41-8109-4809-bd33-bb55d074b7e5" />



## Loss và metric
### Train với 2 epochs 
<img width="1301" height="700" alt="image" src="https://github.com/user-attachments/assets/7fb7a0fe-93f9-41f2-9d65-d63601860dab" />
<img width="1301" height="700" alt="image" src="https://github.com/user-attachments/assets/9dcff059-9c8a-4bf0-b063-4f2d3c32b418" />

### Train với 4 epochs
<img width="1301" height="700" alt="image" src="https://github.com/user-attachments/assets/6611f583-0239-4b38-bfc2-f66aae27d257" />
<img width="1301" height="700" alt="image" src="https://github.com/user-attachments/assets/5777a551-5d71-4d70-871e-74d5d0bcc9ba" />

## So sánh khi tính toán metric với các Threshold khác nhau 
Vì dữ liệu mất cân bằng thiên về 1 phía toxic và dữ liệu khan hiếm (Không thể cải thiện data). Do đó, team đã chọn cách điều chỉnh threshold của từng nhãn để hạn chế tác hại từ việc mất cân bằng dữ liệu.
Tối ưu threshold:
<img width="353" height="181" alt="image" src="https://github.com/user-attachments/assets/5a6896d3-0ba9-4ccf-a2a3-e808e6bb0589" />



Kết quả:

<img width="404" height="197" alt="image" src="https://github.com/user-attachments/assets/5c57e08b-e7bb-4202-8c34-b4070cc87d89" />

Kết quả cho thấy F1 ở các label có sự thay đổi rõ rệt như: identity_hate, threat, severe_toxic.

## 🚀 Hướng dẫn cài đặt (Local)
Model đã được lưu trong huggingface hub: tinhuynh79/bert-toxic-comment-classifier
Tải các thư viện trong file requirements.txt: **pip install -r requirements.txt**

**Dô folder model -> chạy file test_model.py -> ctrl + click chuột vào local URL.**
<img width="819" height="137" alt="image" src="https://github.com/user-attachments/assets/23cbd089-63ea-4446-beb7-7e53503e398b" />

