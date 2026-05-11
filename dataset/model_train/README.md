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
- **Ngôn ngữ:** Python 3.x
- **Framework Deep Learning:** PyTorch
- **Thư viện NLP:** Hugging Face Transformers & Tokenizers
- **Giao diện người dùng:** Gradio
- **Xử lý dữ liệu:** Pandas, Numpy
- **Đánh giá:** Scikit-learn (ROC-AUC, F1-score)
- **Công cụ theo dõi:** Matplotlib
## Thông số đánh giá mô hình trong quá trình training
![Mô tả ảnh](AIO26-Warmup3-Team_LevelUp-Toxic-Comment-\dataset\model_train\images\image-1.png)


## Loss và metric
![alt text](AIO26-Warmup3-Team_LevelUp-Toxic-Comment-\dataset\model_train\images\image-2.png)
## 🚀 Hướng dẫn cài đặt (Local)
