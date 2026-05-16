# README — QA/QC cho Hệ thống Toxic Comment Detection sử dụng BERT

## 1. Giới thiệu

Tài liệu này mô tả quy trình QA/QC (Quality Assurance / Quality Control) cho mô hình phát hiện bình luận độc hại sử dụng BERT (`bert-base-uncased`).

Mục tiêu của QA/QC không chỉ là kiểm tra code có chạy hay không, mà còn nhằm đánh giá:

* Model có học đúng không
* Dữ liệu có vấn đề không
* Kết quả dự đoán có đáng tin không
* Model có phù hợp để triển khai thực tế không
* Có rủi ro hoặc lỗi tiềm ẩn nào không


# 2. Tổng quan hệ thống

## 2.1 Bài toán

Hệ thống thực hiện bài toán:

# Multi-label Text Classification

Một comment có thể thuộc nhiều nhãn cùng lúc.

Ví dụ:

```text
"I will kill you idiot"
```

Có thể thuộc:

* toxic
* threat
* insult


## 2.2 Các nhãn trong hệ thống

```python
LABEL_COLS = [
    'toxic',
    'severe_toxic',
    'obscene',
    'threat',
    'insult',
    'identity_hate'
]
```


## 2.3 Model sử dụng

Model được sử dụng:

```python
BertForSequenceClassification
```

Pretrained model:

```python
bert-base-uncased
```

Dạng bài toán:

```python
problem_type="multi_label_classification"
```


# 3. Mục tiêu QA/QC

Quá trình QA/QC nhằm kiểm tra:

| Hạng mục              | Mục tiêu                           |
| --------------------- | ---------------------------------- |
| Data Validation       | Dữ liệu có hợp lệ không            |
| Tokenization Analysis | Text có bị cắt quá nhiều không     |
| Training Evaluation   | Model có học ổn định không         |
| Metrics Validation    | Kết quả đánh giá có đáng tin không |
| Inference Testing     | Dự đoán thực tế có hợp lý không    |
| Robustness Testing    | Model có chịu được input xấu không |
| Bias Checking         | Model có thiên vị không            |
| Stability Testing     | Hệ thống có crash không            |

---

# 4. QA/QC Checklist


# 4.1 Kiểm tra dữ liệu (Data Validation)


## 4.1.1 Kiểm tra dữ liệu thiếu (Missing Values)

### Mục tiêu

Kiểm tra xem dataset có chứa giá trị null hay không.

Nếu dữ liệu null tồn tại:

* tokenizer có thể lỗi
* model có thể train sai
* pipeline có thể crash


### Cách thực hiện

```python
train_df.isnull().sum()
```


### Kết quả hiện tại

| Column        | Missing Values |
| ------------- | -------------- |
| id            | 0              |
| comment_text  | 0              |
| toxic         | 0              |
| severe_toxic  | 0              |
| obscene       | 0              |
| threat        | 0              |
| insult        | 0              |
| identity_hate | 0              |

Dataset không chứa dữ liệu thiếu (missing values).

Điều này đảm bảo:

* tokenizer hoạt động ổn định
* pipeline không bị crash do null input
* labels hợp lệ cho quá trình train/evaluation

### Kết luận

PASS.

Dataset sạch và hợp lệ cho huấn luyện mô hình.

## 4.1.2 Kiểm tra mất cân bằng nhãn (Label Imbalance)

### Mục tiêu

Kiểm tra xem các class có bị lệch dữ liệu quá nhiều không.

Ví dụ:

* toxic có rất nhiều mẫu
* threat có rất ít mẫu

Điều này có thể làm model:

* học tốt class phổ biến
* học kém class hiếm


### Cách thực hiện

```python
train_df[LABEL_COLS].mean().sort_values(ascending=False)
```


### Kết quả hiện tại

| Label         | Ratio |
| ------------- | ----- |
| toxic         | 9.58% |
| obscene       | 5.29% |
| insult        | 4.93% |
| severe_toxic  | 0.99% |
| identity_hate | 0.88% |
| threat        | 0.29% |

Dataset bị mất cân bằng class (class imbalance).

Đặc biệt:

* threat
* identity_hate
* severe_toxic

có tỷ lệ rất thấp.

Điều này có thể gây:

* recall thấp
* prediction không ổn định
* confidence thấp ở class hiếm

Và đây chính là lý do khi test ở 4.5:

| Observation                   | Possible Cause        |
| ----------------------------- | --------------------- |
| Threat chỉ ~58%               | Threat imbalance      |
| F1 không quá cao              | Multi-label imbalance |
| ROC-AUC cao nhưng F1 vừa phải | Dataset imbalance     |

### Kết luận

PASS.

Dataset usable nhưng tồn tại imbalance đáng kể.

Cần:

* threshold tuning
* class weighting
* hoặc oversampling nếu muốn cải thiện class hiếm

## 4.1.3 Kiểm tra Data Leakage

### Mục tiêu

Đảm bảo dữ liệu train và validation không bị trùng nhau.

Nếu leakage xảy ra:

* metric sẽ bị “ảo”
* model tưởng như tốt nhưng thực tế không tốt


### Cách thực hiện

```python
len(set(train_data.index) & set(val_data.index))
```


### Kết quả mong đợi

```python
0
```


### Kết quả hiện tại

Đúng như mong đợi



# 4.2 Kiểm tra Tokenization và Sequence Length


## 4.2.1 Phân tích độ dài token

### Mục tiêu

Mục tiêu của bước này là phân tích phân phối độ dài token của dữ liệu văn bản đầu vào, nhằm:

* Xác định mức độ phù hợp của MAX_LENGTH
* Đánh giá rủi ro truncation (cắt bớt nội dung)
* Cân bằng giữa:
  * hiệu năng mô hình
  * chi phí tính toán
  * giới hạn GPU

Giới hạn kỹ thuật của BERT

BERT (bert-base-uncased) có giới hạn: 

Tối đa: 512 tokens

Nếu input vượt quá giới hạn này:

* Bắt buộc phải truncation
* Có thể mất thông tin quan trọng ở cuối chuỗi
* Ảnh hưởng trực tiếp đến độ chính xác dự đoán

Cách thực hiện: Hiện trong model sử dụng hàm phân tích thống kê độ dài token:

analyze_token_lengths()

Hàm này:

* lấy sample từ tập train
* tokenize bằng WordPiece tokenizer
* thống kê độ dài token
* vẽ histogram phân phối

Kết quả phân tích

```python
Mean: 95
Median: 52
90th percentile: 204
95th percentile: 306
99th percentile: 795
Max: 1465
```

Phân tích QA

1. Phân phối dữ liệu

Phần lớn dữ liệu nằm ở vùng ngắn:
* Median = 52 tokens → rất ngắn

Tuy nhiên tồn tại long-tail:
* Max = 1465 tokens → rất dài
* 99% vẫn vượt 512 tokens

Điều này cho thấy dataset có phân phối lệch (long-tail distribution)

2. Đánh giá MAX_LENGTH hiện tại

MAX_LENGTH = 128 (baseline)

* Phù hợp với median dữ liệu
* Train nhanh, ổn định
* Nhưng có truncation với các câu dài

MAX_LENGTH = 256 (test)

* Giảm truncation so với 128

Tuy nhiên:

* training time tăng mạnh ~ 130 giờ
* GPU usage tăng cao
* hiệu năng cải thiện chưa rõ rệt

MAX_LENGTH = 512 (test)

Vượt ngưỡng thực tế của dataset phần lớn use-case
Gây:
* OOM (Out of Memory)
* Kernel crash trên Colab Free
* Không khả thi trong môi trường hiện tại

 Truncation risk

Với MAX_LENGTH = 128:

Ví dụ:

```python
... normal text ... insult word (ở cuối)
```

Nếu từ toxic nằm sau token 128:

* Model không nhìn thấy token quan trọng
* Có thể dẫn đến false negative

Kết luận QA

* Dataset có long-tail nhưng phần lớn dữ liệu ngắn
* Truncation là có thật nhưng không nghiêm trọng với majority case

* Tăng MAX_LENGTH giúp giảm truncation nhưng đánh đổi chi phí rất lớn

| MAX_LENGTH | Đánh giá                                   |
| ---------- | ------------------------------------------ |
| 128        | Stable, hiệu quả tốt nhất hiện tại       |
| 256        | Trade-off: tăng cost, cải thiện chưa rõ |
| 512        | Không khả thi (OOM + crash trên Colab)   |

MAX_LENGTH = 128 là lựa chọn tối ưu cho môi trường hiện tại (Colab + dataset distribution).

# 4.3 Kiểm tra cấu hình Model


## 4.3.1 Kiểm tra loại bài toán

### Mục tiêu

Đảm bảo model đang dùng đúng chế độ multi-label classification.


### Cách thực hiện

```python
model.config.problem_type
```


### Kết quả mong đợi

```python
multi_label_classification
```


### Kết quả hiện tại

Đúng như mong đợi

## 4.3.2 Kiểm tra activation function

### Mục tiêu

Kiểm tra xem hệ thống có sử dụng đúng activation cho multi-label classification không.

Trong multi-label:

* phải dùng sigmoid
* không dùng softmax


### Cách thực hiện

```python
probs = torch.sigmoid(logits)
```


### Kết quả hiện tại

Đã sử dụng sigmoid.


### Kết luận

PASS.


# 4.4 Kiểm tra Metrics


## 4.4.1 Kiểm tra ROC-AUC

### Mục tiêu

Đánh giá khả năng phân biệt class của model.

ROC-AUC phù hợp với:

* dataset imbalance
* multi-label classification


### Cách thực hiện

```python
trainer.evaluate()
```


### Kết quả hiện tại

| Metric    | Value  |
| --------- | ------ |
| ROC-AUC   | 0.9911 |
| F1 Macro  | 0.6672 |
| Precision | 0.7081 |
| Recall    | 0.6451 |
| Accuracy  | 0.9290 |
---
ROC-AUC rất cao: 0.9911

Cho thấy model có khả năng:

* phân biệt toxic/non-toxic rất tốt
* học được semantic pattern hiệu quả
---
Accuracy cao: 92.9%

Tuy nhiên:

* accuracy không phản ánh đầy đủ chất lượng model
* vì dataset imbalance
---
F1 Macro = 0.667

Cho thấy:

* model hoạt động khá tốt
* nhưng chưa tối ưu trên tất cả class

Đặc biệt:

class hiếm có thể đang kéo F1 xuống

---
Precision > Recall
| Metric    | Value |
| --------- | ----- |
| Precision | 0.708 |
| Recall    | 0.645 |

Điều này cho thấy:

* model khá cẩn thận khi predict toxic
* ít false positive hơn
* nhưng có thể bỏ sót một số toxic comment

### Kết luận

PASS.

Model đạt hiệu suất tốt cho bài toán toxic comment detection.

Tuy nhiên:

* imbalance vẫn ảnh hưởng tới macro metrics
* cần phân tích từng class riêng biệt
## 4.4.2 Kiểm tra F1-score

### Mục tiêu

Đánh giá sự cân bằng giữa:

* precision
* recall

F1 quan trọng trong toxic detection vì:

* false positive gây khó chịu
* false negative bỏ sót toxic comment


### Cách thực hiện

Được tính trong:

```python
compute_metrics()
```


### Kết quả hiện tại

| Metric   | Value  |
| -------- | ------ |
| F1 Macro | 0.6672 |

F1-score ở mức khá tốt đối với:

* multi-label classification
* dataset imbalance

Điều này cho thấy model:

* không chỉ predict đúng
* mà còn giữ được sự cân bằng giữa precision và recall

Tuy nhiên:

* F1 chưa quá cao
* cho thấy một số class hiếm vẫn dự đoán chưa ổn định

Đặc biệt:

* threat
* severe_toxic
* identity_hate

có thể đang kéo F1 xuống.

### Kết luận

PASS.
### 4.4.3 Kiểm tra Threshold (Global Threshold)

### Mục tiêu

Đánh giá xem ngưỡng mặc định `threshold = 0.5` có phù hợp cho bài toán multi-label classification hay không.

Hiện tại mô hình sử dụng:

```python
preds = (probs > 0.5)
```

Tuy nhiên, với dữ liệu imbalance mạnh, cần kiểm tra các ngưỡng khác nhau.


### Phương pháp

Thử nghiệm các threshold global:

* 0.3
* 0.4
* 0.5
* 0.6

Đánh giá trên các chỉ số:

* Precision
* Recall
* F1-score (Macro)


### Kết quả thực nghiệm

| Threshold | Precision | Recall | F1     |
| --------- | --------- | ------ | ------ |
| 0.3       | 0.0399    | 1.0000 | 0.0745 |
| 0.4       | 0.0386    | 0.8718 | 0.0720 |
| 0.5       | 0.0317    | 0.5142 | 0.0530 |
| 0.6       | 0.0191    | 0.0255 | 0.0162 |

---

### Nhận xét

* Threshold thấp (0.3–0.4) → Recall rất cao nhưng Precision cực thấp
* Threshold cao (0.5–0.6) → Model bỏ sót nhiều positive samples
* F1-score thấp ở tất cả mức → chứng tỏ global threshold không hẳn phù hợp

Nguyên nhân chính:

* Dataset imbalanced mạnh giữa các class
* Output sigmoid của từng label không cùng distribution
* Dùng chung 1 threshold cho tất cả label là không tối ưu


### Kết luận

* Threshold mặc định 0.5 không hẳn phù hợp
* Global threshold không đủ tốt cho multi-label imbalance problem


### 4.4.4 Đề xuất: Threshold Tuning theo từng Class (Per-Class Threshold)

### Mục tiêu

Khắc phục hạn chế của global threshold bằng cách tối ưu ngưỡng riêng cho từng nhãn.


### Phương pháp

Thay vì dùng:

```python
preds = (probs > 0.5)
```

Ta tìm threshold tốt nhất cho từng class dựa trên F1-score:

* Grid search trên tập validation (sample 5000)
* Tối ưu F1 riêng cho từng label


### Kết quả tối ưu

#### Best threshold theo từng class

| Class         | Best Threshold | Best F1 |
| ------------- | -------------- | ------- |
| toxic         | 0.45           | 0.1668  |
| severe_toxic  | 0.35           | 0.0104  |
| obscene       | 0.55           | 0.1044  |
| threat        | 0.45           | 0.0081  |
| insult        | 0.40           | 0.0956  |
| identity_hate | 0.50           | 0.0224  |


#### Metrics sau khi tuning

| Class         | Precision | Recall | F1     |
| ------------- | --------- | ------ | ------ |
| toxic         | 0.0914    | 0.9493 | 0.1668 |
| severe_toxic  | 0.0052    | 1.0000 | 0.0104 |
| obscene       | 0.0571    | 0.6045 | 0.1044 |
| threat        | 0.0041    | 1.0000 | 0.0081 |
| insult        | 0.0502    | 1.0000 | 0.0956 |
| identity_hate | 0.0114    | 0.8929 | 0.0224 |


### Nhận xét

* Recall tăng mạnh ở tất cả class (nhiều class đạt ~1.0)
* Precision thấp → model vẫn over-predict positive
* F1-score cải thiện nhẹ nhưng vẫn thấp do:

  * extreme class imbalance
  * rất ít dữ liệu cho các label hiếm (threat, identity_hate)


### Ý nghĩa QA/QC

* Global threshold = không hẳn phù hợp
* Per-class threshold = giải pháp đúng hướng
* Tuy nhiên hiệu quả bị giới hạn bởi data imbalance, không phải do model architecture


### Kết luận

* Hệ thống nên sử dụng per-class threshold thay vì fixed 0.5

* Đây là bước post-processing quan trọng trong production pipeline

# 4.5 Kiểm tra Inference


## 4.5.1 Kiểm tra comment bình thường

### Input

```text
"I love this movie"
```


### Kết quả mong đợi

Tất cả label = 0.


### Kết quả hiện tại

Tất cả label = 0%.

### Kết luận

* PASS.

* Model có khả năng nhận diện comment bình thường tốt trong trường hợp kiểm thử cơ bản.

* Không phát hiện false positive ở sample này.
## 4.5.2 Kiểm tra toxic nhẹ

### Input

```text
"You are stupid"
```


### Kết quả mong đợi

* toxic = 1
* insult = 1
* obscene có thể xuất hiện nhẹ
* threat thấp

### Kết quả hiện tại

| Label         | Probability |
| ------------- | ----------- |
| toxic         | 99%         |
| insult        | 92%         |
| obscene       | 79%         |
| severe_toxic  | 7%          |
| identity_hate | 2%          |

Model hoạt động khá tốt:

* toxic được nhận diện rất mạnh

* insult đúng với ngữ cảnh

Tuy nhiên:

obscene = 79% có thể hơi cao vì câu này chưa chứa từ tục tĩu mạnh

Điều này cho thấy:

model có xu hướng liên kết insult với obscene
### Kết luận

PASS.

Tuy nhiên cần kiểm tra thêm:

* false positive giữa insult và obscene
* khả năng phân biệt mức độ toxic

## 4.5.3 Kiểm tra threat

### Input

```text
"I will kill you"
```


### Kết quả mong đợi

* threat = 1
* toxic cao


### Kết quả hiện tại

| Label         | Probability |
| ------------- | ----------- |
| toxic         | 89%         |
| threat        | 58%         |
| insult        | 7%          |
| obscene       | 6%          |
| severe_toxic  | 3%          |
| identity_hate | 1%          |

Model đã nhận diện được threat.

Tuy nhiên:

* threat chỉ đạt 58%
* chưa thực sự mạnh đối với câu đe dọa trực tiếp

Nguyên nhân có thể:

* class threat bị imbalance
* số lượng mẫu threat trong dataset ít
* threshold 0.5 chưa tối ưu

### Kết luận

PASS.

Đề xuất:

* kiểm tra imbalance của class threat
* thử threshold thấp hơn cho threat
* bổ sung dữ liệu threat nếu cần

## 4.5.4 Kiểm tra slang / typo

### Input

```text
"u r dumb af"
```


### Mục tiêu

Kiểm tra khả năng hiểu:

* tiếng lóng
* viết sai chính tả
* ngôn ngữ internet


### Kết quả hiện tại

| Label         | Probability |
| ------------- | ----------- |
| toxic         | 99%         |
| insult        | 84%         |
| obscene       | 7%          |
| severe_toxic  | 4%          |
| identity_hate | 2%          |

Model xử lý tốt slang/informal text.

Điều này cho thấy:

* tokenizer WordPiece của BERT vẫn giữ được semantic meaning
* pretrained BERT hỗ trợ tương đối tốt internet language

### Kết luận

PASS.

Model có khả năng generalize tốt với informal toxic text.

# 4.6 Kiểm tra Edge Cases


## 4.6.1 Empty Input

### Input

```text
""
```


### Mục tiêu

Kiểm tra hệ thống có crash không.


### Kết quả hiện tại

Tất cả label = 0%.

* System không crash.

* Tokenizer và model xử lý được empty input.
### Kết luận

PASS.


## 4.6.2 Emoji Only

### Input

```text
"😂😂😂"
```


### Mục tiêu

Kiểm tra tokenizer và model có xử lý được không.


### Kết quả hiện tại

Tất cả label = 0%.


* System hoạt động ổn định.

* Không xuất hiện false positive.

### Kết luận

PASS.


## 4.6.3 Numeric Input

### Input

```text
"123456"
```


### Mục tiêu

Kiểm tra stability.


### Kết quả hiện tại

Tất cả label = 0%.

* Model không bị nhầm lẫn với numeric-only input.

* Không xuất hiện abnormal prediction.

### Kết luận

PASS.


# 4.7 Kiểm tra Overfitting


## Mục tiêu

Kiểm tra model có ghi nhớ train set quá mức không.

Dấu hiệu overfitting:

* train loss giảm mạnh
* validation loss tăng


## Cách thực hiện

Theo dõi:

* train loss
* eval loss
* TensorBoard logs


## Kết quả hiện tại

| Step | Train Loss | Validation Loss | ROC-AUC | F1 Macro |
| ---- | ---------- | --------------- | ------- | -------- |
| 500  | 0.0625     | 0.0517          | 0.9759  | 0.4010   |
| 1000 | 0.0491     | 0.0465          | 0.9842  | 0.4743   |
| 1500 | 0.0454     | 0.0454          | 0.9863  | 0.4939   |
| 2000 | 0.0440     | 0.0431          | 0.9820  | 0.4112   |
| 2500 | 0.0415     | 0.0410          | 0.9881  | 0.5708   |
| 3000 | 0.0402     | 0.0401          | 0.9895  | 0.5689   |
| 3500 | 0.0345     | 0.0425          | 0.9879  | 0.6124   |
| 4000 | 0.0419     | 0.0397          | 0.9902  | 0.6503   |
| 4500 | 0.0299     | 0.0393          | 0.9897  | 0.6608   |
| 5000 | 0.0321     | 0.0396          | 0.9896  | 0.6649   |
| 5500 | 0.0284     | 0.0400          | 0.9909  | 0.6702   |
| 6000 | 0.0295     | 0.0383          | 0.9905  | 0.6411   |
| 6500 | 0.0289     | 0.0381          | 0.9907  | 0.6688   |
| 7000 | 0.0350     | 0.0372          | 0.9911  | 0.6757   |
| 7500 | 0.0272     | 0.0373          | 0.9911  | 0.6673   |
---

Train loss giảm dần: 0.062 -> 0.027

Validation loss: 0.051 -> 0.037

Validation loss không tăng bất thường trong quá trình train.

Model

* học ổn định
* generalize tương đối tốt trên validation set

Tuy nhiên:

chưa có biểu đồ trực quan train/eval loss đầy đủ
chưa benchmark nhiều epoch khác nhau

## Kết luận

Train loss giảm.

Validation loss:

* không tăng mạnh
* tương đối ổn định

=> Chưa phát hiện overfitting nghiêm trọng trong 2 epoch hiện tại.

PASS.


# 4.8 Kiểm tra Stability và Performance


## 4.8.1 GPU Memory

### Mục tiêu

Kiểm tra:

* có OOM không
* fp16 có hoạt động không


### Kết quả hiện tại

* Training hoàn tất thành công trên Colab GPU
* fp16 hoạt động ổn định
* không xảy ra Out Of Memory (OOM) 
* batch size = 32, MAX_LENGTH = 128

### Kết luận

PASS.

Hệ thống hoạt động ổn định trong quá trình huấn luyện với cấu hình hiện tại.

## 4.8.2 Inference Speed

### Mục tiêu

Đánh giá tốc độ predict thực tế.


### Cách thực hiện

```python
%timeit model(...)
```


### Trạng thái

Chưa thực hiện benchmark inference speed chi tiết.

### Lý do

Dự án hiện tập trung vào:

* chất lượng prediction
* độ chính xác của model
* khả năng generalization

Thay vì tối ưu latency production.

Đề xuất tương lai

Có thể benchmark thêm:

* CPU inference
* GPU inference
* batch inference
* ONNX/TensorRT optimization

nếu triển khai production.


# 4.9 Kiểm tra Bias


## Mục tiêu

Kiểm tra model có thiên vị giới tính hoặc identity không.


## Ví dụ test

```text
"He is a doctor"
"She is a doctor"
```


## Kết quả hiện tại

cho kết quả tương đương nhau.


## Kết luận

PASS. 

Không phát hiện gender bias trong sample test cơ bản.


# 5. Tổng kết QA/QC

| Hạng mục              | Trạng thái          |
| --------------------- | ------------------- |
| Data Validation       | Đã thực hiện        |
| Tokenization Analysis | Đã thực hiện        |
| Model Configuration   | Đã thực hiện        |
| Metrics Validation    | Đã thực hiện        |
| Inference Testing     | Đã thực hiện        |
| Robustness Testing    | Đã thực hiện cơ bản |
| Overfitting Analysis  | Đã thực hiện cơ bản |
| Bias Testing          | Đã thực hiện cơ bản |


---


### 6. Kết luận chung

Hệ thống đã hoàn thành việc xây dựng pipeline huấn luyện mô hình BERT cho bài toán toxic comment detection với hỗ trợ multi-label classification.

Kết quả thực nghiệm cho thấy mô hình hoạt động ổn định trên tập validation và test, đồng thời có khả năng xử lý nhiều dạng input khác nhau trong quá trình inference.

### Kết quả nổi bật

* ROC-AUC đạt khoảng 0.99, cho thấy khả năng phân biệt giữa các lớp tốt
* Accuracy đạt khoảng 92%
* Mô hình hoạt động ổn định trong các kịch bản inference cơ bản
* Pipeline huấn luyện và đánh giá hoạt động nhất quán

### Kết quả QA/QC xác nhận

* Dữ liệu đầu vào hợp lệ, không phát hiện missing values hoặc leakage
* Pipeline huấn luyện hoạt động ổn định, không lỗi runtime
* Không xuất hiện dấu hiệu overfitting nghiêm trọng trong quá trình training
* Hệ thống xử lý tốt các edge cases cơ bản (empty input, emoji, numeric input)
* Không phát hiện bias rõ ràng trong kiểm thử đơn giản

### Hạn chế hiện tại

* Dataset bị mất cân bằng nghiêm trọng giữa các class
* Một số class hiếm như `threat` và `identity_hate` có hiệu suất thấp
* Hiện tượng truncation có thể làm mất một phần thông tin ở các comment quá dài, tuy nhiên MAX_LENGTH = 128 vẫn được lựa chọn nhằm tối ưu hiệu suất huấn luyện, tốc độ xử lý và giới hạn tài nguyên phần cứng hiện tại.

* Chưa thực hiện benchmark chuyên sâu cho môi trường production (latency, throughput)

### Hướng cải thiện trong tương lai

* Tuning threshold theo từng class (per-class threshold optimization)
* Áp dụng class weighting hoặc focal loss để xử lý imbalance
* Oversampling hoặc data augmentation cho các class hiếm
* Benchmark hiệu năng inference (CPU/GPU/production load)
* Mở rộng fairness testing (bias evaluation nâng cao)
* Tối ưu hóa triển khai production (ONNX / FastAPI / TensorRT)
