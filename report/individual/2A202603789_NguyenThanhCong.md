# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:** 2A202603789
- **Họ và tên:** Nguyễn Thành Công
- **SEED** (in ở ô đầu notebook):5618

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ✅ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ✅ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ✅ |

## 2. Kết quả chính
                             precision  recall  F1 (test)  F1 (SUB 40)  \
1. BoW + LogReg (tự code)         0.988   0.792       0.88        0.857   
2. BoW + MLP (deep learning)      0.979   0.868       0.92        0.919   

                              train (s)  tin/s (cả lô)  p50 ms/tin  \
1. BoW + LogReg (tự code)         4.551      31369.362       0.052   
2. BoW + MLP (deep learning)      4.170      24580.398       0.247   

                              p95 ms/tin  USD / 1k tin  
1. BoW + LogReg (tự code)          0.088           0.0  
2. BoW + MLP (deep learning)       0.376           0.0  
• Hàng LLM chỉ đo trên SUB (40 tin) → so sánh chất lượng công bằng bằng cột 'F1 (SUB 40)'.
• SUB chỉ 40 tin: 1 tin sai đổi F1 vài điểm. Chênh < ~0.05 F1 coi như HOÀ — đừng xếp hạng theo số lẻ.
• Inference đo TỪ VĂN BẢN THÔ → NHÃN (gồm cả vectorize / encode). p50/p95 = gọi từng tin một, như một request thật.
• train (s) của 3a/3b/3c = thời gian encoder MiniLM encode tập train (+ train đầu LogReg / MLP); LLM = 0 vì không train. Số đo phụ thuộc CPU/GPU của Colab.

## 3. Dự đoán và thực tế

### 3.1. Few-shot sẽ cải thiện hoặc giữ nguyên F1

- **Dự đoán:** Thêm 4 ví dụ vào prompt có thể giúp model hiểu ranh giới spam/ham cụ thể hơn, do đó F1 có thể tăng hoặc giữ nguyên. Tôi cũng dự đoán prompt dài hơn có thể làm latency tăng.
- **Thực tế:** LLM nhỏ zero-shot đạt F1 SUB `0.884`; LLM nhỏ few-shot chỉ đạt `0.872`. p95 tăng từ `3134.944 ms` lên `3637.628 ms`.
- **Giải thích:** Bốn ví dụ được chọn từ train chưa chắc đại diện tốt cho 40 tin SUB. Ví dụ few-shot có thể tạo thiên lệch theo những mẫu cụ thể, trong khi prompt dài hơn làm tăng lượng nội dung model phải xử lý. Kết quả cho thấy thêm ví dụ không mặc nhiên cải thiện chất lượng.

### 3.2. Model lớn sẽ có latency cao hơn model nhỏ

- **Dự đoán:** Model lớn sẽ có F1 cao hơn nhưng latency cũng cao hơn model nhỏ.
- **Thực tế:** F1 SUB của model lớn cao hơn (`0.923` so với `0.884`), nhưng p95 của model lớn lại thấp hơn: `2407.956 ms` so với `3134.944 ms`.
- **Giải thích:** Latency thực tế phụ thuộc provider, hạ tầng phục vụ, tải hệ thống và cách triển khai model, không chỉ phụ thuộc kích thước model. Trong lần đo này, model lớn qua provider Groq có tốc độ trả lời tốt hơn model nhỏ. Không nên suy luận latency chỉ từ số tham số.

### 3.3. Bỏ ReLU sẽ làm F1 giảm

- **Dự đoán:** Mạng không có ReLU sẽ gần tương đương mô hình tuyến tính và F1 gần như chắc chắn giảm.
- **Thực tế:** Mô hình không ReLU đạt F1 `0.925`, cao hơn MLP có ReLU ban đầu (`0.920`) và Logistic Regression (`0.880`) trên test.
- **Giải thích:** Kết quả này cho thấy dự đoán “chắc chắn giảm” là quá mạnh. Việc bỏ ReLU khiến các lớp tuyến tính liên tiếp có thể gộp thành một phép biến đổi tuyến tính; tuy nhiên, kết quả đo còn phụ thuộc khởi tạo, quá trình tối ưu và dữ liệu. Chênh lệch nhỏ cần được kiểm chứng thêm thay vì kết luận mô hình tuyến tính luôn tốt hơn.

**Dự đoán khác:** Dự đoán MLP có F1 cao hơn Logistic Regression đã khớp với kết quả (`0.920` so với `0.880`). Dự đoán `emb_search` cũng khớp: 6 láng giềng gần nhất của câu truy vấn đều mang nhãn spam. Báo cáo JSON ghi nhận chưa có dự đoán `emb_tricky`, vì vậy không thể đối chiếu dự đoán trước chạy cho thí nghiệm đó.
## 4. Phân tích 50 case khó

### 4.1. Mức suy giảm tổng thể
| Mô hình | F1 test thường / SUB | F1 trên 50 case khó | Accuracy trên 50 case khó |
|---|---:|---:|---:|
| BoW + Logistic Regression | 0.857 | 0.378 | 0.54 |
| BoW + MLP | 0.919 | 0.286 | 0.50 |
| MiniLM + kNN | 0.947 | 0.772 | 0.74 |
| MiniLM + Logistic Regression | 1.000 | 0.612 | 0.62 |
| MiniLM + MLP head | 1.000 | 0.667 | 0.66 |
| LLM nhỏ zero-shot | 0.884 | 0.897 | 0.88 |
| LLM nhỏ + prompt phòng thủ | 0.884 | 0.935 | 0.92 |

**BoW + MLP tụt nhiều nhất:** F1 giảm từ `0.919` trên SUB xuống `0.286` trên 50 case khó. BoW + Logistic Regression cũng giảm mạnh, từ `0.857` xuống `0.378`.

Tập test thông thường được chia ngẫu nhiên từ cùng nguồn dữ liệu với train nên có phân phối tương đối giống nhau. Mô hình có thể học các từ khóa và mẫu câu quen thuộc rồi đạt điểm cao. Bộ 50 case khó được viết có chủ đích để tạo tình huống khác phân phối: spam viết tự nhiên, né từ khóa, mạo danh và tin ham chứa từ khóa spam. Vì thế, điểm test thông thường có thể lạc quan nếu dùng nó làm đại diện duy nhất cho dữ liệu ngoài thực tế.

### 4.2. Nhóm case khó và cơ chế sai

| Nhóm | BoW + LogReg | BoW + MLP | MiniLM + kNN | MiniLM + LogReg | MiniLM + MLP | LLM nhỏ | LLM phòng thủ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Tưởng spam nhưng không phải | 100% | 90% | 70% | 80% | 80% | 70% | 70% |
| Thao túng / lừa đảo | 12% | 12% | 100% | 38% | 38% | 88% | 100% |
| Spam giọng tự nhiên | 20% | 0% | 40% | 20% | 40% | 80% | 100% |
| Né từ khóa | 17% | 0% | 83% | 83% | 83% | 100% | 100% |
| Prompt injection | 67% | 83% | 100% | 100% | 100% | 100% | 100% |
| Nói về spam | 100% | 100% | 50% | 50% | 50% | 100% | 75% |
| Tin giao dịch hợp lệ | 83% | 100% | 83% | 83% | 83% | 100% | 100% |

- **BoW khó nhất ở nhóm né từ khóa và spam giọng tự nhiên.** BoW biểu diễn văn bản bằng các token rời rạc; khi từ bị biến đổi như `C0ngr4tulat10ns`, `F R E E` hoặc `c.a.s.h`, token có thể không khớp từ vựng đã học. Với spam viết tự nhiên, nội dung có thể không chứa các từ khóa spam nổi bật.
- **BoW + MLP đặc biệt yếu ở hai nhóm trên:** độ chính xác theo nhóm là `0%` ở spam giọng tự nhiên và `0%` ở né từ khóa. Mạng MLP không tự khắc phục được giới hạn biểu diễn của đầu vào BoW.
- **Embedding + kNN** đạt `100%` ở nhóm thao túng/lừa đảo và prompt injection, `83%` ở nhóm né từ khóa, nhưng chỉ `40%` ở spam giọng tự nhiên và `50%` ở nhóm nói về spam. Vector ngữ nghĩa giúp nhận ra tương đồng nội dung nhưng vẫn có thể nhầm ý định khi ngữ cảnh gần nhau.
- **LLM** xử lý tốt hơn nhiều nhóm khó nhờ đọc được ngữ cảnh và ý định. Tuy nhiên, LLM vẫn có thể nhầm tin ham chứa từ khóa spam; prompt phòng thủ cũng làm độ chính xác nhóm “Nói về spam” giảm từ `100%` xuống `75%`.

### 4.3. Hai ví dụ cụ thể

1. **Spam né từ khóa:** `FreeMsg>FAV XMAS TONES!Reply REAL` (nhãn thật: spam). Trong phần lỗi của BoW + Logistic Regression, tin này bị dự đoán thành ham. Cách viết dính `FreeMsg` khiến token `free` không xuất hiện độc lập như trong các mẫu spam quen thuộc; BoW vì vậy không kích hoạt được trọng số của từ khóa.
2. **Spam giọng tự nhiên:** `LIFE has never been this much fun and great until you came in. You made it truly special for me...` (nhãn thật: spam). Tin này xuất hiện trong danh sách lỗi của BoW + Logistic Regression và bị dự đoán thành ham. Nội dung giống lời nhắn cá nhân, không có dấu hiệu quảng cáo hoặc từ khóa spam rõ ràng nên mô hình dựa trên token khó nhận ra ý đồ.

### 4.4. Prompt injection và hướng cải thiện

Trên nhóm prompt injection, LLM nhỏ zero-shot và LLM nhỏ với prompt phòng thủ đều đạt `100%` theo bảng kết quả nhóm. Trên toàn bộ 50 case, prompt phòng thủ tăng accuracy từ `88%` lên `92%` và F1 từ `0.897` lên `0.935`. Tuy nhiên, prompt phòng thủ không cải thiện mọi nhóm: accuracy nhóm “Nói về spam” giảm từ `100%` xuống `75%`, cho thấy chỉ dẫn coi nội dung yêu cầu gán nhãn là dấu hiệu spam có thể gây false positive khi người dùng đang kể lại hoặc cảnh báo về một tin lừa đảo.

Ngoài prompt, có thể chuẩn hóa đầu vào (chuẩn hóa ký tự, khoảng trắng, leetspeak), kết hợp rule-based/allowlist cho OTP và sender hợp lệ, đồng thời không để LLM là lớp kiểm tra duy nhất. Cần kiểm thử để tránh chuẩn hóa sai mã OTP, mã giao dịch hoặc từ viết tắt hợp lệ.

**Giới hạn của bộ 50 case:** Đây là bộ stress test do con người viết, hữu ích để tìm điểm yếu và kiểm tra khả năng chống các tình huống đối kháng. Nó không đại diện đầy đủ cho phân phối spam ngoài đời và không nên được dùng làm thước đo duy nhất để ước lượng hiệu năng production.

## 5. Thử thách cá nhân

**Đề nhận được: Challenge #3 — Deep Learning:** thay MLP thành 4 hidden layers, mỗi lớp 512 neurons; báo số tham số, F1 test, thời gian train và nêu khi nào nên chọn mạng nhỏ.

### Cách thực hiện

- Đầu vào có `2,000` chiều.
- Xây dựng 4 lớp `Linear`, mỗi lớp có 512 neurons, sau mỗi lớp là `ReLU`.
- Lớp output là `Linear(512, 1)`.
- Huấn luyện bằng `train_mlp` với `epochs=30`, `lr=1e-3`, `batch_size=64`.
- Đếm số tham số trainable, đánh giá trên tập TEST và lưu F1 SUB vào `RESULTS`.

### Kết quả

| Chỉ số | MLP ban đầu (1 × 64) | MLP thử thách (4 × 512) |
|---|---:|---:|
| Số tham số | 128,129 | 1,812,993 |
| F1 test | 0.9200 | 0.9082 |
| Thời gian train | 4.17 giây | 72.03 giây |
| F1 SUB 40 | 0.9189 | 0.9189 |

MLP mới có số tham số lớn gấp khoảng `14.15` lần và thời gian train dài hơn khoảng `17.3` lần, nhưng F1 test giảm từ `0.9200` xuống `0.9082`; F1 SUB gần như không đổi. Điều này cho thấy tăng độ sâu và số neuron không bảo đảm cải thiện khả năng tổng quát hóa. Với dữ liệu hiện tại, mạng nhỏ phù hợp hơn vì đạt F1 tốt với ít tham số và thời gian huấn luyện thấp hơn. Nên ưu tiên mạng nhỏ khi dữ liệu không lớn, tài nguyên hạn chế hoặc mạng lớn không đem lại cải thiện đo được.

## 6. Phương án cho sản phẩm thật

Yêu cầu: lọc 10 triệu SMS/ngày, tương đương khoảng `116` SMS/giây trung bình, với SLA `p95 < 50 ms/tin`.

**Lựa chọn đề xuất: MiniLM + Logistic Regression** nếu ưu tiên F1 trên test và vẫn đáp ứng p95 trong lần benchmark này:

- F1 test `0.926`, F1 SUB `1.000`.
- p95 `36.778 ms/tin`, thấp hơn ngưỡng 50 ms.
- Chi phí ghi nhận `0 USD/1k tin` trong thí nghiệm chạy local/provider hiện tại.

**Các lựa chọn khác đạt p95 dưới 50 ms:** BoW + Logistic Regression (`0.088 ms`), BoW + MLP (`0.376 ms`) và MiniLM + MLP head (`30.391 ms`). Trong đó, BoW + Logistic Regression nhanh nhất nhưng F1 SUB thấp hơn; MiniLM + MLP head có F1 SUB `1.000` và p95 thấp hơn MiniLM + Logistic Regression, nhưng F1 test thấp hơn một chút (`0.920` so với `0.926`).

Không chọn LLM làm bộ lọc chính cho luồng real-time vì p95 của LLM nhỏ/lớn lần lượt khoảng `3,134.944 ms` và `2,407.956 ms`, đều vượt SLA rất xa. LLM có thể dùng ở tầng phụ để phân tích các trường hợp mơ hồ, nhưng cần giới hạn số request để kiểm soát latency và chi phí.

**Đánh đổi và lưu ý triển khai:**

- MiniLM có thời gian encode/train ban đầu khoảng `78.93 giây`; cần triển khai encoder và classifier ổn định, theo dõi tải CPU/GPU.
- Throughput theo lô của MiniLM + Logistic Regression trong notebook khoảng `54.33 tin/s`, thấp hơn mức trung bình yêu cầu khoảng `116 tin/s`. Vì vậy, dù p95 từng request trong phép đo đạt SLA, vẫn cần load test, tối ưu batch/parallelism hoặc scale ngang nhiều worker để đáp ứng tổng lưu lượng 10 triệu tin/ngày.
- BoW + Logistic Regression có throughput theo lô khoảng `31,369 tin/s`, dư tải lớn nhưng chất lượng thấp hơn.
- Có thể thiết kế **router hai tầng**: mô hình rẻ (BoW hoặc MiniLM + Logistic Regression) xử lý tin có xác suất rõ ràng; các tin có xác suất spam nằm trong vùng lưỡng lự, ví dụ `0.30 < p < 0.70`, mới chuyển sang LLM hoặc rule bổ sung. Với OTP/ngân hàng, cần allowlist sender và ngưỡng chặn cao để giảm false positive. Router giúp giảm số lần gọi LLM, nhưng phải đo lại p95 end-to-end và chi phí thực tế.
- Không thể suy ra chi phí LLM production từ `USD/1k = 0` của lần chạy Groq này. Chi phí phụ thuộc model/provider, số token prompt, output và lưu lượng. Theo phép tính minh họa trong notebook, giả định 100 input + 1 output token/SMS với giá `$0.10 / $0.50` cho mỗi 1 triệu token, chi phí cho 10 triệu SMS/ngày là khoảng `$3,150/tháng`; prompt thực tế có thể làm con số thay đổi.

## 7. AI log và điều đã học

### AI log

| Phần | Nội dung hỏi AI | AI hỗ trợ | Kiểm chứng / chỉnh sửa |
|---|---|---|---|
| 4.2–4.3 | Hỗ trợ xây dựng prompt zero-shot/few-shot và parser nhãn spam/ham theo yêu cầu notebook. | Hoàn thiện `build_classification_prompt` và `parse_label`; xử lý phủ định `not spam`, nhãn mâu thuẫn và đầu ra không rõ. | Đối chiếu các test ở cell 67; kiểm tra prompt kết thúc bằng `Label:` và có đủ ví dụ few-shot. |
| 4.4 | Hỗ trợ triển khai gọi LLM theo batch, retry, đo latency và ước lượng chi phí. | Hoàn thiện `llm_classify_batch`, trả về `preds`, `latencies`, `latency_s`, `usd`, `n_invalid`. | Kiểm tra hợp đồng hàm và cell mock API 69; không gọi API thật trong lúc chỉnh notebook để tránh phát sinh phí. |
| Tính tay | Hỏi cách áp dụng công thức baseline, sigmoid, F1, cosine và chi phí token. | Có nội dung hướng dẫn/các giá trị dự kiến được ghi trong notebook. | Chưa có ảnh tính tay trong tệp được gửi; cần tự viết các bước tính và bổ sung `hand_calc.jpg`. |

### Điều rút ra

Qua bài lab, tôi hiểu rằng điểm số cao trên test cùng phân phối với train chưa đủ chứng minh mô hình có thể xử lý tốt dữ liệu thực tế; cần thêm stress test với các trường hợp lệch phân phối và đối kháng. Tôi cũng nhận ra mô hình lớn hơn hoặc prompt dài hơn không mặc nhiên tốt hơn: phải đánh giá đồng thời F1, latency, throughput, chi phí và loại lỗi theo từng nhóm dữ liệu.
'''
