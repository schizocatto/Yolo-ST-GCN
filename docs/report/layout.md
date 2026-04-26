# [Tên Đồ Án Của Bạn - Ví dụ: Nhận Dạng Động Tác Thể Dục Dụng Cụ Với Mạng Tích Chập Đồ Thị Không-Thời Gian Bằng Khung Xương]

**Thông tin nhóm sinh viên thực hiện:**
- [cite_start]Họ và tên 1 - MSSV 1 - Email 1 - SĐT 1 [cite: 53]
- [cite_start]Họ và tên 2 - MSSV 2 - Email 2 - SĐT 2 [cite: 53]

---

**Abstract + Keywords**
> *Chú thích:* Tóm tắt ngắn gọn 150-200 chữ: Bài toán là gì? Dùng mô hình gì (ST-GCN 2 luồng)? Dataset gì (Gym99)? Kết quả đạt được ra sao? (Điền con số Acc tốt nhất sau khi đã fix xong thực nghiệm).

## 1. Giới Thiệu
> *Chú thích:* Nêu bối cảnh bài toán nhận dạng hành động hạt nhỏ (fine-grained). Tại sao lại chọn thể dục dụng cụ (thách thức về sự tương đồng giữa các động tác). Giới thiệu tóm tắt cấu trúc bài báo.

## 2. Các Nghiên Cứu Liên Quan
> *Chú thích:* Phần này bạn làm theo đúng Yêu cầu 1 của đề đồ án (Khảo sát nghiên cứu).
  2.1 Nhận Dạng Hành Động Dựa Trên Khung Xương                                                                                                           
  2.2 Mạng Tích Chập Đồ Thị                                                                                                                              
  2.3 Mô Hình Hai Luồng Cho Khung Xương                                                                                                                  
  2.4 Nhận Dạng Hành Động Hạt Nhỏ Trong Thể Thao 

## 3. Tập Dữ Liệu Gym99
> *Chú thích:* Trình bày rõ ràng về dữ liệu để người đọc thấy được độ khó (mất cân bằng, số lượng mẫu ít).
  3.1 Xây Dựng Từ FineGym288                                                                                                                             
  3.2 Biểu Diễn Khung Xương                                                                                                                              
  3.3 Phân Tích Tập Con FX (Floor Exercise)

## 4. Kiến Trúc Mô Hình Và Các Đề Xuất Cải Tiến
> *Chú thích:* Đây là phần ăn điểm cho Yêu cầu 2 (Đề xuất mô hình). Tách rõ ranh giới: từ 4.1 đến 4.6 là mô hình gốc, từ 4.7 trở đi là các cải tiến của bạn để đối phó với dữ liệu Gym99.
  **[A] Kiến trúc cơ sở (Baseline ST-GCN)**
  4.1 Đồ Thị Không-Thời Gian                                                                                                                             
  4.2 Tích Chập Đồ Thị Không Gian                                                                                                                        
  4.3 Chiến Lược Phân Vùng                                                                                                                               
      - Uni-labeling (K=1)                                                                                                                               
      - Distance Partitioning (K=2)                                                                                                                      
      - Spatial Configuration Partitioning (K=3)                                                                                                         
  4.4 Tích Chập Thời Gian                                                                                                                                
  4.5 Trọng Số Tầm Quan Trọng Cạnh Có Thể Học                                                                                                            
  4.6 Kiến Trúc Mạng Gốc
  
  **[B] Đề xuất cải tiến cho bộ dữ liệu Gym99**
  4.7 Kiến Trúc Hai Luồng (Two-Stream)
  > *Chú thích:* Trình bày lý do kết hợp Joint và Bone. Cấu trúc ghép nối điểm số (Score fusion) như thế nào.
  4.8 Các kỹ thuật tối ưu và chống Overfitting
  > *Chú thích:* Nêu các phương pháp bạn đã dùng: Cấu hình Focal Loss (giải quyết mất cân bằng lớp), Warm-up learning rate, và Dropout. Giải thích ngắn gọn tại sao lại áp dụng chúng.

## 5. Thực Nghiệm Và Phân Tích Kết Quả
> *Chú thích:* Đây là phần quan trọng nhất (Yêu cầu 3 & 4). Phải có BẢNG và BIỂU ĐỒ.
  5.1 Thiết Lập Huấn Luyện
  > *Chú thích:* Ghi rõ phần cứng (Colab/Kaggle GPU gì), tham số huấn luyện (batch size, epochs, learning rate, optimizer).
  
  5.2 Đánh Giá Mô Hình Đề Xuất (Ablation Study)
  > *Chú thích:* Phải có 1 bảng so sánh độ chính xác (Accuracy). Ví dụ:
  > - Dòng 1: Baseline 1 luồng (89%)
  > - Dòng 2: Baseline + 2 Streams
  > - Dòng 3: 2 Streams + Focal Loss
  > - Dòng 4: Mô hình hoàn chỉnh (thêm Dropout/Warmup)
  > Sau đó viết 1-2 đoạn văn nhận xét bảng này. Thấy rõ cái nào giúp tăng điểm nhiều nhất.
  
  5.3 Phân Tích Quá Trình Huấn Luyện (Training Analysis)
  > *Chú thích:* Chèn biểu đồ Loss và Accuracy của tập Train và tập Val qua các epoch. Phân tích rõ hiện tượng Overfitting ở các epoch nào (khi Train Loss tiếp tục giảm nhưng Val Loss đi ngang hoặc tăng). Các kỹ thuật ở phần 4.8 đã giúp giảm tình trạng này ra sao (nếu có).
  
  5.4 Phân Tích Lỗi (Error Analysis / Confusion Matrix)
  > *Chú thích:* Chèn Ma trận nhầm lẫn (Confusion Matrix). Bạn không cần vẽ cho cả 99 lớp (vì quá to), hãy chọn ra top 5 hoặc top 10 lớp hay bị mô hình dự đoán nhầm với nhau nhất. Phân tích tại sao (ví dụ: động tác nhảy lộn vòng tiến và lùi quá giống nhau về quỹ đạo khung xương).

## 6. Kết Luận
> *Chú thích:* Tóm tắt lại những gì đã làm được. Nêu ra những hạn chế hiện tại (ví dụ: mô hình vẫn còn hơi overfitting ở các lớp thiểu số) và hướng phát triển nếu có thêm thời gian.

## 7. Tài Liệu Tham Khảo
> *Chú thích:* Bắt buộc phải có và tuân thủ định dạng IEEE (VD: [1] Tên tác giả, "Tên bài báo," Tên tạp chí/hội nghị, năm).