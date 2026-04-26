Ngoài: penn-yolo-keypoint-error.jpg so sánh khớp yolo và penn trên cùng 1 ảnh, yolo bị lệch chân.

# Chart Reference

Tổng quan ý nghĩa từng biểu đồ được trích xuất từ hai notebook.

---

## Thư mục: `stgcn_notebook/`
*(Notebook: `stgcn_notebook_with_viz-2.ipynb` — Huấn luyện ST-GCN trên Penn Action Dataset với Ground Truth keypoints)*

| Tên file | Ý nghĩa |
|---|---|
| `action_class_distribution.png` | Bar chart phân phối **toàn bộ 15 lớp hành động** trong Penn Action Dataset (2326 video). Cho thấy sự mất cân bằng dữ liệu giữa các class. |
| `data_stats_and_length_distribution.png` | Hai biểu đồ: (trái) số lượng video theo từng lớp trong nhóm **8 bài tập** được chọn (1163 video); (phải) phân phối độ dài video thô trước khi temporal alignment (min=18, max=663, mean≈83 frames). |
| `sample_video_frames.png` | Lưới ảnh minh họa một số frames thô từ một video ngẫu nhiên trong dataset. |
| `gt_skeleton_overlay.png` | Frame video với **khung xương Ground Truth** (13 joints từ file `.mat`) vẽ đè lên — minh họa chất lượng annotation gốc của Penn Action. |
| `sample_skeleton_gt.png` | Khung xương mẫu dạng đồ thị (stick figure) từ Ground Truth keypoints, hiển thị 14 nodes (13 joints + 1 virtual center). |
| `graph_skeleton_partitions.png` | Đồ thị khung xương 14 nodes tô màu theo **3 loại partition** của ST-GCN: self-loop (xanh lá), centripetal — hướng về trung tâm (cam), centrifugal — hướng ra ngoài (đỏ). Minh họa cấu trúc đồ thị spatial của mô hình. |
| `adjacency_matrix_heatmaps.png` | Ba heatmap của **ma trận kề A[0], A[1], A[2]** sau chuẩn hóa — tương ứng với 3 partition: self-loop, centripetal, centrifugal. Kích thước mỗi ma trận: 14×14. |
| `model_architecture_table.png` | Bảng mô tả **kiến trúc ST-GCN**: các layer block, số channels (64→64→128→128→256→256), kích thước tensor output theo từng tầng, và tổng số tham số. |
| `training_curves.png` | Ba đường cong huấn luyện qua **80 epochs**: (trái) Train/Val Loss; (giữa) Train/Val Accuracy; (phải) Val Macro F1. Best val accuracy ≈ **99.14%** (epoch 45), best F1 ≈ **0.9890**. |
| `confusion_matrix_gt.png` | **Confusion Matrix** trên tập val (233 videos) khi dùng Ground Truth keypoints. Accuracy tổng thể: **98%**. Hầu hết class đạt F1 ≥ 0.97. |
| `per_class_f1_gt.png` | Bar chart **F1-score theo từng class** trên val set với GT keypoints. Các class như `jumping_jacks`, `pullup` đạt F1 = 1.00; thấp nhất là `bench_press` (0.98). |

---

## Thư mục: `yolo_stgcn_notebook/`
*(Notebook: `yolo_stgcn_inference_progress_bar_run_completed.ipynb` — Pipeline đầu cuối YOLOv8-pose + ST-GCN)*

| Tên file | Ý nghĩa |
|---|---|
| `yolo_skeleton_demo.png` | Demo pipeline trên 1 video: (trái) frame với **khung xương YOLO** vẽ đè (17 COCO → 13 Penn → 14 nodes, điểm vàng = virtual center); (phải) **confidence bar chart** của ST-GCN cho từng class. Video demo: `0341`, GT = `bench_press`, Pred = `bench_press` ✓. |
| `yolo_keypoint_quality.png` | Bar chart **sai số keypoint YOLO** so với Ground Truth, chuẩn hóa theo chiều cao người, trên 1158 video. Overall mean error = **3.37** (đơn vị: tỷ lệ chiều cao). Lỗi cao nhất ở `l_ankle`/`r_ankle` (~4.88), thấp nhất ở `head` (~2.70). Màu: xanh < 0.10 (tốt), cam 0.10–0.20 (trung bình), đỏ > 0.20 (kém). |
| `pipeline_confusion_matrix.png` | **Confusion Matrix** của pipeline YOLO + ST-GCN trên 233 videos val set. Accuracy = **73.82%**, Macro F1 = **72.39%**. Tệ nhất: `bench_press` (F1=0.38, chỉ recall 25%); tốt nhất: `jumping_jacks` (F1=0.94), `pullup` (F1=0.93). |

---

---

## Thư mục gốc `figures/` — EDA tập FX của Gym99
*(Notebook: `FX_train_eda.ipynb` — phân tích tập huấn luyện FX apparatus)*

| Tên file | Ý nghĩa |
|---|---|
| `FX_train_class_distribution.png` | Ba biểu đồ phân phối lớp tập huấn luyện FX: (trái) bar chart kích thước lớp sắp xếp giảm dần (train xanh + val cam), đường ngang = mean 235 mẫu; (giữa) histogram số mẫu/lớp; (phải) đường cong Lorenz (Gini = 0,306). 35 lớp, train=5824, val=2411. |
| `FX_train_split_quality.png` | Hai biểu đồ chất lượng chia train/val: (trái) histogram tỉ lệ train trên tổng số mẫu mỗi lớp (mean ≈ 0.71); (phải) bar chart train vs val song song theo từng lớp. Xác nhận 0 lớp không có mẫu val (split sạch). |
| `FX_train_frame_count_histogram.png` | Phân phối số frame thô của video FX train và val (hai subplot song song). Đường đỏ nét đứt = TARGET 64 frame; đường đen chấm = median. Cho thấy phần lớn video ngắn hơn target. |
| `FX_train_per_class_frame_count.png` | Bar chart độ dài frame trung bình (± std) theo từng lớp trong tập train FX, sắp xếp giảm dần. Đường đỏ = TARGET 64 frame. Cho thấy sự đa dạng đáng kể về độ dài clip giữa các lớp. |
| `FX_train_per_joint_heatmap.png` | Lưới 3×6 heatmap 2D cho 17 khớp COCO trên tập train FX (5824 mẫu). Mỗi ô là histogram 2D tọa độ pixel (trục Y đảo ngược). Cho thấy vận động viên tập trung ở trung tâm khung hình (X≈1088, Y≈488 px trên ảnh 1920×1080). |
| `FX_train_skeleton_bbox_distribution.png` | Phân phối chiều rộng và chiều cao bounding-box khung xương trên ảnh gốc 1920×1080. Chiều rộng ≈ 200 px, chiều cao ≈ 400 px (trung bình). |
| `FX_train_per_joint_velocity.png` | Bar chart vận tốc tuyệt đối trung bình (pixel/frame) cho 18 khớp COCO-18 trên toàn bộ train FX. Đỏ = Q3+ (hoạt động nhất); xanh dương = Q1- (ít hoạt động nhất). Các khớp chi (cổ tay, mắt cá chân) hoạt động nhiều hơn khớp trục thân (hông). |
| `FX_train_motion_analysis.png` | Ba subplot phân tích chuyển động: (trái) histogram tỉ lệ keypoint bằng 0 mỗi mẫu; (giữa) histogram displacement trung bình, phân tách mẫu valid vs corrupt; (phải) scatter plot zero-fraction vs displacement. |
| `FX_train_most_least_dynamic_classes.png` | Hai bar chart ngang: (trái) top-10 lớp năng động nhất; (phải) top-10 lớp ít năng động nhất, tính trung bình displacement trên mẫu valid. |
| `FX_train_pca.png` | Scatter PCA 2D trên đặc trưng trung bình thời gian (36 chiều, 20 lớp phổ biến nhất FX train). Sự chồng chéo cao giữa các lớp xác nhận tính fine-grained của tập dữ liệu. |
| `FX_train_tsne.png` | Scatter t-SNE 2D (20 lớp phổ biến nhất FX train, perplexity=30, 1000 iter). Tương tự PCA, ít cụm rõ ràng — phản ánh độ khó phân tách lớp. |
| `FX_train_skeleton_grid.png` | Lưới khung xương mid-frame cho 12 lớp × 3 mẫu ngẫu nhiên (tập train FX). Mỗi ô hiển thị stick figure tại frame giữa của chuỗi. |
| `FX_train_frame_strips.png` | Bốn dải frame: mỗi dải là một mẫu ngẫu nhiên từ train FX, hiển thị 8 frame cách đều nhau. Màu gradient (plasma) theo thứ tự thời gian. |

*(Notebook: `FX_val_eda.ipynb` — phân tích tập kiểm tra FX apparatus)*

| Tên file | Ý nghĩa |
|---|---|
| `FX_val_class_distribution.png` | Ba biểu đồ tập val FX: (trái) val class size giảm dần; (giữa) tỉ lệ val/train mỗi lớp; (phải) đường Lorenz val (cam) vs train (xanh) — Gini val = 0,306. Min val/class = 24, max = 183. |
| `FX_val_split_quality.png` | Hai biểu đồ: (trái) histogram tỉ lệ val trên tổng mỗi lớp (mean ≈ 0.29); (phải) train vs val bar chart song song. Kiểm tra leakage: ✓ sạch (0 video_id trùng). |
| `FX_val_frame_count_histogram.png` | Phân phối số frame thô của val và train FX (hai subplot riêng). Đường đỏ = TARGET 64. |
| `FX_val_frame_count_train_val_overlay.png` | Histogram phân phối frame count train (xanh) và val (cam) chồng lên nhau — xác nhận phân phối tương tự nhau giữa hai split. |
| `FX_val_per_joint_heatmap.png` | Lưới heatmap 2D cho 17 khớp COCO trên tập val FX (2411 mẫu). Phân bố tương tự tập train. |
| `FX_val_skeleton_bbox_distribution.png` | Phân phối bbox width/height trên tập val FX. So sánh với tập train để xác nhận tính nhất quán. |
| `FX_val_per_joint_velocity.png` | Vận tốc trung bình theo khớp trên tập val FX. Mẫu hoạt động tương tự train — chi trên/dưới năng động hơn trục thân. |
| `FX_val_motion_analysis.png` | Ba subplot phân tích chuyển động trên val FX: zero-fraction, displacement histogram, scatter. |
| `FX_val_most_least_dynamic_classes.png` | Top-10 lớp năng động nhất/ít nhất trong val FX (trên mẫu valid). |
| `FX_val_pca.png` | PCA 2D trên val FX (20 lớp phổ biến nhất). |
| `FX_val_tsne.png` | t-SNE 2D trên val FX. |
| `FX_val_skeleton_grid.png` | Lưới khung xương mid-frame cho 12 lớp × 3 mẫu val FX (màu cam). |
| `FX_val_frame_strips.png` | Bốn dải frame từ mẫu val FX ngẫu nhiên. |

---

## Thư mục: `FX/FX-experiment-result/`
*(Notebooks: `Baseline-result.ipynb`, `Ver2-result.ipynb` — Kết quả huấn luyện Expert FX trên Gym99)*

| Tên file | Ý nghĩa |
|---|---|
| `FX_baseline_confusion_matrix.png` | Confusion Matrix của **FX Baseline Expert** trên tập val FX. Hiển thị hiệu suất phân loại 35 lớp FX apparatus với keypoints GT. |
| `FX_baseline_training_curves.png` | Đường cong huấn luyện (Train/Val Loss, Accuracy, F1) của **FX Baseline Expert** theo từng epoch, đọc từ `history.json`. |
| `FX_ver2_confusion_matrix.png` | Confusion Matrix của **FX Ver2 Expert** trên tập val FX. So sánh với Baseline để đánh giá cải tiến của phiên bản 2. |
| `FX_ver2_training_curves.png` | Đường cong huấn luyện của **FX Ver2 Expert** theo từng epoch, đọc từ `history.json`. |

---

## Tóm tắt so sánh

| Metric | ST-GCN + GT Keypoints | YOLO + ST-GCN Pipeline |
|---|---|---|
| Accuracy | **98.28%** | 73.82% |
| Macro F1 | **0.9814** | 0.7239 |
| Nguồn keypoints | Ground Truth `.mat` | YOLOv8n-pose (auto) |

Khoảng cách ~24% accuracy phản ánh sai số lớn của YOLO keypoints (mean normalized error = 3.37) — đặc biệt ở các khớp chi dưới và vai — so với Ground Truth của Penn Action Dataset.
