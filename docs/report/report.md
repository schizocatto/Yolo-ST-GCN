# [Tên Đồ Án Của Bạn - Ví dụ: Nhận Dạng Động Tác Thể Dục Dụng Cụ Với Mạng Tích Chập Đồ Thị Không-Thời Gian Bằng Khung Xương]

**Thông tin nhóm sinh viên thực hiện:**
- Họ và tên 1 - MSSV 1 - Email 1 - SĐT 1 
- Họ và tên 2 - MSSV 2 - Email 2 - SĐT 2 

---

**Abstract**
Bài báo này trình bày ứng dụng kiến trúc biến thể của Mạng Tích Chập Đồ Thị Không-Thời Gian (ST-GCN) cho bài toán nhận dạng hành động thể dục dụng cụ hạt nhỏ trên tập dữ liệu Gym99 (cụ thể tập trung phân tích trên tập con Floor Exercise - FX). Thay vì xử lý trực tiếp trên không gian ảnh điểm như thông thường, khung xương được biểu diễn cấu trúc hóa dưới dạng đồ thị không-thời gian với 18 node (17 khớp theo định nghĩa giải phẫu COCO và 1 node trung tâm ảo) và được học hỏi đặc trưng bởi kiến trúc tổng hợp hai luồng (Joint stream và Bone stream) bổ trợ thông tin cho nhau, kết hợp cùng các chiến lược phân vùng không gian tinh vi đồ thị. Khác biệt với mô hình cơ sở lý thuyết (baseline) có độ sâu 10 block, trong cấu hình mạng hai luồng, chúng tôi đang thử nghiệm các kiến trúc lớp được tinh giản mạnh mẽ với độ sâu lần lượt chỉ gồm 8, 6 và 4 block. Chiến lược cắt giảm độ sâu này đồng thời kết hợp với hàm tối ưu Focal Loss, bộ lập lịch Linear Warm-up, cùng Gradient Clipping nhằm tập trung giải quyết bài toán chống hiện tượng Overfitting do mất cân bằng dữ liệu cực đa dạng của Gym99 mang lại.

**Keywords:** nhận dạng hành động hạt nhỏ, khung xương, ST-GCN, FineGym, Gym99, tích chập đồ thị, hai luồng.

## 1. Giới Thiệu
Nhận dạng hành động người (Human Action Recognition – HAR) là một trong những bài toán cốt lõi của thị giác máy tính đương đại. Gần đây, dòng chảy nghiên cứu đang có sự dịch chuyển mạnh từ nhận dạng ở cấp độ *thô* (coarse-grained) như phân biệt khái niệm "chạy" với "đứng", sang nhận dạng ở cấp độ *hạt nhỏ* (fine-grained), tạo ra yêu cầu phân loại những biến thể hành động cực kỳ tinh tế trong cùng một họ. Trong bối cảnh đó, bộ môn Thể dục dụng cụ tạo ra một miền bao phủ các thách thức: ví dụ, các động tác như "lăn ra trước" và "lăn ra sau" có cấu trúc các khớp xương vận động trong không gian gần như đối xứng, sự khác biệt mấu chốt để phân loại lúc này chỉ còn nằm ở thứ tự sinh ra và pha thời gian chênh lệch của các khớp.

Hầu hết các phương pháp HAR truyền thống nếu được phát triển dựa trên đặc trưng RGB sẽ tỏ ra nhạy cảm với điều kiện nhiễu ngoại cảnh (chất lượng ánh sáng, trang phục tuyển thủ, góc độ thay đổi của camera). Phương pháp trích xuất đặc trưng thuần tuý dựa trên cấu hình bộ không gian khung xương đã trực tiếp khắc phục được yếu điểm này nhờ việc mô hình hoán cơ thể người về một đồ thị hình học cấu trúc rõ ràng. Mô hình ST-GCN (Spatial Temporal Graph Convolutional Networks) là khung tiêu chuẩn tiên phong áp dụng tích chập đồ thị lên các đoạn chuỗi bộ khung xương biến thiên tự do, học đồng thời tổng quan cấu trúc không gian hình học và tiến hóa thời gian cơ học của con người.

Tập dữ liệu Gym99 nằm trong dự án FineGym với 99 loại hành động hạt nhỏ. Điểm đặc trưng lớn ở cấu hình dữ liệu là sự thất thoát phân bố với hiện trạng mất cân bằng lớp quá mức và độ dài các mô đun chuỗi frames trích không đồng đều và ngắt quãng tạo thách thức phi tầm thường lúc phân loại.

## 2. Các Nghiên Cứu Liên Quan

### 2.1 Nhận Dạng Hành Động Dựa Trên Khung Xương
Hướng tiếp cận giải quyết trực tiếp trên bề mặt khung xương thực chất là khai thác tọa độ các khớp cốt lõi như một biểu diễn đặc trưng nén của chuyển động. Mô hình trước đây thường tính các góc xoay, hiệp phương sai tọa độ hoặc sử dụng công thức làm phẳng tọa độ vào các mạng tuần hoàn (RNN/LSTM) hay dùng mạng tích chập 1D trên mảng tọa độ chuỗi (TCN). Sự kiện ST-GCN ra đời đã tạo một bước ngoặt lớn về ý tưởng bằng cách biểu diễn vẹn nguyên khung xương thành một cấu phần đồ thị không-thời gian thay vì làm phẳng.

### 2.2 Mạng Tích Chập Đồ Thị
Cốt lõi toán học của ST-GCN là đồ thị (GCN) được giải quyết theo hướng *không gian* (spatial). Khác với *phổ* tính biến đổi fourier, cách của ST-GCN tích chập thẳng trên hàng xóm không gian liền kề của tập điểm biên bằng cánh chia chúng ra thành nhiều vùng đa lớp, mỗi lớp nắm vai trò tạo khối cho mô hình và sở hữu một ma trận trọng số tối ưu.

### 2.3 Mô Hình Hai Luồng Cho Khung Xương
Mọi thuật toán hoạt động trên khung xương sẽ suy kiệt thông tin nếu chỉ chăm chăm dùng tọa độ khớp (Joint). Việc học qua cấu trúc đường biên (vector xương trỏ nối giữ 2 điểm gọi là Bone) giúp nhận diện rõ sự co giãn độ dài và các tư thế mang tính tương đối cho kết cấu cục bộ. Sự song song kết hợp này (score fusion) tại layer đầu ra mang lại chỉ số Acc tăng mạnh mẽ.

### 2.4 Nhận Dạng Hành Động Hạt Nhỏ Trong Thể Thao
Với tính chất FineGym tổng quan và Gym99 làm chuẩn mực, sự tương đồng vô biên ở hình khối do bản tính của các bài khởi động cùng pha là thử thách khắc nghiệt đòi hỏi không phải các mô hình trích đặc trưng chung chung, mà là sự lưu tâm trên mọi khớp biên không gian-thời gian vi mô kết xuất ra để đào sâu được khác biệt.

## 3. Tập Dữ Liệu Gym99

### 3.1 Xây Dựng Từ FineGym288
Bộ cấu hình Gym99 trên thực tế được chúng tôi ánh xạ qua lại, gạn lọc ra từ 288 lớp chuyên nghiệp của bản FineGym. Gym99 bao phủ tận 4 dụng cụ thể luyện nhưng ta chủ ý phân tích vào cấu phần nhức nhối là hệ bài tập Floor Exercise (FX) chiếm đa số gồm 35 lớp (được định chuẩn với cụm nhãn $[6,40]$). 

### 3.2 Biểu Diễn Khung Xương
Đồ thị được tổng quát theo biểu diễn tiêu chuẩn COCO-17 cho video kích thước thật $1920 \times 1080$. Việc mô phỏng chuyển động sẽ khiêm tốn nếu thiếu vùng không định hình hướng tâm, thế nên một **node trung tâm ảo** (hay còn gọi node 18) được chủ mưu thêm vào dựa trên cơ sở lấy trung độ của tam giác/tứ giác đỉnh góc vai và đỉnh hông. Từ đây, vạn mô hình đều nhận input bằng tensor 5D chuyên biệt: `(N, C=2, T=48, V=18, M=1)`. Các khớp khi đưa qua luồng Bone sẽ được làm vector hiệu số giữ node lá và nhãn đầu rễ của chi cạnh. 

Quá trình huấn luyện không thể không chuẩn hóa bounding-box của đối tượng về khoảng vị trị $[0,1]$ trên toàn video kèm theo chiến lược Center Normalize đẩy tọa độ trượt quanh gốc phân khung.

### 3.3 Phân Tích Tập Con FX (Floor Exercise)
Tập FX trên sự bố trí lấy theo nhãn train và val mang trong mình tính Gini thấp hơn (0.306 so với 0.438) hệ tập 99 nhưng sự nhượng bộ là tỉ lệ lệch hạng vẫn cực xáo trộn với tối đa là biến thiên $8\times$. Ma trận biểu diễn PCA đa chiều phân vùng ra mớ hỗn độn trồng lấn giữa các hạng mục (động tác thực thể tự do dường như không sai số nhiều về sự tương quan tĩnh).

## 4. Kiến Trúc Mô Hình Và Các Đề Xuất Cải Tiến

**[A] Kiến trúc cơ sở (Baseline ST-GCN)**

### 4.1 Đồ Thị Không-Thời Gian
Thiết kế $N=18$ khớp xuyên xuốt $T$ frame mô hình $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ cấu hình không gian với tập $\mathcal{E}_S$ giải phẫu cơ địa 19 cạnh kết cấu nội bộ rễ nhánh, kết hợp với các nối khung $\mathcal{E}_F$ nối quỹ đạo định hạn của 1 tọa độ dọc qua vô tận frame chuyển dịch liên tục.

### 4.2 Tích Chập Đồ Thị Không Gian
Mô hình GCN thay thế Convolution 2D trên mảng pixel bằng toán tử tích lũy dựa vào hàm mẫu tập lân cận (không gian quanh khớp) kèm ánh xạ biên cho tham số học chuẩn hóa và nhận nhiễu ngẫu nhiên học được để phỏng đoán chuyển biến.

### 4.3 Chiến Lược Phân Vùng
Thiết lập bộ 3 chiến lược:
- **Uni-labeling ($K=1$)**: Kết hợp lan truyền hỗn mang nguyên khối vào trong bộ tham số.
- **Distance Partitioning ($K=2$)**: Chia nhánh lân cận đơn giản để phân biệt giữa trung tâm và ngoài vi.
- **Spatial Configuration Partitioning ($K=3$)**: Sự phân luồng trọng tâm khi mô hình gán nhãn hàng xóm theo ba phân loại cốt lỏi: tự lặp, nút hướng tâm (tịnh tiến nội lực cơ thể) và hướng ly tâm. Ánh xạ này cho thấy các quy đồng về giải phẫu với tính năng nắm rõ khối cơ được phân mảnh trên sinh thái.

### 4.4 Tích Chập Thời Gian
Nút GCN đồ thị liên đới khung TCN (kernel $9 \times 1$) trên mảng $\Gamma$ quét phạm vi cục bộ $\pm 4$ frame ứng với $\approx 0.28$ giây trên thực tế.

### 4.5 Trọng Số Tầm Quan Trọng Cạnh Có Thể Học
Biến đối một ma trận mặt nạ trọng số $\mathbf{M}$ tích Hadamard vào biên giúp STGCN phân công tự động mức quan tâm, giải phóng độ chú tâm cứng nhắc ban sơ theo kiến trúc xương quy ước tự nhiên nhằm đánh bóng các hành động dựa vào chân nhiều hơn khi đang nhào lộn hay bấu víu dụng cụ tay không đều tạo sự tự chủ.

### 4.6 Kiến Trúc Mạng Gốc
Phiên bản Baseline chuẩn thiết kế lồng 9 cụm block lặp nối theo Res, mỗi khối là chồng chất của Convolution đồ thị không gian kèm theo lớp kích hoạt thời gian. Tầng xuất thông tin là Global Average Pooling ép khung qua phép tính dự phóng $1 \times 1$ tuyến tính cuối cùng cho bộ vector số hóa. 

**[B] Đề xuất cải tiến cho bộ dữ liệu Gym99**

### 4.7 Kiến Trúc Hai Luồng (Two-Stream)
Vì đặc tả Baseline chỉ làm việc trên dữ liệu Joint gốc, việc bổ sung khối tiếp nhận dữ liệu nhánh theo định nghĩa luồng Bone là tối quan trọng nhằm triệt bỏ sự nhút nhát của STGCN trước góc khuất không gian mở rộng biên độ. Các đồ thị luồng ghép sẽ trỏ hai input riêng, và đi qua siêu mô hình STGCN cùng cấp. Giá trị xuất ở output cho từng class là tổng bằng phép cộng vector logit Late Fusion ($\mathbf{s}_{\text{final}} = \mathbf{s}^{(j)} + \mathbf{s}^{(b)}$) gia cường tính rõ ràng cho kết luận chẩn đoán.

### 4.8 Các kỹ thuật tối ưu và chống Overfitting
Do giới hạn sự mất cân bằng và độ đặc thù lượng dữ liệu hạn hẹp trên các lớp nhỏ ở tệp FX Gym99, chúng tôi trình bày các chiến lược can thiệp vào quá trình đào tạo mô hình như sau:
- **Rút Gọn Cấu Trúc (Reduced Model Depth)**: Mạng baseline (10 layers block) có số tham số cực lớn và rủi ro Memorization (Học Vẹt) rất cao khi dung nạp Gym99. Trong kiến trúc hai luồng song song, chúng tôi cố ý định hình lại tham số ở mức thu hẹp, trực tiếp giới hạn thiết lập của mô hình về độ sâu siêu tinh gọn với 8, 6, và 4 tầng tích chập block. Sự cắt gọn này làm giảm đáng kể áp lực bộ nhớ và tăng cường tính tổng quát (Generalization).
- **Focal Loss Có Trọng Số Mượt (Alpha Smoothing)**: Quá trình hồi quy và đánh sai lệch thực tế được tối ưu hóa thông qua cơ chế hội tụ Focal Loss ($\gamma = 2.0$), tích hợp nhân đôi sức mạnh bằng hệ số $\alpha$ nghịch đảo chuẩn bình phương chống xói mòn gradient dành cho các class thiểu số.
- **Warm-up Learning Rate và Gradient Clipping**: Trạng thái bùng nổ mất mát có thể bẻ cong quỹ đạo lan truyền khi phân bố dữ liệu quá tản mạn. Mô hình được khởi xướng với nhịp Linear Warm-up LR nhẹ nhàng ở những Epoch đầu sau đó dẫn dắt mềm mại bởi định tuyến Cosine Annealing. Bên trong guồng huấn luyện (Train Loop), chúng tôi cấu hình gắt gao biên đạo hàm chuẩn hóa L2 bởi Gradient Clipping Norm chặn tại mốc $1.0$. Tổ hợp này khiến cực trị không bị trôi giạt và kịch bản phân lớp tối hội sẽ bền bỉ theo dài hạn.

