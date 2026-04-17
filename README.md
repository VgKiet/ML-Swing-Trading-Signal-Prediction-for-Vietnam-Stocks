# ML Swing Trading Signal Prediction for Vietnam Stocks

## Giới thiệu

Dự án xây dựng mô hình Machine Learning nhằm dự đoán tín hiệu mua (BUY)
và bán (SELL) cổ phiếu theo phương pháp Swing Trading trên thị trường
chứng khoán Việt Nam bằng Logistic Regression kết hợp với các chỉ báo kỹ
thuật.

## Mô tả bài toán

Trong thị trường chứng khoán, việc xác định thời điểm mua và bán cổ
phiếu hợp lý đóng vai trò quan trọng để tối ưu lợi nhuận và giảm rủi ro.
Tuy nhiên, việc xác định các điểm đảo chiều giá (Swing High / Swing Low)
bằng phương pháp thủ công thường phụ thuộc nhiều vào kinh nghiệm nhà đầu
tư và dễ bị ảnh hưởng bởi nhiễu thị trường.

Do đó, dự án tập trung xây dựng một mô hình Machine Learning có khả năng
tự động dự đoán tín hiệu BUY/SELL dựa trên dữ liệu giá lịch sử và các
chỉ báo kỹ thuật.

## Mục tiêu đề tài

-   Xây dựng mô hình Logistic Regression để dự đoán tín hiệu BUY/SELL
-   Xác định các điểm đảo chiều giá (Swing High / Swing Low)
-   Kết hợp các chỉ báo kỹ thuật: Supertrend, STC, Donchian Channel, ATR
-   Lọc tín hiệu nhiễu để tăng độ chính xác dự đoán
-   Trực quan hóa tín hiệu giao dịch trên biểu đồ giá cổ phiếu
-   Lưu mô hình đã huấn luyện để sử dụng cho dự đoán realtime

## Công nghệ sử dụng

-   Python
-   Pandas, NumPy
-   Scikit-learn
-   Matplotlib
-   tvDatafeed

## Quy trình thực hiện

1.  Thu thập dữ liệu cổ phiếu từ HOSE bằng tvDatafeed
2.  Tạo nhãn dữ liệu Swing High / Swing Low
3.  Trích xuất đặc trưng từ các chỉ báo kỹ thuật
4.  Chuẩn hóa dữ liệu bằng StandardScaler
5.  Huấn luyện Logistic Regression
6.  Tối ưu siêu tham số bằng GridSearchCV
7.  Lọc tín hiệu bằng Donchian Channel và ATR
8.  Trực quan hóa tín hiệu BUY/SELL trên biểu đồ

## Ứng dụng Streamlit
Dự án được tích hợp ứng dụng Streamlit thông qua file `app.py` nhằm hỗ trợ dự đoán tín hiệu BUY/SELL cổ phiếu theo thời gian thực.

Ứng dụng cho phép người dùng:

- Nhập mã cổ phiếu cần dự đoán
- Tải dữ liệu giá mới nhất từ HOSE bằng tvDatafeed
- Sử dụng mô hình đã huấn luyện (`model.pkl`) để dự đoán tín hiệu giao dịch
- Hiển thị tín hiệu BUY/SELL trực tiếp trên biểu đồ giá cổ phiếu
- Hỗ trợ trực quan hóa kết quả giúp nhà đầu tư dễ dàng theo dõi xu hướng thị trường

### Live Demo
[https://ten-app.streamlit.app](https://ml-swing-trading-signal-prediction-for-vietnam-stocks.streamlit.app/)

## Cách chạy ứng dụng
``` bash
streamlit run app.py
```

## Kết quả

Mô hình có khả năng phát hiện các điểm mua và bán tiềm năng trên dữ liệu
cổ phiếu lịch sử, hỗ trợ nhà đầu tư trong việc ra quyết định giao dịch
theo phương pháp Swing Trading.

![demo](https://github.com/VgKiet/ML-Swing-Trading-Signal-Prediction-for-Vietnam-Stocks/blob/master/image/demo.png?raw=true)

## File mô hình

Sau khi huấn luyện, mô hình được lưu tại:

model.pkl

để sử dụng cho các bước phát triển tiếp theo như dự đoán realtime hoặc
triển khai ứng dụng Streamlit.
