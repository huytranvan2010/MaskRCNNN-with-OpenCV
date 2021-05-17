# Image Segmentattion
Có một số algorithms để thực hiện segmentation như:
* Region-based Segmentation: phân chia các objects thành các vùng khác nhau dựa trên một số ngưỡng (số ngưỡng tùy chọn)
* Edge Detection Segmentation: dựa trên tính không liên tục của features để phát hiện các viền, do đó có thể xác định được ranh giới của các vật thể (hay dùng các filter như Sobel...)
* Segmentation based on clustering: chia các pixel thành các nhóm khác nhau (có thể sử dụng phương pháp K-means để phân cụm)
* Mask R-CNN: trả lại cho chúng ta 3 outputs cho mỗi object trong ảnh: class, bounding box và object mask

# MaskRCNNN-with-OpenCV
Trong các bài toán object detection chúng ta nhận được bounding box xung quanh vật thể và class của vật thể. Tuy nhiên bounding box nhận được không cho chúng ta biết:
- Pixels nào thuộc về vật thể
- Pixels nào thuộc về background
Để có thể tách riêng các pixels thuộc về một object chúng ta đi xây dựng bài toán segmentation (phân đoạn ảnh). Bài toán segmentation có 2 dạng:
- Instance segmentation: phân đoạn riêng rẽ từng vật thể
- Semantic segmentation: phân đoạn các vật thể theo nhóm như người, ô tô... (không phân biệt các objects cảu cùng class). Cái này hay được sử dụng trong xe tự hành
![images](https://pyimagesearch.com/wp-content/uploads/2018/11/mask_rcnn_segmentation_types.jpg)

Kiến trúc Mask R-CNN là một ví dụ của `instance segmentation` algorithm.

Mask R-CNN được xây dựng dựa trên Faster R-CNN với 2 điểm mới như sau:
- Thay thể ROI pooling module bằng ROI Align module chính xác hơn
- Chèn them một nhánh ra (2 CONV layers) từ ROI Align để tạo ra `mask`

![image2](https://pyimagesearch.com/wp-content/uploads/2018/11/mask_rcnn_arch.png)

Như chúng ta đã biết Faster R-CNN sử dụng Region Proposal Network để tạo ra các vùng trong ảnh có khả năng chứa object trong đó. Mỗi vùng này được sắp xếp theo "objectness score" (khả năng có chứa object ở trong đó). N vùng có objectness scores lớn nhất được giữ lại. Trong bài báo gốc Faster R-CNN sử dụng N=2000, thực tế ta có thể sử dụng các giá trị N nhỏ hơn mà vẫn cho kết quả tốt, ví dụ N = {30, 50, 100, 200}

Trong thư mục `mask-rcnn-coco` chứa:

# Tài liệu tham khảo
https://www.analyticsvidhya.com/blog/2019/04/introduction-image-segmentation-techniques-python/
