
import numpy as np 
import argparse 
import random 
import time 
import cv2 
import os

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
ap.add_argument("-m", "--mask_rcnn", required=True, help="path to the mask-rcnn directory")
ap.add_argument("-v", "--visualize", type=int, default=0, help="whether or not we are going to visualize each instance")     # A positive value indicates that we want to visualize how we extracted the masked region on our screen
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probabilityy to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3, help="minimum threshold for pixel-wise maske segmentation")
args = vars(ap.parse_args())

# load the COCO class labels (Mask R-CNN trong này được trained dựa trên COCO dataset 90 classes)
labelsPath = os.path.sep.join([args["mask_rcnn"], "object_detection_classes_coco.txt"])
# Trả về list of labels, nên xem trước định dạng
LABELS = open(labelsPath).read().strip().split("\n")    # strip() loại bỏ spapce, "\n", "\t" ở đầu cuối, split("\n") tách thành list

# load the set of colors sử dụng để hiển thị các instance segmentation
colorsPath = os.path.sep.join([args["mask_rcnn"], "colors.txt"])
# Trả về list of colors, nên xem trước định dạng
COLORS = open(colorsPath).read().strip().split("\n")    # nên nhớ vẫn đang ơ string và chưa đúng định dạng màu
COLORS = [np.array(c.split(",")).astype("int") for c in COLORS]     # chuyển thành list cho từng màu
COLORS = np.array(COLORS, dtype="uint8")

# # Kiểm tra xem lấy màu đứng chưa
# print(COLORS)

# Lấy path đến Mask R-CNN weights and configuration
weightsPath = os.path.sep.join([args["mask_rcnn"], "frozen_inference_graph.pb"])
configPath = os.path.sep.join([args["mask_rcnn"], "mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"])

# Load our Mask R-CNN trained on the COCO dataset (90 classes)
print("[INFO] loading Mask R-CNN from the disk...")
net = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)

# load ảnh, lấy dimensions
image = cv2.imread(args["image"])
(H, W) = image.shape[:2]

# tạ blob (như input đầu vào), thực hiện forward pas cho Mask-RCNN, chungsta nhận được
# bounding boxes và pixel-wise segmentation cho mỗi object
blob = cv2.dnn.blobFromImage(image, swapRB=True, crop=False)
net.setInput(blob)

start = time.time()
(boxes, masks) = net.forward(["detection_out_final", "detection_masks"])
end = time.time()

# Hiển thị thời gian xử lý và thôn tin về Mask R-CNN
print("[INFO] Mask R-CNN took {:.2f} seconds".format(end - start))
print("[INFO] boxes shape: {}".format(boxes.shape))
print("[INFO] boxes shape: {}".format(masks.shape))

""" 
    Có được bounding boxes và các masks rồi sẽ tiến hành filter out các boxes và hiển thị
"""
# Duyệt qua số object phát hiện được:
for i in range(0, boxes.shape[2]):
    # trích xuất class ID cùng với confidence liên quan đến dự đoán
    classID = int(boxes[0, 0, i, 1])
    confidence = boxes[0, 0, i, 2]

    # Lọc các dự đoán có probability nhỏ (giữ lại dự đoán có confidence lớn)
    if confidence > args["confidence"]:
        # clone the original image để vẽ
        clone = image.copy()

        # chuyển đổi toạ độ tưng đối về tọa độ thật và tính kích thước của bounding box
        box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
        """ Nhớ Mask R-CNN trả bounding box ở dạng (startX, startY, endX, endY"""
        (startX, startY, endX, endY) = box.astype("int")    # vì theo pixel
        boxW = endX - startX
        boxH = endY - startY

        # Trích xuất pixel-wise segmentation cho object, resize the mask để
        # nó có cùng dimensions với bounding box, cuối cùng sẽ phân ngưỡng để
        # tạo *binary* mask
        mask = masks[i, classID]
        mask = cv2.resize(mask, (boxW, boxH), interpolation=cv2.INTER_NEAREST)
        mask = (mask > args["threshold"])

        # Trích xuất ROI
        roi = clone[startY:endY, startX:endX]

        # Kiểm tra xem có muốn visualize how to extract the masked region itself
        if args["visualize"] > 0:
            # chuyển mask từ boolean sang integer mask với values thuôc [0, 255]
            # sau đó apply the mask
            visMask = (mask * 255).astype("uint8")
            instance = cv2.bitwise_and(roi, roi, mask=visMask)

            # Hiển thị the extracted ROI, the mask cùng với segmented instance
            cv2.imshow("ROI", roi)
            cv2.imshow("Mask", visMask)
            cv2.imshow("Segmented", instance)

        # Trích xuất duy nhất the masked region của ROI bằng cách đưa vào the booleaan mask aray
        # như slice condition
        roi = roi[mask]

        # Chọn ngẫu nhiên màu sử dụng để hiển thị particular instance segmentation sau đó tạo
        # a transparent overlay by blending the randomly selected color with the ROI
        color = random.choice(COLORS)
        blended = ((0.4 * color) + (0.6 * roi)).astype("uint8")

        # Lưu the blended ROI in the original image
        clone[startY:endY, startX:endX][mask] = blended

        """ Vẽ rectangle, text class label + confidence value """
        color = [int(c) for c in color]
        cv2.rectangle(clone, (startX, startY), (endX, endY), color, 2)

        # Vẽ the predicted label và probability của instance segmentation on the image
        text = "{}: {:.4f}".format(LABELS[classID], confidence)
        cv2.putText(clone, text, (startX, startY - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # show th output image
        cv2.imshow("Output", clone)
        cv2.waitKey(0)