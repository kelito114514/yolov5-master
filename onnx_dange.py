import cv2 as cv
import numpy as np
import os

# 加载模型
models = cv.dnn.readNetFromONNX("yolo_helmet.onnx")

# 读取类别名称
names = []
with open("labels.txt", mode="r") as f:
    for line in f.readlines():
        names.append(line.strip())

# 定义文件夹路径
folder_path = "test_im"  # 替换为你的图像文件夹路径

# 遍历文件夹中的所有图像文件
for filename in os.listdir(folder_path):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        file_path = os.path.join(folder_path, filename)

        # 读取图像
        im = cv.imread(file_path)
        print(f"读入图像 {filename} 的形状:", im.shape)

        # 预处理图像
        blob = cv.dnn.blobFromImage(im, 1/255.0, (640, 640), swapRB=True, crop=False)
        print("模型输入的形状:", blob.shape)

        # 设置模型输入并进行前向传播
        models.setInput(blob)
        detections = models.forward()

        # 对返回值进行极大值抑制
        index = cv.dnn.NMSBoxes(detections[0][:, :4], detections[0][:, 4], 0.5, 0.2)
        raw_boxes = detections[0][:, :4][index]
        raw_confidences = detections[0][:, 4][index]
        raw_probablities = detections[0][:, 5:][index]

        # 设置一个置信度阈值，进行二次筛选
        threshold = 0.6
        index = raw_confidences > threshold
        raw_boxes = raw_boxes[index]
        raw_confidences = raw_confidences[index]
        raw_probablities = raw_probablities[index]

        # 还原目标框在原图上的位置
        x_factor = im.shape[1] / 640
        y_factor = im.shape[0] / 640
        for (x, y, w, h), confidence, probablities in zip(raw_boxes, raw_confidences, raw_probablities):
            box_x = int((x - w / 2) * x_factor)
            box_y = int((y - h / 2) * y_factor)
            box_w = int(w * x_factor)
            box_h = int(h * y_factor)
            print(box_x, box_y, box_w, box_h)
            index = np.argmax(probablities)
            name = names[index]
            name_confidence = name + "%.2f%%" % (confidence * 100)
            cv.rectangle(im, pt1=(box_x, box_y), pt2=(box_x + box_w, box_y + box_h), color=(0, 0, 255))
            cv.putText(im, text=name_confidence, org=(box_x, box_y - 10), fontFace=cv.FONT_HERSHEY_SIMPLEX,
                       fontScale=1, color=(0, 0, 0), thickness=2, lineType=cv.LINE_AA)

        # 显示检测结果
        cv.imshow("result", im)
        cv.waitKey(0)  # 按任意键继续处理下一张图像

cv.destroyAllWindows()