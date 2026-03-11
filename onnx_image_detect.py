import cv2 as cv
import numpy as np



def get_labelname(path):
    name_list = []
    with open(path, mode="r") as f:
        for line in f.readlines():
            name_list.append(line.strip())
    return name_list


# 读取图像
image = cv.imread("./test_im/helmet2.jpg")
# 读取标签
names = get_labelname("labels.txt")
# 加载模型
model = cv.dnn.readNetFromONNX("best.onnx")
# 将图像转为blob数据
blob = cv.dnn.blobFromImage(image, 1 / 255.0, (640, 640), swapRB=True, crop=False)
# 设置模型输入
model.setInput(blob)
# 调用模型完成推理
detections = model.forward()
# 进行极大值抑制
index = cv.dnn.NMSBoxes(detections[0][:, :4], detections[0][:, 4], 0.5, 0.2)
raw_box = detections[0][:, :4][index]
raw_confidence = detections[0][:, 4][index]
raw_probabilities = detections[0][:, 5:][index]
# 设置阈值完成二次筛选
threshold = 0.6
index = raw_confidence > threshold
raw_box = raw_box[index]
raw_confidence = raw_confidence[index]
raw_probabilities = raw_probabilities[index]

# 将坐标映射回原图
y_factor = image.shape[0] / 640
x_factor = image.shape[1] / 640

# x,y,w,h 为中心点坐标，目标框的宽和高
for (x, y, w, h), confidence, probabilities in zip(raw_box, raw_confidence, raw_probabilities):
    box_x = int((x - w / 2) * x_factor)
    box_y = int((y - h / 2) * y_factor)
    box_w = int(w * x_factor)
    box_h = int(h * y_factor)
    index = np.argmax(probabilities)
    name = names[index]
    info = name + "%.2f%%" % (confidence * 100)
    # 绘制结果
    cv.rectangle(image, pt1=(box_x, box_y), pt2=(box_x + box_w, box_y + box_h), thickness=2, color=(0, 0, 255),
                 lineType=cv.LINE_AA)
    cv.putText(image, text=info, org=(box_x, box_y + 20), fontFace=cv.FONT_HERSHEY_SIMPLEX,
               fontScale=0.8, color=(0, 255, 0), thickness=2)

# 显示最终结果
cv.namedWindow("1", cv.WINDOW_NORMAL)
cv.imshow("1", image)
cv.waitKey()
