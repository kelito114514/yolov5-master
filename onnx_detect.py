import cv2 as cv
import numpy as np

models = cv.dnn.readNetFromONNX("yolo_helmet.onnx")

im = cv.imread("helmet2.jpg")
print("读入图像的形状:",im.shape)

blob = cv.dnn.blobFromImage(im,1/255.0,(640,640),swapRB=True,crop=False)
print("模型输入的形状:",blob.shape)

models.setInput(blob)

detections = models.forward()
#对返回值进行极大值抑制
index = cv.dnn.NMSBoxes(detections[0][:, :4],detections[0][:, 4],0.5,0.2)
raw_boxes = detections[0][:, :4][index]
raw_confidences = detections[0][:,4][index]
raw_probablities = detections[0][:,5:][index]
print(detections)
#设置一个置信度阈值，进行二次筛选
threshold = 0.6
index = raw_confidences > threshold
raw_boxes = raw_boxes[index]
raw_confidences=raw_confidences[index]
raw_probablities = raw_probablities[index]

#names = ["helmet","head"]
names = []
with open("labels.txt",mode="r") as f:
    for line in f.readline():
        names.append(line.strip())
#还原目标框在原图上的位置
x_factor = im.shape[1]/640
y_factor = im.shape[0]/640
for (x,y,w,h),confidence,probablities in zip(raw_boxes,raw_confidences,raw_probablities):
    box_x = int((x-w/2)*x_factor)
    box_y = int((y-h/2)*y_factor)
    box_w = int(w * x_factor)
    box_h = int(h * y_factor)
    print(box_x,box_y,box_w,box_h)
    index = np.argmax(probablities)
    name = names[index]
    name_confidence = name + "%.2f%%"%(confidence*100)
    cv.rectangle(im,pt1=(box_x,box_y),pt2=(box_x+box_w,box_y+box_h),color=(0,0,255))
    cv.putText(im,text=name_confidence,org=(box_x,box_y-10),fontFace=cv.FONT_HERSHEY_SIMPLEX,
               fontScale=1,color=(0,0,0),thickness=2,lineType=cv.LINE_AA)

cv.imshow("1",im)
cv.waitKey()