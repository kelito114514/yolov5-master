import cv2 as cv
import numpy as np

def im_detect(im,model,names,threshold):
    blob = cv.dnn.blobFromImage(im,1/255.0,(640,640),swapRB=True,crop=False)

    model.setInput(blob)

    detections = model.forward()
    #对返回值进行极大值抑制
    index = cv.dnn.NMSBoxes(detections[0][:, :4],detections[0][:, 4],0.5,0.2)
    raw_boxes = detections[0][:, :4][index]
    raw_confidences = detections[0][:,4][index]
    raw_probablities = detections[0][:,5:][index]
    #设置一个置信度阈值，进行二次筛选
    index = raw_confidences > threshold
    raw_boxes = raw_boxes[index]
    raw_confidences=raw_confidences[index]
    raw_probablities = raw_probablities[index]

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
    return im

if __name__=="__main__":
    names = []
    with open("labels.txt",mode="r") as f:
        for line in f.readline():
            names.append(line.strip())
    model = cv.dnn.readNetFromONNX("yolo_helmet.onnx")

    v1 = cv.VideoCapture("workers.mp4")

    fps = v1.get(cv.CAP_PROP_FPS)
    w = int(v1.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(v1.get(cv.CAP_PROP_FRAME_HEIGHT))

    vw = cv.VideoWriter("result.avi",cv.VideoWriter.fourcc(*"XVID"),fps,(w,h))

    flag,im = v1.read()
    while flag:
        result = im_detect(im,model,names,threshold=0.6)
        cv.imshow("detect",result)
        key = cv.waitKey(1)
        if key == 27: #通过esc退出
            break
        flag,im = v1.read()
    v1.release()