import numpy as np
import torch
import cv2 as cv
import os


# 读取图像
image_path = 'test_im/helmet.jpg'  # 替换为你的图像路径
image = cv.imread(image_path)

def im_detect(im,model):
    results = model(image)
    # 显示检测结果
    for info in results.xyxy:
        info = info.cpu().numpy()
        x1,y1,x2,y2 = info[0,:4]
        probability = info[0,4:]
        index = np.argmax(probability)
        confidence = probability[index]*100
        cv.rectangle(image, pt1=(int(x1), int(y1)), pt2=(int(x2), int(y2)), color=(0,0,255),
                     thickness=2, lineType=cv.LINE_AA)
        cv.putText(image, text=results.names.get(index), org=(int(x2), int(y2)),
                   fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,255,0),
                   thickness=1)
        cv.putText(image, text="%.2f%%" % confidence, org=(int(x2), int(y2+30)),
                   fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,255,0),
                   thickness=1)
    return im

# 加载自己训练的YOLOv5模型
model = torch.hub.load(r'D:\python项目\yolov5-master', 'custom',
                       path='yolo_helmet.pt', source='local')
files_path=".//test_im"
#遍历文件夹
for fn in os.listdir(files_path):
    image_path = os.path.join(files_path,fn)
    im = cv.imread(image_path)
    result = im_detect(im,model)
    cv.imwrite("result//"+fn,result)