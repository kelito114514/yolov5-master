import os
import xml.etree.ElementTree as ET




def convert(size, box):
    """
    将边界框坐标转换为YOLO格式
    :param size: 图像的宽度和高度
    :param box: 边界框的坐标 (xmin, xmax, ymin, ymax)
    :return: YOLO格式的边界框坐标 (x_center, y_center, width, height)
    """
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


def convert_annotation(xml_file_path, output_dir):
    """
    将单个XML标注文件转换为YOLO格式的标注文件
    :param xml_file_path: XML标注文件的路径
    :param output_dir: 输出YOLO格式标注文件的目录
    """
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    txt_file_name = os.path.splitext(os.path.basename(xml_file_path))[0] + '.txt'
    txt_file_path = os.path.join(output_dir, txt_file_name)

    with open(txt_file_path, 'w') as out_file:
        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in classes or int(difficult) == 1:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                 float(xmlbox.find('ymax').text))
            bb = convert((w, h), b)
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


def voc_to_yolo(voc_annotations_dir, yolo_labels_dir):
    """
    将整个VOC格式数据集转换为YOLO格式数据集
    :param voc_annotations_dir: VOC格式标注文件所在的目录
    :param yolo_labels_dir: 输出YOLO格式标注文件的目录
    """
    if not os.path.exists(yolo_labels_dir):
        os.makedirs(yolo_labels_dir)

    for xml_file in os.listdir(voc_annotations_dir):
        if xml_file.endswith('.xml'):
            xml_file_path = os.path.join(voc_annotations_dir, xml_file)
            convert_annotation(xml_file_path, yolo_labels_dir)


# 定义类别名称到类别ID的映射
classes = ["helmet", "head"]
voc_annotations_dir = r'.//small_head_helmet/annotations'
yolo_labels_dir = r'.//small_head_helmet/labels'
if not os.path.exists(yolo_labels_dir): # 如果当前文件夹不存在
    os.makedirs(yolo_labels_dir)        # 创建一个文件夹
voc_to_yolo(voc_annotations_dir, yolo_labels_dir)

