# yolo-seg-ort


<!-- PROJECT SHIELDS -->
![Contributors](https://img.shields.io/github/contributors/7emotions/yolo-seg-ort.svg?style=flat-square)
![Forks](https://img.shields.io/github/forks/7emotions/yolo-seg-ort.svg?style=flat-square)
![Stargazers](https://img.shields.io/github/stars/7emotions/yolo-seg-ort.svg?style=flat-square)
![Issues](https://img.shields.io/github/issues/7emotions/yolo-seg-ort.svg?style=flat-square)
![MIT License](https://img.shields.io/github/license/7emotions/yolo-seg-ort.svg?style=flat-square)
![LinkedIn](https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555)

<!-- PROJECT LOGO -->
<br />

<p align="center">  <a href="https://github.com/7emotions/yolo-seg-ort/">
  <img src="images/logo.png" alt="Logo">
  </a>
  <h3 align="center">yolo-seg-ort</h3>
  <p align="center">
    采用纯ONNX Runtime实现YOLOv11-seg的onnx模型。<br />
    <a href="https://github.com/7emotions/yolo-seg-ort"><strong>探索本项目的文档 »</strong></a><br />
    <br />
    <a href="https://github.com/7emotions/yolo-seg-ort/releases">查看发布</a>
    ·
    <a href="https://github.com/7emotions/yolo-seg-ort/issues">报告Bug</a>
    ·
    <a href="https://github.com/7emotions/yolo-seg-ort/issues">提出新特性</a>
  </p>
</p>

## 1. 模型转换

```python

from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("best.pt")

# Export the model to ONNX format
model.export(format="onnx")  # creates 'yolo11n.onnx'

```

## 2. 安装依赖

```bash
pip install -r requirements.txt
```

## 3. 用法

```python
from yolo_seg_ort import YOLOSeg
import cv2

onnx_path = "best.onnx"
image_path = "test.jpg"

image = cv2.imread(image_path)

model = YOLOSeg(
    onnx_model=onnx_path,
    classes=["Grass", "Ground", "Ramp", "Road", "Stairs"],
    conf=0.25,
    iou=0.7,
    imgsz=640,
)

result = model(image)

if result:
    result[0].save("./results.jpg")
    # result[0].show()
else:
    print("未检测到任何对象或结果为空。")

```

## 4. 结果

<img src="test/test.jpg"/><img src="test/results.jpg" />

## 5. 贡献者

[7emotions](https://github.com/7emotions)

## 6. 许可证

本项目采用 MIT 许可证。有关详细信息，请查看 [LICENSE](LICENSE) 文件。
