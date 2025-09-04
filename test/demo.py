import cv2,os

import sys
import pathlib
__dir__ = pathlib.Path(os.path.abspath(__file__))
sys.path.append(str(__dir__))
sys.path.append(str(__dir__.parent.parent))

from yolo_seg_ort.core import YOLOSeg


def main():
    image_path = r"E:\RFID\projects\songtao_jiaojie\sigmodel_test\downloaded_images_bohai_keji\20250815\9\FL0102001-01-14.jpg.jpg"
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"无法加载图像：{image_path}。请检查路径是否正确。")
    except Exception as e:
        print(f"加载图像时发生错误：{e}")
        print("请确保您有一个可用的图像文件路径，例如 'sample_image.jpg'")
        return

    onnx_path = r"D:\projects\RFID\OCR\seg-server\yolo-seg-ort\test\shelf_best.onnx"

    try:
        model = YOLOSeg(
            onnx_model=onnx_path,
            classes=["shelf"],
            conf=0.25,
            iou=0.7,
            imgsz=640,
        )
    except Exception as e:
        print(f"加载 ONNX 模型时发生错误：{e}")
        print("请确保您有一个有效的 ONNX 模型文件路径，例如 'best.onnx'")
        return

    print("模型加载成功，正在进行推理...")
    result = model(image)

    if result:
        result[0].save("./results.jpg")
    else:
        print("未检测到任何对象或结果为空。")


if __name__ == "__main__":
    main()
