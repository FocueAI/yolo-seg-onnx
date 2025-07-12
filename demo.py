import cv2
from yolo_seg_ort.core import YOLOSeg


def main():
    image_path = "./images/test.jpg"
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"无法加载图像：{image_path}。请检查路径是否正确。")
    except Exception as e:
        print(f"加载图像时发生错误：{e}")
        print("请确保您有一个可用的图像文件路径，例如 'sample_image.jpg'")
        return

    onnx_path = "./models/best.onnx"

    try:
        model = YOLOSeg(onnx_model=onnx_path)
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
