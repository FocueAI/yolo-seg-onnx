import os,cv2
import numpy as np
from yolo_seg_ort.core import YOLOSeg

class shelf_divider:
    def __init__(self, model_path="shelf_best.pt"):
        try:
            self.model = YOLOSeg(model_path,
                                classes=['shelf'],
                                conf=0.25,
                                iou=0.7,
                                imgsz=640,
                                )
        except Exception as e:
            print(f"加载模型时发生错误：{e}")
            print("请确保您有一个有效的模型文件路径，例如 'shelf_best.pt'")
            return 
        
    def predict(self, img_path, is_show=False):
        image = cv2.imread(img_path)
        results = self.model(image)
        line_l = []
        for result in results:
            detect_zone = result.masks[0] # [h,w]
            detect_zone = detect_zone.astype(np.uint8)
            
            
            contours, _ = cv2.findContours(detect_zone*255,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
            largest = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(largest)
            box  = cv2.boxPoints(rect)     # 4 个角点
            box  = np.int32(box)
            edges = [np.linalg.norm(box[i] - box[(i+1) % 4]) for i in range(4)]
            longest_edge_idx = np.argmax(edges)
            p1, p2 = box[longest_edge_idx].tolist(), box[(longest_edge_idx+1) % 4].tolist()
            # draw.line([tuple(p1), tuple(p2)], fill='red', width=3)
            line = [tuple(p1), tuple(p2)]
            line_l.append(line)  
        if is_show: self.show(image, line_l)
        return line_l   
    
    def show(self, cv_img, line_l):
        for line in line_l:
            cv2.line(cv_img, line[0], line[1], (255, 0, 0), 3)
        window_name ='Image Window'  
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # 使用NORMAL标志允许调整窗口大小  
        cv2.resizeWindow(window_name, 2000, 1080)  # 设置窗口的初始宽度和高度  
        cv2.imshow(window_name,cv_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    
    
    
          
            
image_path = r"E:\RFID\projects\songtao_jiaojie\sigmodel_test\downloaded_images_bohai_keji\20250815\9\FL0102001-01-14.jpg.jpg"
model_path = r"D:\projects\RFID\OCR\seg-server\yolo-seg-ort\test\shelf_best.onnx"
lines_l = shelf_divider(model_path=model_path).predict(image_path, is_show=True) 

