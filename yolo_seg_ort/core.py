import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Tuple, Union

from .utils import non_max_suppression, scale_boxes, scale_masks, crop_mask


class Results:
    def __init__(
        self,
        orig_img: np.ndarray,
        path: str,
        names: List[str],
        boxes: np.ndarray,
        masks: np.ndarray = None,
    ):
        self.orig_img = orig_img.copy()
        self.path = path
        self.names = names
        self.boxes = boxes  # N, 6 (x1, y1, x2, y2, conf, cls_id)
        self.masks = masks  # N, H_orig, W_orig (binary masks)

    def show(
        self, window_name: str = "YOLO Segmentation Result", show_mask: bool = True
    ):
        img = self.legend(show_mask)
        cv2.imshow(window_name, img)

    def save(self, path: str):
        cv2.imwrite(path, self.legend(show_mask=True))

    def legend(self, show_mask=True):

        display_img = self.orig_img.copy()

        h, w, _ = display_img.shape

        colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
        ]
        num_colors = len(colors)

        outline_color = (0, 0, 255)
        outline_thickness = 8

        if show_mask and self.masks is not None and self.masks.shape[0] > 0:
            for i, mask in enumerate(self.masks):
                cls_id = int(self.boxes[i, 5])
                if cls_id != 1 and cls_id != 4:
                    continue

                color_idx = int(self.boxes[i, 5]) % num_colors
                mask_color = colors[color_idx]

                binary_mask = mask.astype(np.uint8) * 255

                colored_mask = np.zeros_like(display_img, dtype=np.uint8)
                colored_mask[binary_mask > 0] = mask_color

                display_img = cv2.addWeighted(display_img, 1.0, colored_mask, 0.5, 0)

                contours, _ = cv2.findContours(
                    binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(
                    display_img, contours, -1, outline_color, outline_thickness
                )

        for i, box in enumerate(self.boxes):
            x1, y1, x2, y2, conf, cls_id = map(int, box[:6])
            if cls_id != 1 and cls_id != 4:
                continue

            conf_val = box[4]

            color_idx = int(cls_id) % num_colors
            box_color = colors[color_idx]

            cv2.rectangle(display_img, (x1, y1), (x2, y2), box_color, 2)

            class_name = self.names[int(cls_id)]
            label = f"{class_name} {conf_val:.2f}"
            print(f"检测到 {class_name}，置信度：{conf_val:.2f}")
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            font_thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(
                label, font, font_scale, font_thickness
            )

            cv2.rectangle(
                display_img,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width + 5, y1),
                box_color,
                -1,
            )

            cv2.putText(
                display_img,
                label,
                (x1 + 2, y1 - baseline - 2),
                font,
                font_scale,
                (255, 255, 255),
                font_thickness,
                cv2.LINE_AA,
            )

        return display_img


class YOLOSeg:
    def __init__(
        self,
        onnx_model: str,
        conf: float = 0.25,
        iou: float = 0.7,
        imgsz: Union[int, Tuple[int, int]] = 640,
    ):
        self.session = ort.InferenceSession(
            onnx_model,
            providers=["CPUExecutionProvider"],
        )
        self.imgsz = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz
        self.classes = ["Grass", "Ground", "Ramp", "Road", "Stairs"]
        self.conf = conf
        self.iou = iou

    def __call__(self, img: np.ndarray) -> List[Results]:
        orig_img_shape = img.shape[:2]
        prep_img, ratio_pad = self.preprocess(img, self.imgsz)

        outs = self.session.run(None, {self.session.get_inputs()[0].name: prep_img})

        return self.postprocess(img, prep_img, outs, orig_img_shape, ratio_pad)

    def letterbox(
        self, img: np.ndarray, new_shape: Tuple[int, int] = (640, 640)
    ) -> Tuple[np.ndarray, Tuple[float, Tuple[int, int]]]:

        shape = img.shape[:2]

        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2

        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )

        gain = r
        pad = (left, top)
        return img, (gain, pad)

    def preprocess(
        self, img: np.ndarray, new_shape: Tuple[int, int]
    ) -> Tuple[np.ndarray, Tuple[float, Tuple[int, int]]]:
        img, ratio_pad = self.letterbox(img, new_shape)
        img = img[..., ::-1].transpose([2, 0, 1])[None]
        img = np.ascontiguousarray(img)
        img = img.astype(np.float32) / 255
        return img, ratio_pad

    def postprocess(
        self,
        orig_img: np.ndarray,
        prep_img: np.ndarray,
        outs: List[np.ndarray],
        orig_img_shape: Tuple[int, int],
        ratio_pad: Tuple[float, Tuple[int, int]],
    ) -> List[Results]:

        preds, protos = outs

        num_classes = len(self.classes)
        preds_processed = non_max_suppression(
            preds, self.conf, self.iou, nc=num_classes
        )

        results = []
        for i, pred_single_image in enumerate(preds_processed):
            if pred_single_image.shape[0] == 0:
                results.append(
                    Results(
                        orig_img,
                        path="",
                        names=self.classes,
                        boxes=np.zeros((0, 6)),
                        masks=np.zeros((0, orig_img_shape[0], orig_img_shape[1])),
                    )
                )
                continue
            scaled_boxes = scale_boxes(
                prep_img.shape[2:],
                pred_single_image[:, :4],
                orig_img_shape,
                ratio_pad=ratio_pad,
            )

            masks_in = pred_single_image[:, 6:]

            masks = self.process_mask(protos[i], masks_in, scaled_boxes, orig_img_shape)

            results.append(
                Results(
                    orig_img,
                    path="",
                    names=self.classes,
                    boxes=np.concatenate(
                        (scaled_boxes, pred_single_image[:, 4:6]), axis=1
                    ),
                    masks=masks,
                )
            )

        return results

    def process_mask(
        self,
        protos: np.ndarray,
        masks_in: np.ndarray,
        bboxes: np.ndarray,
        shape: Tuple[int, int],
    ) -> np.ndarray:

        masks_linear = masks_in @ protos.reshape(protos.shape[0], -1)
        masks = masks_linear.reshape(
            masks_in.shape[0], protos.shape[1], protos.shape[2]
        )  # (N, H_proto, W_proto)

        masks = scale_masks(masks, shape)

        masks = crop_mask(masks, bboxes)

        return masks > 0.0
