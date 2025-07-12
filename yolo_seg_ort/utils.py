import numpy as np
import cv2
import time
from typing import Tuple


def xywh2xyxy(x: np.ndarray) -> np.ndarray:
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # x1
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # y1
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # x2
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # y2
    return y


def box_iou(box1: np.ndarray, box2: np.ndarray) -> np.ndarray:
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    inter_x1 = np.maximum(b1_x1[:, None], b2_x1)
    inter_y1 = np.maximum(b1_y1[:, None], b2_y1)
    inter_x2 = np.minimum(b1_x2[:, None], b2_x2)
    inter_y2 = np.minimum(b1_y2[:, None], b2_y2)

    inter_width = np.maximum(0, inter_x2 - inter_x1)
    inter_height = np.maximum(0, inter_y2 - inter_y1)
    inter_area = inter_width * inter_height

    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = b1_area[:, None] + b2_area - inter_area

    iou = inter_area / (union_area + 1e-6)
    return iou


def non_max_suppression(
    prediction: np.ndarray,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    classes=None,
    agnostic: bool = False,
    multi_label: bool = False,
    max_det: int = 300,
    nc: int = 0,
    max_time_img: float = 0.05,
    max_nms: int = 30000,
    max_wh: int = 7680,
):
    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]

    bs = prediction.shape[0]
    nc = nc or (prediction.shape[1] - 4)
    extra = prediction.shape[1] - nc - 4
    mi = 4 + nc

    prediction = prediction.transpose(0, 2, 1)

    xc = prediction[:, :, 4:mi].max(axis=2) > conf_thres

    time_limit = 2.0 + max_time_img * bs
    multi_label &= nc > 1

    t = time.time()
    output = [np.zeros((0, 6 + extra), dtype=np.float32)] * bs

    for xi in range(bs):
        x = prediction[xi]

        filt = xc[xi]
        x_filtered = x[filt]

        if not x_filtered.shape[0]:
            continue

        boxes_xywh = x_filtered[:, :4]
        boxes_xyxy = xywh2xyxy(boxes_xywh.copy())
        x_processed = np.concatenate((boxes_xyxy, x_filtered[:, 4:]), axis=1)

        box = x_processed[:, :4]
        cls = x_processed[:, 4:mi]
        mask = x_processed[:, mi:]

        if multi_label:
            conf_per_class = cls
            conf = np.max(conf_per_class, axis=1, keepdims=True)
            j = np.argmax(conf_per_class, axis=1, keepdims=True)
            best_class_conf_filt = conf.flatten() > conf_thres
            box = box[best_class_conf_filt]
            conf = conf[best_class_conf_filt]
            j = j[best_class_conf_filt]
            mask = mask[best_class_conf_filt]

            x_final = np.concatenate((box, conf, j.astype(np.float32), mask), axis=1)

        else:
            conf = np.max(cls, axis=1, keepdims=True)
            j = np.argmax(cls, axis=1, keepdims=True)

            filt_conf = conf.flatten() > conf_thres
            x_final = np.concatenate((box, conf, j.astype(np.float32), mask), axis=1)[
                filt_conf
            ]

        if classes is not None and len(x_final) > 0:
            classes_np = np.array(classes)
            filt_class = np.isin(x_final[:, 5], classes_np)
            x_final = x_final[filt_class]

        n = x_final.shape[0]
        if not n:
            continue

        if n > max_nms:
            order = x_final[:, 4].argsort(axis=0)[::-1][:max_nms]
            x_final = x_final[order]

        boxes = x_final[:, :4]
        scores = x_final[:, 4]
        classes_for_nms = x_final[:, 5]

        if agnostic:
            boxes_for_nms = boxes
        else:
            boxes_for_nms = boxes + classes_for_nms[:, None] * max_wh

        keep = []
        if n > 0:
            indices = np.argsort(scores)[::-1]

            while len(indices) > 0:
                i = indices[0]
                keep.append(i)

                if len(indices) == 1:
                    break

                current_box = boxes_for_nms[i]
                other_boxes = boxes_for_nms[indices[1:]]

                if other_boxes.shape[0] == 0:
                    break

                ious = box_iou(current_box[None, :], other_boxes).flatten()

                low_iou_indices = np.where(ious <= iou_thres)[0]

                indices = indices[1:][low_iou_indices]

        i = np.array(keep, dtype=np.int32)
        i = i[:max_det]

        output[xi] = x_final[i]

        if (time.time() - t) > time_limit:
            print(f"NMS 时间限制 {time_limit:.3f}s 超出")
            break

    return output


def clip_boxes(boxes: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    height, width = shape
    boxes[..., 0] = np.clip(boxes[..., 0], 0, width)
    boxes[..., 2] = np.clip(boxes[..., 2], 0, width)
    boxes[..., 1] = np.clip(boxes[..., 1], 0, height)
    boxes[..., 3] = np.clip(boxes[..., 3], 0, height)
    return boxes


def scale_boxes(
    img1_shape: Tuple[int, int],
    boxes: np.ndarray,
    img0_shape: Tuple[int, int],
    ratio_pad=None,
    padding: bool = True,
    xywh: bool = False,
) -> np.ndarray:
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])

        pad_w = round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1)
        pad_h = round((img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1)
        pad = (pad_w, pad_h)
    else:
        gain = ratio_pad[0]
        pad = ratio_pad[1]

    if padding:
        boxes[..., 0] -= pad[0]  # x 填充
        boxes[..., 1] -= pad[1]  # y 填充
        if not xywh:
            boxes[..., 2] -= pad[0]  # x 填充
            boxes[..., 3] -= pad[1]  # y 填充
    boxes[..., :4] /= gain
    return clip_boxes(boxes, img0_shape)


def scale_masks(
    masks: np.ndarray, shape: Tuple[int, int], padding: bool = True
) -> np.ndarray:
    num_masks, mh_proto, mw_proto = masks.shape

    gain = min(mh_proto / shape[0], mw_proto / shape[1])  # old / new
    pad_w_total = mw_proto - shape[1] * gain
    pad_h_total = mh_proto - shape[0] * gain

    if padding:
        left = int(round(pad_w_total / 2 - 0.1))
        top = int(round(pad_h_total / 2 - 0.1))
        right = mw_proto - int(round(pad_w_total / 2 + 0.1))
        bottom = mh_proto - int(round(pad_h_total / 2 + 0.1))
        masks_cropped = masks[:, top:bottom, left:right]
    else:
        masks_cropped = masks

    rescaled_masks = []
    target_h, target_w = shape
    for i in range(num_masks):
        mask_resized = cv2.resize(
            masks_cropped[i].astype(np.float32),
            (target_w, target_h),
            interpolation=cv2.INTER_LINEAR,
        )
        rescaled_masks.append(mask_resized)

    return np.array(rescaled_masks)  # (N, target_H, target_W)


def crop_mask(masks: np.ndarray, bboxes: np.ndarray) -> np.ndarray:
    n, h, w = masks.shape
    x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]

    r = np.arange(w, dtype=np.float32)[None, None, :]  # (1,1,w)
    c = np.arange(h, dtype=np.float32)[None, :, None]  # (1,h,1)

    x1 = x1[:, None, None]
    y1 = y1[:, None, None]
    x2 = x2[:, None, None]
    y2 = y2[:, None, None]

    cropped_mask_indices = (r >= x1) * (r < x2) * (c >= y1) * (c < y2)

    return masks * cropped_mask_indices.astype(masks.dtype)
