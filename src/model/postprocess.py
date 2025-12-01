import numpy as np

ANCHORS = np.array([
    [(12,16), (19,36), (40,28)],     # small objects
    [(36,75), (76,55), (72,146)],    # medium
    [(142,110), (192,243), (459,401)]  # large
], dtype=np.float32)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def decode_yolo_output(feats, anchors, num_classes, input_shape):
    """
    Convert raw YOLO output into actual bounding boxes.
    """
    grid_h, grid_w = feats.shape[1:3]
    box_xy = sigmoid(feats[..., :2])
    box_wh = np.exp(feats[..., 2:4]) * anchors

    # grid offsets
    grid_x = np.arange(grid_w).reshape(1, grid_w, 1, 1)
    grid_y = np.arange(grid_h).reshape(grid_h, 1, 1, 1)
    grid_x = np.tile(grid_x, (grid_h, 1, len(anchors), 1))
    grid_y = np.tile(grid_y, (1, grid_w, len(anchors), 1))

    box_xy = (box_xy + np.concatenate([grid_x, grid_y], axis=-1)) / np.array([grid_w, grid_h])
    box_wh = box_wh / np.array(input_shape)

    # xywh → x_min, y_min, x_max, y_max
    box_x1y1 = box_xy - (box_wh / 2)
    box_x2y2 = box_xy + (box_wh / 2)

    boxes = np.concatenate([box_x1y1, box_x2y2], axis=-1)
    objectness = sigmoid(feats[..., 4])
    class_probs = sigmoid(feats[..., 5:])

    return boxes, objectness, class_probs


def non_max_suppression(boxes, scores, threshold=0.5):
    """
    Standard NMS for bounding boxes.
    """

    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    areas = (x2-x1) * (y2-y1)
    order = scores.argsort()[::-1]  # descending

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2-xx1)
        h = np.maximum(0, yy2-yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou <= threshold)[0]
        order = order[inds + 1]

    return keep


def yolo_postprocess(outputs, input_shape, num_classes, conf_threshold=0.4, nms_threshold=0.5):
    """
    Postprocess raw YOLOv4 output tensors (3 heads) into
    clean list of bounding boxes + class scores.
    """

    decoded = []
    for i, out in enumerate(outputs):
        grid_h, grid_w = out.shape[1:3]
        anchors = ANCHORS[i]

        boxes, obj, class_probs = decode_yolo_output(out[0], anchors, num_classes, input_shape)

        # flatten per-anchor predictions
        boxes = boxes.reshape(-1, 4)
        obj = obj.reshape(-1)
        class_probs = class_probs.reshape(-1, num_classes)

        scores = obj * class_probs  # elementwise
        class_ids = np.argmax(scores, axis=-1)
        class_scores = np.max(scores, axis=-1)

        # confidence filter
        mask = class_scores >= conf_threshold
        boxes = boxes[mask]
        class_scores = class_scores[mask]
        class_ids = class_ids[mask]

        decoded.append((boxes, class_scores, class_ids))

    # concatenate outputs from all 3 YOLO scales
    boxes = np.concatenate([d[0] for d in decoded], axis=0)
    scores = np.concatenate([d[1] for d in decoded], axis=0)
    class_ids = np.concatenate([d[2] for d in decoded], axis=0)

    # NMS
    keep = non_max_suppression(boxes, scores, threshold=nms_threshold)

    results = []
    for i in keep:
        results.append({
            "class_id": int(class_ids[i]),
            "score": float(scores[i]),
            "box": boxes[i].tolist()    # normalized [x1, y1, x2, y2]
        })

    return results

