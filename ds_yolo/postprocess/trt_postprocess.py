from loguru import logger

import numpy as np

np.set_printoptions(threshold=np.inf)

INPUT_HEIGHT = 640
INPUT_WIDTH = 640


class BoundingBox:
    def __init__(self, classID, confidence, x1, x2, y1, y2, image_width, image_height):
        self.classID = classID
        self.confidence = confidence
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.u1 = x1 / image_width
        self.u2 = x2 / image_width
        self.v1 = y1 / image_height
        self.v2 = y2 / image_height
    
    def box(self):
        return self.x1, self.y1, self.x2, self.y2
        
    def width(self):
        return self.x2 - self.x1
    
    def height(self):
        return self.y2 - self.y1

    def center_absolute(self):
        return 0.5 * (self.x1 + self.x2), 0.5 * (self.y1 + self.y2)
    
    def center_normalized(self):
        return 0.5 * (self.u1 + self.u2), 0.5 * (self.v1 + self.v2)
    
    def size_absolute(self):
        return self.x2 - self.x1, self.y2 - self.y1
    
    def size_normalized(self):
        return self.u2 - self.u1, self.v2 - self.v1


def nms(boxes, box_confidences, nms_threshold=0.5):
    x_coord = boxes[:, 0]
    y_coord = boxes[:, 1]
    width = boxes[:, 2]
    height = boxes[:, 3]

    areas = width * height
    ordered = box_confidences.argsort()[::-1]

    keep = list()
    while ordered.size > 0:
        i = ordered[0]
        keep.append(i)
        xx1 = np.maximum(x_coord[i], x_coord[ordered[1:]])
        yy1 = np.maximum(y_coord[i], y_coord[ordered[1:]])
        xx2 = np.minimum(x_coord[i] + width[i], x_coord[ordered[1:]] + width[ordered[1:]])
        yy2 = np.minimum(y_coord[i] + height[i], y_coord[ordered[1:]] + height[ordered[1:]])

        width1 = np.maximum(0.0, xx2 - xx1 + 1)
        height1 = np.maximum(0.0, yy2 - yy1 + 1)
        intersection = width1 * height1
        union = (areas[i] + areas[ordered[1:]] - intersection)

        iou = intersection / union

        indexes = np.where(iou <= nms_threshold)[0]
        ordered = ordered[indexes + 1]
    keep = np.array(keep).astype(int)
    return keep


def postprocess(buffer, image_width, image_height, conf_threshold=0.4, nms_threshold=0.5):
    detected_objects = []
    img_scale = [image_width / INPUT_WIDTH, image_height / INPUT_HEIGHT, image_width / INPUT_WIDTH,
                 image_height / INPUT_HEIGHT]
    num_bboxes = int(buffer[0, 0, 0, 0])

    if num_bboxes:
        bboxes = buffer[0, 1: (num_bboxes * 6 + 1), 0, 0].reshape(-1, 6)
        labels = set(bboxes[:, 5].astype(int))

        for label in labels:
            selected_bboxes = bboxes[np.where((bboxes[:, 5] == label) & (bboxes[:, 4] >= conf_threshold))]
            selected_bboxes_keep = selected_bboxes[nms(selected_bboxes[:, :4], selected_bboxes[:, 4], nms_threshold)]
            for idx in range(selected_bboxes_keep.shape[0]):
                box_xy = selected_bboxes_keep[idx, :2]
                box_wh = selected_bboxes_keep[idx, 2:4]
                score = selected_bboxes_keep[idx, 4]

                box_x1y1 = box_xy - (box_wh / 2)
                box_x2y2 = np.minimum(box_xy + (box_wh / 2), [INPUT_WIDTH, INPUT_HEIGHT])
                box = np.concatenate([box_x1y1, box_x2y2])
                box *= img_scale

                if box[0] == box[2]:
                    continue
                if box[1] == box[3]:
                    continue
                detected_objects.append(
                    BoundingBox(label, float(score), box[0], box[2], box[1], box[3], image_height, image_width))

    return detected_objects
