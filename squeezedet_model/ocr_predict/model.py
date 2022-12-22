import numpy as np
import pathlib
import tensorflow as tf
import json
import cv2

class Model(object):
    def __init__(self, model_filepath, model_desc_filepath):
        if not pathlib.Path(model_filepath).is_file():
            raise str(model_filepath) + ' file not exist!'
        if not pathlib.Path(model_desc_filepath).is_file():
            raise str(model_desc_filepath) + ' file not exist!'

        self.graph = tf.Graph()
        self.sess = tf.compat.v1.InteractiveSession(graph=self.graph)

        with tf.compat.v1.gfile.GFile(model_filepath, 'rb') as f:
            self.graph_def = tf.compat.v1.GraphDef()
            self.graph_def.ParseFromString(f.read())

        self.load_config(model_desc_filepath)
        self.FINAL_THRESHOLD = 0.2
        self.EXP_THRESH = 1e-5
        self.TOP_N_DETECTION = 64
        self.NMS_THRESH = 0.3
        self.PROB_THRESH = 0.4

        self.input = tf.compat.v1.placeholder(np.float32, shape=[None, self.IMAGE_HEIGHT, self.IMAGE_WIDTH, 3], name=self.INPUT_NAME)
        tf.import_graph_def(self.graph_def, {self.INPUT_NAME: self.input})

    def load_config(self, model_desc_filepath):
        with open(model_desc_filepath,'r') as f:
            desc_data = json.load(f)
        self.INPUT_NAME = str(desc_data['input_names'][0])
        self.OUTPUT_NAME = 'import/' + str(desc_data['output_names'][0]) + ':0'
        self.CLASSES = desc_data['anchors']['classes']
        self.IMAGE_HEIGHT = int(desc_data['anchors']['height']) * 16
        self.IMAGE_WIDTH = int(desc_data['anchors']['width']) * 16
        self.ANCHOR_SEED = desc_data['anchors']['sizes_wh']
        self.ANCHOR_PER_GRID = np.shape(self.ANCHOR_SEED)[0]
        self.set_anchors()


    def predict(self, image):
        self.SCALE = (np.shape(image)[0]/self.IMAGE_HEIGHT, np.shape(image)[1]/self.IMAGE_WIDTH)
        image = cv2.resize(image, (self.IMAGE_WIDTH, self.IMAGE_HEIGHT))
        image = image.astype(np.float32, copy=False)
        image = (image - np.mean(image)) / (np.std(image) + self.EXP_THRESH)

        output_tensor = self.graph.get_tensor_by_name(self.OUTPUT_NAME)
        orig_shape = np.shape(image)
        image_data = np.reshape(image, (1, orig_shape[0], orig_shape[1], orig_shape[2]))
        output = self.sess.run(output_tensor, feed_dict={self.input: image_data})
        return self.filter_batch(output)

    def get_layers(self):
        return [op.name for op in self.graph.get_operations()]


    #compute the anchors for the grid from the seed
    def set_anchors(self):
        H, W, B = self.IMAGE_HEIGHT // 16, self.IMAGE_WIDTH // 16, self.ANCHOR_PER_GRID

        anchor_shapes = np.reshape(
          [self.ANCHOR_SEED] * H * W,
          (H, W, B, 2)
        )
        center_x = np.reshape(
          np.transpose(
              np.reshape(
                  np.array([np.arange(1, W+1)*float(self.IMAGE_WIDTH)/(W+1)]*H*B),
                  (B, H, W)
              ),
              (1, 2, 0)
          ),
          (H, W, B, 1)
        )
        center_y = np.reshape(
          np.transpose(
              np.reshape(
                  np.array([np.arange(1, H+1)*float(self.IMAGE_HEIGHT)/(H+1)]*W*B),
                  (B, W, H)
              ),
              (2, 1, 0)
          ),
          (H, W, B, 1)
        )
        anchors = np.reshape(
          np.concatenate((center_x, center_y, anchor_shapes), axis=3),
          (-1, 4)
        )

        self.ANCHOR_BOX = anchors
        self.N_ANCHORS_HEIGHT = H
        self.N_ANCHORS_WIDTH = W
        self.ANCHORS = len(self.ANCHOR_BOX)


    def filter_batch(self, y_pred):
        # slice predictions vector
        pred_class_probs, pred_conf, pred_box_delta = self.slice_predictions(y_pred)
        det_boxes = self.boxes_from_deltas(pred_box_delta)

        # compute class probabilities
        probs = pred_class_probs * np.reshape(pred_conf, [1, self.ANCHORS, 1])
        det_probs = np.max(probs, 2)
        det_class = np.argmax(probs, 2)

        # count number of detections
        num_detections = 0
        # filter predictions with non maximum suppression
        filtered_bbox, filtered_score, filtered_class = \
            self.filter_prediction(det_boxes[0], det_probs[0], det_class[0])

        # you can use this to use as a final filter for the confidence score
        keep_idx = [idx for idx in range(len(filtered_score))
                    if filtered_score[idx] > float(self.FINAL_THRESHOLD)
                    ]
        final_boxes = [filtered_bbox[idx] for idx in keep_idx]
        final_probs = [filtered_score[idx] for idx in keep_idx]
        final_class = [filtered_class[idx] for idx in keep_idx]
        final_class_name = [self.CLASSES[idx] for idx in final_class ]
        num_detections += len(filtered_bbox)

        for idx in range(len(final_boxes)):
            final_boxes[idx][0] *= self.SCALE[1]
            final_boxes[idx][1] *= self.SCALE[0]
            final_boxes[idx][2] *= self.SCALE[1]
            final_boxes[idx][3] *= self.SCALE[0]
            final_boxes[idx] = self.bbox_transform(final_boxes[idx])

        return final_boxes, final_probs, final_class, final_class_name

    def filter_prediction(self, boxes, probs, cls_idx):
        # check for top n detection flags
        if self.TOP_N_DETECTION < len(probs) and self.TOP_N_DETECTION > 0:
            order = probs.argsort()[:-self.TOP_N_DETECTION - 1:-1]
            probs = probs[order]
            boxes = boxes[order]
            cls_idx = cls_idx[order]
        else:
            filtered_idx = np.nonzero(probs > self.PROB_THRESH)[0]
            probs = probs[filtered_idx]
            boxes = boxes[filtered_idx]
            cls_idx = cls_idx[filtered_idx]

        final_boxes = []
        final_probs = []
        final_cls_idx = []

        # go trough classes
        for c in range(len(self.CLASSES)):
            idx_per_class = [i for i in range(len(probs)) if cls_idx[i] == c]

            # do non maximum suppresion
            keep = self.nms(boxes[idx_per_class], probs[idx_per_class], self.NMS_THRESH)
            for i in range(len(keep)):
                if keep[i]:
                    final_boxes.append(boxes[idx_per_class[i]])
                    final_probs.append(probs[idx_per_class[i]])
                    final_cls_idx.append(c)

        return final_boxes, final_probs, final_cls_idx

    def boxes_from_deltas(self, pred_box_delta):
        # Keras backend allows no unstacking
        delta_x = pred_box_delta[:, :, 0]
        delta_y = pred_box_delta[:, :, 1]
        delta_w = pred_box_delta[:, :, 2]
        delta_h = pred_box_delta[:, :, 3]

        # get the coordinates and sizes of the anchor boxes from config
        anchor_x = self.ANCHOR_BOX[:, 0]
        anchor_y = self.ANCHOR_BOX[:, 1]
        anchor_w = self.ANCHOR_BOX[:, 2]
        anchor_h = self.ANCHOR_BOX[:, 3]

        # as we only predict the deltas, we need to transform the anchor box values before computing the loss
        box_center_x = anchor_x + delta_x * anchor_w
        box_center_y = anchor_y + delta_y * anchor_h
        box_width = anchor_w * self.safe_exp(delta_w, self.EXP_THRESH)
        box_height = anchor_h * self.safe_exp(delta_h, self.EXP_THRESH)

        # tranform into a real box with four coordinates

        xmins, ymins, xmaxs, ymaxs = self.bbox_transform([box_center_x, box_center_y, box_width, box_height])

        # trim boxes if predicted outside

        xmins = np.minimum(
            np.maximum(0.0, xmins), self.IMAGE_WIDTH - 1.0)
        ymins = np.minimum(
            np.maximum(0.0, ymins), self.IMAGE_HEIGHT - 1.0)
        xmaxs = np.maximum(
            np.minimum(self.IMAGE_WIDTH - 1.0, xmaxs), 0.0)
        ymaxs = np.maximum(
            np.minimum(self.IMAGE_HEIGHT - 1.0, ymaxs), 0.0)

        det_boxes = np.transpose(
            np.stack(self.bbox_transform_inv([xmins, ymins, xmaxs, ymaxs])),
            (1, 2, 0)
        )

        return (det_boxes)

    def batch_iou(self, boxes, box):
        lr = np.maximum(
            np.minimum(boxes[:, 0] + 0.5 * boxes[:, 2], box[0] + 0.5 * box[2]) - \
            np.maximum(boxes[:, 0] - 0.5 * boxes[:, 2], box[0] - 0.5 * box[2]),
            0
        )
        tb = np.maximum(
            np.minimum(boxes[:, 1] + 0.5 * boxes[:, 3], box[1] + 0.5 * box[3]) - \
            np.maximum(boxes[:, 1] - 0.5 * boxes[:, 3], box[1] - 0.5 * box[3]),
            0
        )
        inter = lr * tb
        union = boxes[:, 2] * boxes[:, 3] + box[2] * box[3] - inter
        return inter / union

    def nms(self, boxes, probs, threshold):
        order = probs.argsort()[::-1]
        keep = [True] * len(order)

        for i in range(len(order) - 1):
            ovps = self.batch_iou(boxes[order[i + 1:]], boxes[order[i]])
            for j, ov in enumerate(ovps):
                if ov > threshold:
                    keep[order[j + i + 1]] = False
        return keep

    def slice_predictions(self, y_pred):
        # calculate non padded entries
        n_outputs = len(self.CLASSES) + 1 + 4
        # slice and reshape network output
        y_pred = y_pred[:, :, 0:n_outputs]
        y_pred = np.reshape(y_pred, (1, self.N_ANCHORS_HEIGHT, self.N_ANCHORS_WIDTH, -1))

        # number of class probabilities, n classes for each anchor
        num_class_probs = self.ANCHOR_PER_GRID * len(self.CLASSES)

        # slice pred tensor to extract class pred scores and then normalize them
        pred_class_probs = np.reshape(
            self.softmax(
                np.reshape(
                    y_pred[:, :, :, :num_class_probs],
                    [-1, len(self.CLASSES)]
                )
            ),
            [1, self.ANCHORS, len(self.CLASSES)],
        )

        # number of confidence scores, one for each anchor + class probs
        num_confidence_scores = self.ANCHOR_PER_GRID + num_class_probs

        # slice the confidence scores and put them trough a sigmoid for probabilities
        pred_conf = self.sigmoid(
            np.reshape(
                y_pred[:, :, :, num_class_probs:num_confidence_scores],
                [1, self.ANCHORS]
            )
        )

        # slice remaining bounding box_deltas
        pred_box_delta = np.reshape(
            y_pred[:, :, :, num_confidence_scores:],
            [1, self.ANCHORS, 4]
        )

        return [pred_class_probs, pred_conf, pred_box_delta]

    def bbox_transform(self, bbox):
        cx, cy, w, h = bbox
        out_box = [[]] * 4
        out_box[0] = cx - w / 2
        out_box[1] = cy - h / 2
        out_box[2] = cx + w / 2
        out_box[3] = cy + h / 2
        return out_box


    def bbox_transform_inv(self, bbox):
        xmin, ymin, xmax, ymax = bbox
        out_box = [[]] * 4
        width = xmax - xmin + 1.0
        height = ymax - ymin + 1.0
        out_box[0] = xmin + 0.5 * width
        out_box[1] = ymin + 0.5 * height
        out_box[2] = width
        out_box[3] = height
        return out_box

    def softmax(self, x, axis=-1):
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / np.expand_dims(np.sum(e_x, axis=axis), axis=axis)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.maximum(x, -20)))

    def safe_exp(self, w, thresh):
      slope = np.exp(thresh)
      lin_bool = w > thresh
      lin_region = lin_bool.astype(float)
      lin_out = slope*(w - thresh + 1.)
      exp_out = np.exp(np.where(lin_bool, np.zeros_like(w), w))
      out = lin_region*lin_out + (1.-lin_region)*exp_out
      return out