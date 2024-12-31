import onnxruntime as ort
import cv2
import yaml
import numpy as np


with open('configs/inference.yaml', 'r') as configs:
    configs = yaml.load(configs, Loader=yaml.FullLoader)

classes_labels = [
    'F0', 'F1',
    'L0', 'L1',
    'S0', 'S1',
    'R0', 'R1',
]

session = ort.InferenceSession(configs['model-path'], providers=configs['session-providers'])


def preprocess(image):
    inputs = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose((2, 0, 1))
    inputs = inputs / 255.0
    inputs = np.expand_dims(inputs, axis=0)

    if configs['precision'] == 'fp16':
        return inputs.astype(np.float16)
    else:
        return inputs.astype(np.float32)


def get_valid_outputs(outputs):
    valid_outputs = outputs[np.amax(outputs[:, 4:12], axis=1) > configs['conf-threshold']]

    bboxes = valid_outputs[:, 0:4]
    scores = valid_outputs[:, 4:12]

    return bboxes.astype(np.int32), np.amax(scores, axis=1), np.argmax(scores, axis=1)


def non_max_suppression(outputs):
    bboxes, scores, classes = get_valid_outputs(outputs)

    bboxes[:, 0] -= bboxes[:, 2] >> 1
    bboxes[:, 1] -= bboxes[:, 3] >> 1

    for index in cv2.dnn.NMSBoxes(bboxes, scores, configs['conf-threshold'], configs['iou-threshold'], eta=0.5):
        x1 = bboxes[index, 0]
        y1 = bboxes[index, 1]

        x2 = bboxes[index, 2] + bboxes[index, 0]
        y2 = bboxes[index, 3] + bboxes[index, 1]

        yield (x1, y1, x2, y2), classes_labels[classes[index]]


def detection_inference(image):
    outputs = session.run(['output0'], {'images': preprocess(image)})
    outputs = outputs[0]
    outputs = outputs.squeeze().transpose()

    return [detection for detection in non_max_suppression(outputs)]


def inference(image):
    detections = detection_inference(image)

    result0 = False
    result1 = False
    result2 = True

    for _, label in detections:
        if label == 'F1':
            result0 = True
            result1 = True

    for _, label in detections:
        if label == 'L0':
            result0 = False
        if label == 'L1':
            result0 = True
        if label == 'S0':
            result1 = False
        if label == 'S1':
            result1 = True
        if label == 'R0':
            result2 = False
        if label == 'R1':
            result2 = True

    return detections, (result0, result1, result2)
