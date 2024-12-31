import cv2
import os
import time

import inferences.inference as inference


def save_results(image, image_name, detections, results):
    for bbox, label in detections:
        x1 = bbox[0]
        y1 = bbox[1]
        x2 = bbox[2]
        y2 = bbox[3]

        image = cv2.putText(image, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255))
        image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255))

    for index, x in enumerate((20, 50, 80)):
        image = cv2.circle(image, (x, 20), 12, (0, 255, 0) if results[index] else (0, 0, 255), -1)

    cv2.imwrite(f'inferences/results/result_{image_name}', image)


def main():
    for image_name in os.listdir('inferences/images'):
        image = cv2.imread(f'inferences/images/{image_name}')

        time1 = time.perf_counter()
        detections, results = inference.inference(image)
        time2 = time.perf_counter()

        save_results(image, image_name, detections, results)

        cv2.imshow('Inference Result', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print(f'Image: {image_name:<8} Time: {time2 - time1:.3f}s')


if __name__ == '__main__':
    main()
