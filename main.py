import argparse
import cv2
import utils
import inferences.engines as engines


def main(args):
    sources = args.sources
    outputs = args.outputs

    for source, output in zip(sources, outputs):
        image = cv2.imread(source)
        results, detections = engines.inference(image)

        for detection in detections:
            image = utils.draw_bound(image, detection)
            image = utils.draw_label(image, detection)

        image = utils.draw_result(image, (20, 20), results[0])
        image = utils.draw_result(image, (50, 20), results[1])
        image = utils.draw_result(image, (80, 20), results[2])

        cv2.imwrite(output, image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--sources', nargs='+', required=True)
    parser.add_argument('-o', '--outputs', nargs='+', required=True)

    main(parser.parse_args())
