import csv
import time

import easyocr
import copy
import glob
import os.path
import cv2
import argparse

import tqdm
import logging
import colorlog


def define_color_logger():
    # Create a logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Create a colorized formatter
    formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(message)s",
        datefmt=None,
        reset=True,
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        },
        secondary_log_colors={},
        style='%'
    )

    # Create a console handler and set the formatter
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)

    # Add the console handler to the logger
    logger.addHandler(ch)
    return logger


def process_ocr_results(results, **kwargs):
    visualize=kwargs.pop("visualize",False)
    save=kwargs.pop("save",False)
    image_path=kwargs.pop("image_path",None)
    save_path=kwargs.pop("save_path",None)
    name=kwargs.pop("name",None)

    if  image_path is None:
        raise ValueError("image_path input is required.")
    if save and save_path is None or name is None:
        raise ValueError("save_path:str, index inputs are required if you want to save results.")
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if save or visualize:
        # Draw bounding boxes around the detected text
        for detection in results:
            try:
                top_left = tuple([int(param) for param in detection[0][0]])
                bottom_right = tuple([int(param) for param in detection[0][2]])
                text = detection[1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                color = (255, 0, 0)  # Blue color in BGR
                thickness = 1
                cv2.rectangle(image, top_left, bottom_right, color, thickness)
                cv2.putText(image, text, top_left, font, font_scale, color, thickness, cv2.LINE_AA)
            except Exception as e:
                print(e)


        if save:
            os.makedirs(os.path.join(save_path + f"/results/{name}"),exist_ok=True)
            path = os.path.join(save_path + f"/results/{name}/result.png")
            cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

            output_file = os.path.join(save_path + f"/results/{name}/result.csv")
            with open(output_file, 'w', newline='') as csvfile:
                fieldnames = ['top_left', 'bottom_right', 'text', 'confidence']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for detection in results:
                    top_left = ','.join(map(str, detection[0][0]))
                    bottom_right = ','.join(map(str, detection[0][2]))
                    text = detection[1]
                    confidence = detection[2]
                    writer.writerow(
                        {'top_left': top_left, 'bottom_right': bottom_right, 'text': text, 'confidence': confidence})
        if visualize:
            cv2.imshow('OCR Results', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)
            cv2.destroyAllWindows()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EasyOCR Inference")
    parser.add_argument("--images", required=True, help="Path to folder contains images file")
    parser.add_argument("--save_path",  help="Path to save the results",default=None)
    parser.add_argument("--visualize",default=False)
    parser.add_argument("--save", default=False)
    args = parser.parse_args()

    logger=define_color_logger()
    images = glob.glob(os.path.join(args.images, "*.*"))
    if images == []:
        raise ValueError(f"No image found under {args.images} folder.")
    model_ocr=easyocr.Reader(['en'])
    outer = tqdm.tqdm(total=len(images), desc='Files', position=0)
    for image in images:

        name=image.replace('\\','/').split("/")[-1].split('.')[0]
        outer.set_description_str(f"Process {name} image")
        results = model_ocr.readtext(image)
        outer.update(1)
        process_ocr_results(image_path=image, results=results,
                          save_path=args.save_path,
                          name=name,visualize=(args.visualize==True), save=(args.save==True))




