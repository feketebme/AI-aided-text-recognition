import copy
import glob
import os.path
from ultralytics import YOLO
import cv2
import argparse
import tqdm


# Visualize YOLO results on the image

def load_model_and_predict(path_to_weights_yolo: str, ):
    model_yolo = YOLO(path_to_weights_yolo)
    #model_ocr = model
    return model_yolo


def process_yolo_results(results, **kwargs):
    visualize=kwargs.pop("visualize",False)
    save=kwargs.pop("save",False)
    image_path=kwargs.pop("image_path",None)
    save_path=kwargs.pop("save_path",None)
    name=kwargs.pop("name",None)

    if  image_path is None:
        raise ValueError("image_path input is required.")
    if save and save_path is None or name is None:
        raise ValueError("save_path:str, name inputs are required if you want to save results.")
    bounding_boxes=[]
    [bounding_boxes.append(result.boxes.xywh.squeeze().tolist()) for result in results]
    # Read the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if save or visualize:
        img=copy.deepcopy(image)
        [cv2.rectangle(img, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2+10)), (0, 0, 255), 2) for x, y, w, h in bounding_boxes]

        if save:
            os.makedirs(os.path.join(save_path + f"/results/{name}"),exist_ok=True)
            path = os.path.join(save_path + f"/results/{name}/yolo.png")
            cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if visualize:
            cv2.imshow('YOLO Results', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    cropped_images = [image[int(y - h / 2):int(y + h / 2+10), int(x - w / 2):int(x + w / 2)] for x, y, w, h in
                      bounding_boxes]
    return cropped_images



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8n Inference")
    parser.add_argument("--model_weights_yolo", required=True, help="Path to the pretrained YOLOv8n model")
    parser.add_argument("--images", required=True, help="Path to folder contains images file")
    parser.add_argument("--save_path",  help="Path to save the results",default=None)
    parser.add_argument("--visualize", default=False)
    parser.add_argument("--save", default=False)
    args = parser.parse_args()
    # Load a pretrained YOLOv8n model
    images = glob.glob(os.path.join(args.images, "*.*"))
    if images == []:
        raise ValueError(f"No image found under {args.images} folder.")
    model_yolo = load_model_and_predict(path_to_weights_yolo=args.model_weights_yolo)
    outer = tqdm.tqdm(total=len(images), desc='Files', position=0)
    for image in images:
        name = image.replace('\\', '/').split("/")[-1].split('.')[0]
        outer.set_description_str(f"Process {name} image")
        results = model_yolo.predict(image, conf=0.5)  # list of Results objects
        outer.update(1)

        cropped_images=process_yolo_results(image_path=image, results=results[0],
                          save_path=args.save_path,
                          name=name,visualize=(args.visualize==True), save=(args.save==True))
        # for image in cropped_images:
        #     cv2.imshow('YOLO Results', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()


