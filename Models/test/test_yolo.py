import glob
import os.path

from ultralytics import YOLO
import cv2

import argparse

# Visualize YOLO results on the image

def load_model_and_predict(path_to_weights:str,):
    model = YOLO(path_to_weights)
    # Run inference on the source

    return model
def visualize_results(image_path:str, results,save_path:str,index:int):
    # Read the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Plot bounding boxes on the image
    for result in results:
        # Get bounding box coordinates
        x, y, w, h = result.boxes.xywh.squeeze().tolist()

        # Draw bounding box
        cv2.rectangle(image, (int(x-w/2), int(y-h/2)), (int(x + w/2), int(y + h/2)), (0, 0, 255), 2)

        # Display class and confidence
        #plt.text(x, y, f"{result['class']} ({result['confidence']:.2f})", color='red')

    try:
        path=os.path.join(save_path+f"/results/{index}.png")
        cv2.imwrite(path,cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    except Exception as e:
        print(e)
    #cv2.imshow('YOLO Results', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="YOLOv8n Inference")
    parser.add_argument("--model", required=True, help="Path to the pretrained YOLOv8n model")
    parser.add_argument("--image", required=True, help="Path to the image file")
    args = parser.parse_args()
    # Load a pretrained YOLOv8n model
    path_to_folder=r"C:\Users\balazs.fekete\Desktop\Senior AI Developer Homework Assignment\testfiles"
    images=glob.glob(os.path.join(path_to_folder,"*.*"))
    model = load_model_and_predict(r'D:\PycharmProjects\graphisoft\Models\train\best.pt')
    for index,image in enumerate(images):
        results = model.predict(image, conf=0.5)  # list of Results objects
        visualize_results(image_path=image,results= results[0],save_path=r"C:\Users\balazs.fekete\Desktop\Senior AI Developer Homework Assignment", index=index)

