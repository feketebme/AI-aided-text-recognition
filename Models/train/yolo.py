
from ultralytics import YOLO



if __name__=="__main__":
    # Load a model
    model =  YOLO('yolov8n.yaml').load('best_2.pt')

    results= model.train(data="D:\PycharmProjects\graphisoft\configs\data\yolo.yaml",epochs=100,patience=10,batch=-1, plots=True,imgsz=640,verbose=True, device=[0])