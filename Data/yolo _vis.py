import cv2
import matplotlib.pyplot as plt

def read_annotation_file(annotation_path):
    with open(annotation_path, 'r') as file:
        lines = file.readlines()
    return lines

def parse_annotation_line(line):
    values = line.strip().split()
    _,x_center, y_center, width, height = map(float, values)
    return x_center, y_center, width, height

def draw_bbox(image, x_center, y_center, width_box, height_box):
    height, width, _ = image.shape
    x_min = int((x_center - width_box/2) * width)
    y_min = int((y_center - height_box/2) * height)
    x_max = int((x_center + width_box/2) * width)
    y_max = int((y_center + height_box/2) * height)

    color = (0, 255, 0)  # Green color for bounding box
    thickness = 2

    image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)
    return image

def visualize_annotation(image_path, annotation_path):
    image = cv2.imread(image_path)
    lines = read_annotation_file(annotation_path)

    for line in lines:
        x_center, y_center, width, height = parse_annotation_line(line)
        image = draw_bbox(image, x_center, y_center, width, height)

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('YOLO Annotation Visualization')
    plt.show()

if __name__ == "__main__":
    for i in range(1,100):
        image_path= fr"D:\PycharmProjects\graphisoft\Data\my_generated_images\images\val\{i}.png"
        annotation_path=fr"D:\PycharmProjects\graphisoft\Data\my_generated_images\labels\val\{i}.txt"
        visualize_annotation(image_path, annotation_path)
