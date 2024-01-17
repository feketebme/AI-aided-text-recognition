import cv2
import numpy as np
import random
import nltk
from nltk.corpus import words
import os
import tqdm
import logging
import colorlog
from utils import initialize_with_config, do_overlap, Point
import glob
import argparse
import albumentations as A


class DatasetGenerator:

    def __init__(self, number_of_images: int, **kwargs):
        nltk.download('words')
        self.word_list = words.words() + [str(i) for i in range(10 ** 5)] + [chr(ord('a') + i) for i in range(26)] + [
            chr(ord('a') + i) + chr(ord('a') + j) for i in range(26) for j in range(26)]
        self.number_of_images = number_of_images
        self.max_num_words = kwargs.pop("max_num_words", 10)
        self.output_folder = kwargs.pop("output_folder", "./pictures")
        self.possible_background_colors = kwargs.pop("possible_background_colors",
                                                     [(255, 255, 255), (200, 200, 200), (150, 150, 150)])
        self.possible_fonts = kwargs.pop("possible_fonts", ['cv2.FONT_HERSHEY_SIMPLEX', 'cv2.FONT_HERSHEY_COMPLEX',
                                                            'cv2.FONT_HERSHEY_SCRIPT_SIMPLEX'])
        self.image_size_properties = kwargs.pop("image_size_properties", [(300, 780), [500, 1200]])
        self.path_to_logo = kwargs.pop("path_to_logo", "./logo")
        self.max_num_logo = kwargs.pop("max_num_logo", 5)
        self.max_num_rectangle = kwargs.pop("max_num_rectangle", 5)
        self.max_attempts_avoid_overlapping = kwargs.pop("max_attempts_avoid_overlapping", 100)
        self.aim = kwargs.pop("aim", "yolo")
        self.augmentation = kwargs.pop("augmentation", self.get_default_augment_yolo() if self.aim=="yolo" else self.get_default_augment_ocr())
        self.type = kwargs.pop("type", "train")
        self.probabilities_background_colors = kwargs.pop("probabilities_background_colors", [])


        self.logos = glob.glob(os.path.join(self.path_to_logo, "*.*"))
        self.define_color_logger()

    @staticmethod
    def get_default_augment_yolo():
        _ = A.Compose([
            A.Rotate(limit=90, p=0.4, border_mode=cv2.BORDER_CONSTANT)
        ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.2))
        return _

    @staticmethod
    def get_default_augment_ocr():
        _ = A.Compose([
            A.Rotate(limit=10, p=0.4, border_mode=cv2.BORDER_CONSTANT),
            A.GaussianBlur(p=0.2)
        ])
        return _

    def define_color_logger(self):
        # Create a logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

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
        self.logger.addHandler(ch)

    def generate_real_text(self, num_of_words: int):
        return random.sample(self.word_list, num_of_words)

    @staticmethod
    def put_text_to_image(image: np.ndarray, text: str, font, position: tuple, text_scale: float):
        """
            Put text on the image with randomized size and color.

            Args:
                image (numpy.ndarray): The input image.
                text (str): The text to be added to the image.
                font : OpenCV font type.
                text_scale (float): Scale factor for the text size.
                position (tuple): (x, y) coordinates for the starting position of the text.
        """

        # Randomize text color
        text_color = tuple(np.random.randint(0, 256, 3).tolist())

        # Calculate the position for the text
        y_position = position[1]
        x_position = position[0]

        # Put text on the image with randomized size
        cv2.putText(image, text, (x_position, y_position), font, text_scale, text_color, 2)

    def put_logo(self, img: np.ndarray):
        """
                Put a randomly selected logo onto the input image.

                Args:
                    img (numpy.ndarray): The input image.

                Notes:
                    The method randomly selects a logo from the available set of logos, places it on the input image at a
                    random position, and replaces the corresponding region in the input image with the logo.

                    The dimensions of the selected logo must fit within the dimensions of the input image for successful placement.

                Returns:
                    None: The input image is modified in-place with the added logo.
        """
        img_logo = cv2.imread(np.random.choice(self.logos))
        if (img.shape[1] - img_logo.shape[1] + 1) > 0 and (img.shape[0] - img_logo.shape[0] + 1 > 0):
            x1, y1 = np.random.randint(0, img.shape[1] - img_logo.shape[1] + 1), np.random.randint(0, img.shape[0] -
                                                                                                   img_logo.shape[
                                                                                                       0] + 1)
            img[y1:(y1 + img_logo.shape[0]), x1:(x1 + img_logo.shape[1])] = img_logo

    @staticmethod
    def put_rectangle(img: np.ndarray):
        """
            Draws a random rectangle on the input image.

            Parameters:
            - img (np.ndarray): Input image as a NumPy array.

            The function generates random parameters for the rectangle, including
            its starting position (x, y) and dimensions (width, height). It then
            draws the random rectangle on the input image using OpenCV's rectangle function.
        """
        # Generate random rectangle parameters
        x = random.randint(0, img.shape[1] - 1)
        y = random.randint(0, img.shape[0] - 1)
        width = random.randint(10, 150)
        height = random.randint(10, 150)

        # Draw the random rectangle on the image
        color = (0, 0, 0)  # BGR color (here, it's red)
        thickness = 2
        cv2.rectangle(img, (x, y), (x + width, y + height), color, thickness)

    def get_random_position(self, used_positions, image_size, text_size):
        """
        Get a random position within the image boundaries that does not overlap with used positions.

        Args:
            used_positions (list): List of tuples representing used positions.
            image_size (tuple): (width, height) of the image.
            text_size (tuple): (width, height) of the text.

        Returns:
            tuple: (x, y) coordinates representing the random position.
        """

        attempt = 0

        while attempt < self.max_attempts_avoid_overlapping:
            x = random.randint(0, max(1, image_size[0] - text_size[0]))
            y = random.randint(text_size[1], image_size[1])

            # Check if the new position overlaps with any used positions
            overlapping = any(do_overlap(Point(x_1, (y_1 + h_1)), Point((x_1 + w_1), y_1),
                                         Point(x, (y + text_size[1])), Point((x + text_size[0]), y))
                              for x_1, y_1, w_1, h_1 in used_positions)

            if not overlapping:
                return x, y

            attempt += 1

        # If after maximum attempts a non-overlapping position is not found, return a random position
        self.logger.warning("Overlapping text")
        return random.randint(0, max(1, image_size[0] - text_size[0])), random.randint(text_size[1],
                                                                                       max(1, image_size[1]))

    def add_text_to_image(self, index: int, texts: list, used_positions: list, image_width: int, image_height: int,
                          yolo_pose: list, img: np.ndarray):
        """
                Add text to the input image at a random position with random font, scale, and YOLO pose information.

                Args:
                    index (int): Index of the text to be added from the list of texts.
                    texts (list): List of texts to choose from.
                    used_positions (list): List of previously used text positions to avoid overlap.
                    image_width (int): Width of the input image.
                    image_height (int): Height of the input image.
                    yolo_pose (list): List to store YOLO pose information for the added text.
                    img (numpy.ndarray): The input image.

                Notes:
                    The method randomly selects a text from the list, along with a font and text scale.
                    It calculates the dimensions of the text bounding box, gets a random position that doesn't overlap
                    with previously added text, adds the text to the image using the `put_text_to_image` method,
                    updates the list of used positions, and generates YOLO pose information for the added text.

                Returns:
                    None: The input image is modified in-place with the added text.
                """
        text = texts[index]
        text_scale = np.random.uniform(0.3, 1.7)
        font = eval(random.choice(self.possible_fonts))
        text_width, text_height = cv2.getTextSize(text, font, text_scale, 2)[0]
        position = self.get_random_position(used_positions, (image_width, image_height),
                                            (text_width, text_height))

        self.put_text_to_image(text=text, font=font, position=position, image=img, text_scale=text_scale)
        used_positions.append((position[0], position[1], text_width, text_height))
        yolo_pose.append(
            [(position[0] + text_width / 2) / image_width, (position[1] - text_height / 2) / image_height,
             text_width / image_width, text_height / image_height, int(0)])

    def generate_folder_structure(self):

        """Creates the necessary folder structure for images and labels."""
        if self.aim == "yolo":
            os.makedirs(os.path.join(self.output_folder + f"/images/{self.type}"), exist_ok=True)
            os.makedirs(os.path.join(self.output_folder + f"/labels/{self.type}"), exist_ok=True)
            os.makedirs(os.path.join(self.output_folder + f"/labels_for_rec/{self.type}"), exist_ok=True)
        else:
            os.makedirs(os.path.join(self.output_folder + f"/images/{self.type}"), exist_ok=True)
            os.makedirs(os.path.join(self.output_folder + f"/labels/{self.type}"), exist_ok=True)

    def generate_synthetic_yolo_dataset(self):

        """
           Generates a synthetic YOLO dataset by creating images with text.

           Steps:
           1. Create the necessary folder structure.
           2. Log information about the image generation process.
           3. Generate synthetic YOLO images and save them. (Random logo, rectangle, text positions colors)
           4. Save labels in YOLO format and labels for rectangles in separate text files.

        """
        self.generate_folder_structure()
        self.logger.info(f"Generate yolo images to {self.output_folder} folder")
        self.logger.info(f"Produce {self.number_of_images} images.")
        self.logger.warning("Please do not turn off the computer until this process finishes.")

        for i in tqdm.tqdm(range(self.number_of_images)):
            # Generate a random background
            background_color = \
                random.choices(self.possible_background_colors, weights=self.probabilities_background_colors)[0]

            # Generate random image size between 780x1200 and 300x500
            image_width = random.randint(self.image_size_properties[1][0], self.image_size_properties[1][1])
            image_height = random.randint(self.image_size_properties[0][0], self.image_size_properties[0][1])
            img = np.full((image_height, image_width, 3), background_color, dtype=np.uint8)
            [self.put_logo(img) for _ in range(random.randint(1, self.max_num_logo))]
            [self.put_rectangle(img) for _ in range(random.randint(1, self.max_num_rectangle))]

            num_words_per_image = random.randint(2, self.max_num_words)
            used_positions = []
            texts = self.generate_real_text(num_words_per_image)
            yolo_pose = []

            [self.add_text_to_image(index, texts, used_positions, image_width, image_height, yolo_pose, img) for
             index in range(num_words_per_image)]
            boxes = np.asarray(yolo_pose)
            try:
                transformed = self.augmentation(image=img, bboxes=boxes)

                transformed_img = transformed['image']
                transformed_boxes = np.asarray(transformed['bboxes'])
            except ValueError:
                transformed_boxes = boxes
                transformed_img = img
            if transformed_boxes.size > 0:
                transformed_boxes = np.concatenate((transformed_boxes[:, -1].reshape(-1, 1), transformed_boxes[:, :-1]),
                                                   axis=1)

            # Save the image
            image_path = f"{self.output_folder}/images/{self.type}/{i + 1}.png"
            try:
                cv2.imwrite(image_path, transformed_img)
            except Exception as e:
                self.logger.error(e)
            with open(f"{self.output_folder}/labels/{self.type}/{i + 1}.txt", "w") as file:
                file.writelines(
                    '\n'.join([' '.join([str(int(row[0]))] + list(map(str, row[1:]))) for row in transformed_boxes]))
            with open(f"{self.output_folder}/labels_for_rec/{self.type}/{i + 1}.txt", "w") as file:
                file.writelines('\n'.join(texts))

    def generate_synthetic_ocr_dataset(self):
        """
           Generates a synthetic OCR dataset by creating images with text.

           Steps:
           1. Create the necessary folder structure.
           2. Log information about the image generation process.
           3. Augment the images and save the augmented versions.
           4. Save labels into one text file.

        """
        self.generate_folder_structure()
        self.logger.info(f"Generate OCR images to {self.output_folder} folder")
        self.logger.info(f"Produce {self.number_of_images} images.")
        self.logger.warning("Please do not turn off the computer until this process finishes.")
        labels=[]

        for i in tqdm.tqdm(range(self.number_of_images)):
            # Generate a random background
            background_color = \
                random.choices(self.possible_background_colors, weights=self.probabilities_background_colors)[0]
            text = self.generate_real_text(1)[0]
            text_scale = np.random.uniform(0.8, 4)
            font = eval(random.choice(self.possible_fonts))
            text_width, text_height = cv2.getTextSize(text, font, text_scale, 2)[0]

            image_width = text_width + +random.randint(40, 70)
            image_height = text_height + random.randint(40, 70)
            img = np.full((image_height, image_width, 3), background_color, dtype=np.uint8)
            self.put_text_to_image(image=img, text=text, font=font, position=(random.randint(0,image_width-text_width),image_height-random.randint(0,image_height-text_height)), text_scale=text_scale)
            # Save the image
            image_path = f"{self.output_folder}/images/{self.type}/{i + 1}.png"
            labels.append(f"{image_path}; {text}")
            try:
                transformed = self.augmentation(image=img)

                transformed_img = transformed['image']
                cv2.imwrite(image_path, transformed_img)
            except Exception as e:
                self.logger.error(e)
        with open(f"{self.output_folder}/labels/{self.type}/labels.txt", "w") as file:
            file.writelines('\n'.join(labels))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize with config file")
    parser.add_argument("--config", type=str, help="Path to the config file", required=True)
    args = parser.parse_args()

    dataset = initialize_with_config(DatasetGenerator, config_file=args.config)
    dataset.logger.info(f"Initialize with config {args.config} file.")

    try:
        dataset.generate_synthetic_dataset() if dataset.aim == 'yolo' else dataset.generate_synthetic_ocr_dataset()
        dataset.logger.info("Process finished.")
    except Exception as e:
        dataset.logger.error(e)
