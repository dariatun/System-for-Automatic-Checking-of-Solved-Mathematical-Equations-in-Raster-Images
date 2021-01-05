import argparse
import os
import sys
import cv2
import numpy as np

from application.answer_correctness_checker import confirm_results, get_emotion, get_text_from_result
from application.changeables import BOOL_SHOW, BOOL_SAVE
from application.constants import WINDOW_WIDTH, WINDOW_HEIGHT, DETAILED_MODE, EVALUATIONAL_MODE, LABELS_PATH, \
    CONFIG_PATH, WEIGHTS_PATH, DEFAULT_PROGRAM_IMGS_PATH, CORNER_IMAGE_PATH, SIMPLE_MODE, DEFAULT_IMG_NAME, \
    FINAL_TEXT_LBL, SMILE, SMILE_PATH, NEUTRAL_PATH, NEUTRAL, SAD_PATH, OUTPUT_PATH, TXT, NUMBER_OF_IMAGES, \
    NUMBER_IMAGES_PODTYPES, NUMBER_OF_IMAGE_TYPES, HANDWRITTEN, EQUATIONS, DEBUG_FILENAME, DEBUG_ONE_FILE, JPG, BOX, \
    CLASS_ID, PREDICTION, EXPRESSIONS_LBL, TEXT_RESULTS_LBL, TEST_DATA_PATH
from application.object_localisation import extract_boxes_confidences_classids, run_nms_algorithm
from application.image_manipulation import rotate_image, draw_rectangle, put_text
from application.logger import print_overall_prediction_correctness, print_prediction_percent, print_merged_answer
from application.object_detection import run_object_detection
from application.prediction_matrix_creation import create_matrix_from_lines, sort_by_y_coordinate, sort_by_x_coordinate, \
    divide_by_lines
from application.scaner import scan
from application.evaluation_mode import compare_predictions, compare_box_predictions
from application.utils import add_to_dictionary, get_element_with_the_biggest_value
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QMenuBar, QAction
from PyQt5.QtGui import QPixmap, QGuiApplication, QFont


def get_application_arguments():
    """
    Process the arguments of the application
    :return: application's arguments as a variable
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, default=DEFAULT_PROGRAM_IMGS_PATH,
                        help='Path to the images for recognition')
    parser.add_argument('-e', '--evaluation', action='store_true', default=False,
                        help='The program will start evaluation')
    return parser.parse_args()


class Application(QWidget):

    def __init__(self):
        super().__init__()
        self.title = 'Application'
        self.left = 10
        self.top = 10
        self.width = WINDOW_WIDTH
        self.height = WINDOW_HEIGHT
        self.widgets()
        self.mode = DETAILED_MODE

        args = get_application_arguments()

        self.search_dir = args.path
        if args.evaluation:
            self.mode = EVALUATIONAL_MODE

        self.labels = open(LABELS_PATH).read().strip().split('\n')
        self.colors = np.random.randint(0, 255, size=(len(self.labels) + 1, 3), dtype='uint8')
        self.net = cv2.dnn.readNetFromDarknet(CONFIG_PATH, WEIGHTS_PATH)
        self.layer_names = self.net.getLayerNames()
        self.layer_names = [self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        self.start_button = QPushButton("START", self)
        self.start_button.setFont(QFont("Tekton Pro", 50))
        self.start_button.setStyleSheet("background-color: #b31c48;"
                                        "border: 5px solid;"
                                        "border-color: #003286;"
                                        "color: #ffda00")
        self.start_button.clicked.connect(self.start_button_action)

        self.create_start_button()

        self.pred_matrixes = []

        self.objects_number = [0, 0]

        # evaluation files and variables
        self.pred_boxes_file = None
        self.pred_file_merged = None
        self.pred_file = None
        self.pred_file_equation = None
        self.pred_file_handwritten = None
        self.pred_file_equation_merged = None
        self.pred_file_handwritten_merged = None

        self.boxes_incorrect_file = None
        self.pred_incorrect_file = None
        self.pred_incorrect_merged_file = None

        self.overall_pred_correctness_merged = 0
        self.overall_pred_correctness = 0
        self.overall_box_pred_correctness = 0
        self.overall_pred_correctness_obj = [0, 0]
        self.overall_pred_correctness_obj_merged = [0, 0]

        self.showed_labels = []

        self.filenames_list = []
        self.filename_indx = 0

        self.set_background()
        if self.mode == EVALUATIONAL_MODE:
            self.run_evaluation()
        else:
            self.show()

    def show_image(self, x, y, width, height, image=None, path='', to_save=True):
        """
        Displays image in the application
        :param x: x-coordinate of the image location
        :param y: y-coordinate of the image location
        :param width: width of the image
        :param height: height of the image
        :param image: the image opened in OpenCV
        :param path: path to the image
        :param to_save: boolean, decides if the label of the image will be saved or not
        :return:
        """
        label = QLabel(self)
        filename = path
        if len(path) == 0:
            filename = "temporary/{}.png".format(os.getpid())
            cv2.imwrite(filename, image)

        pixmap = QPixmap(filename)
        label.setPixmap(pixmap.scaled(width, height))
        label.move(x, y)
        if len(path) == 0:
            os.remove(filename)
        if to_save:
            self.showed_labels.append(label)
        label.show()

    def setup_corner_images(self):
        """
        Adds corner images to the application window
        :return:
        """
        corner_image = cv2.imread(CORNER_IMAGE_PATH, cv2.IMREAD_UNCHANGED)
        x = 0
        y = 0
        width = int(WINDOW_WIDTH / 4)
        height = int(WINDOW_HEIGHT / 4)
        i = 0
        while True:
            filename = "temporary/{}.png".format(os.getpid())
            cv2.imwrite(filename, corner_image)
            self.show_image(x, y, width, height, path=filename, to_save=False)
            os.remove(filename)

            i += 1
            if i == 4:
                break
            corner_image = cv2.rotate(corner_image, cv2.cv2.ROTATE_90_CLOCKWISE)

            if x == 0:
                x = WINDOW_WIDTH - width
            elif y == 0:
                y = WINDOW_HEIGHT - height
            else:
                x = 0

    def create_start_button(self):
        """
        Sets up the "start" button
        :return:
        """
        self.start_button.setGeometry(WINDOW_WIDTH / 2 - 100, WINDOW_HEIGHT / 2 - 50, WINDOW_WIDTH / 6,
                                      WINDOW_HEIGHT / 8)
        self.start_button.setText('START')

    def create_next_button(self):
        """
        Sets up the "next" button
        :return:
        """
        self.start_button.setGeometry(WINDOW_WIDTH / 2 - 100, WINDOW_HEIGHT - 60, WINDOW_WIDTH / 8,
                                      WINDOW_HEIGHT / 14)
        self.start_button.setText('Next')

    def create_menubar(self):
        """
        Creates the menu bar of the application window
        :return:
        """
        menu = QMenuBar(self)

        file = menu.addMenu("File")

        simple_mode_action = QAction("Simple mode", self)
        simple_mode_action.triggered.connect(self.turn_on_simple_mode)
        file.addAction(simple_mode_action)

        detailed_mode_action = QAction("Detailed mode", self)
        detailed_mode_action.triggered.connect(self.turn_on_detailed_mode)
        file.addAction(detailed_mode_action)

    def widgets(self):
        """
        Sets up the application window
        :return:
        """
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.create_menubar()

    def turn_on_evaluation_mode(self):
        """
        Sets up the EVALUATION mode
        :return:
        """
        self.mode = EVALUATIONAL_MODE
        self.clear_labels()
        self.create_start_button()
        self.set_background()
        self.title = 'Evaluation mode'

    def turn_on_simple_mode(self):
        """
        Sets up the SIMPLE mode
        :return:
        """
        self.mode = SIMPLE_MODE
        self.clear_labels()
        self.create_start_button()
        self.set_background()
        self.title = 'Simple mode'

    def turn_on_detailed_mode(self):
        """
        Sets up DETAILED mode
        :return:
        """
        self.mode = DETAILED_MODE
        self.clear_labels()
        self.create_start_button()
        self.set_background()
        self.title = 'Detailed mode'

    def set_background(self):
        """
        Window's background set up
        :return:
        """
        if self.mode == EVALUATIONAL_MODE:
            self.setStyleSheet("background-color: teal;")
        elif self.mode == SIMPLE_MODE:
            self.setStyleSheet("background-color: white;")
            self.setup_corner_images()
        elif self.mode == DETAILED_MODE:
            self.setStyleSheet("background-color: pink;")
            self.setup_corner_images()
        self.show()
        QGuiApplication.processEvents()

    @pyqtSlot()
    def start_button_action(self):
        """
        Starts recognition
        :return:
        """
        self.create_next_button()
        self.create_filename_list()
        if len(self.filenames_list) == self.filename_indx:
            self.start_button.deleteLater()
            self.clear_labels()
            self.show_emotion(confirm_results(self.get_merged_predictions()), show_text=True)
            self.title = 'Results'
        else:
            img_path = self.filenames_list[self.filename_indx]
            self.process_one_image(name=DEFAULT_IMG_NAME, img_path=img_path)
            prediction_matrix = self.get_merged_predictions()
            results = confirm_results(prediction_matrix)
            if self.mode == SIMPLE_MODE:
                self.show_emotion(results)
            else:
                self.show_expressions(results, prediction_matrix)

            self.filename_indx += 1

    def create_filename_list(self):
        """
        Creates list of images to process
        :return:
        """
        if len(self.filenames_list) == 0:
            for filename in os.listdir(self.search_dir):
                re, ext = os.path.splitext(filename)
                if ext == '.jpg' or ext == '.JPG':
                    self.filenames_list.append(self.search_dir + filename)
            self.filenames_list.sort(key=lambda x: os.path.getmtime(x))
        else:
            self.add_filenames_to_list()

    def add_filenames_to_list(self):
        """
        Adds images to list to process
        :return:
        """
        for filename in os.listdir(self.search_dir):
            re, ext = os.path.splitext(filename)
            path = self.search_dir + filename
            if not path in self.filenames_list and (ext == '.jpg' or ext == '.JPG'):
                self.filenames_list.append(path)
        self.filenames_list.sort(key=lambda x: os.path.getmtime(x))

    def get_merged_predictions(self):
        """
        Merges predictions from files
        :return: matrix with merged results
        """
        prediction_matrix = []
        for i in range(0, len(self.pred_matrixes)):
            prediction_matrix.append([])
            for j in range(0, len(self.pred_matrixes[i])):
                dictionary = self.pred_matrixes[i][j]
                el = get_element_with_the_biggest_value(dictionary)
                if len(el[0]) == 0 and len(dictionary) > 1:
                    el = get_element_with_the_biggest_value(
                        {k: dictionary[k] for k in dictionary if k != el[0]})
                prediction_matrix[i].append(el[0])
        return prediction_matrix

    def show_final_text(self, success_count, defined_count):
        """
        Displays final text
        :param success_count: count of correctly solved expressions
        :param defined_count: count of solved expressions
        :return:
        """
        text = str(success_count) + ' out of ' + str(defined_count) + ' are correct!'
        self.create_text_label(text, FINAL_TEXT_LBL, WINDOW_WIDTH / 3 + 50, WINDOW_HEIGHT / 16 * 15)

    def show_emotion(self, results, show_text=False):
        """
        Display emotion in a simple mode or on the final screen
        :param results:
        :param show_text:
        :return:
        """
        emotion, success_count, defined_count = get_emotion(results)
        if emotion == SMILE:
            img_path = SMILE_PATH
        elif emotion == NEUTRAL:
            img_path = NEUTRAL_PATH
        else:
            img_path = SAD_PATH
        if show_text:
            self.show_final_text(success_count, defined_count)
        self.show_image(WINDOW_WIDTH / 4, WINDOW_HEIGHT / 4, WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2, path=img_path)

    def open_evaluation_files(self):
        self.pred_boxes_file = open(OUTPUT_PATH + 'pred_boxes_percent' + TXT, "w+")
        self.pred_file_merged = open(OUTPUT_PATH + 'pred_percent_merged' + TXT, "w+")
        self.pred_file = open(OUTPUT_PATH + 'pred_percent' + TXT, "w+")
        self.pred_file_equation = open(OUTPUT_PATH + 'pred_percent_equation' + TXT, "w+")
        self.pred_file_handwritten = open(OUTPUT_PATH + 'pred_percent_handwritten' + TXT, "w+")
        self.pred_file_equation_merged = open(OUTPUT_PATH + 'pred_percent_equation_merged' + TXT, "w+")
        self.pred_file_handwritten_merged = open(OUTPUT_PATH + 'pred_percent_handwritten_merged' + TXT, "w+")

        self.boxes_incorrect_file = open(OUTPUT_PATH + 'boxes_incorrect' + TXT, "w+")
        self.pred_incorrect_file = open(OUTPUT_PATH + 'pred_incorrect' + TXT, "w+")
        self.pred_incorrect_merged_file = open(OUTPUT_PATH + 'pred_incorrect_merged' + TXT, "w+")

    def close_evaluation_files(self):
        self.pred_boxes_file.close()
        self.pred_file_merged.close()
        self.pred_file.close()
        self.pred_file_equation.close()
        self.pred_file_handwritten.close()
        self.pred_file_equation_merged.close()
        self.pred_file_handwritten_merged.close()

        self.boxes_incorrect_file.close()
        self.pred_incorrect_file.close()
        self.pred_incorrect_merged_file.close()

    def print_evaluation_results(self):
        merged_imgs_num = NUMBER_OF_IMAGES * NUMBER_IMAGES_PODTYPES
        imgs_num = merged_imgs_num * NUMBER_OF_IMAGE_TYPES

        print_overall_prediction_correctness(self.overall_pred_correctness, imgs_num, '')
        print_overall_prediction_correctness(self.overall_pred_correctness_merged, merged_imgs_num, 'merged')
        print('overall box prediction correctness: ' + str(self.overall_box_pred_correctness / imgs_num))

        print_overall_prediction_correctness(self.overall_pred_correctness_obj[HANDWRITTEN], imgs_num, 'handwritten')
        print_overall_prediction_correctness(self.overall_pred_correctness_obj_merged[HANDWRITTEN],
                                             merged_imgs_num, 'handwritten merged')

        print_overall_prediction_correctness(self.overall_pred_correctness_obj[EQUATIONS], imgs_num, 'equation')
        print_overall_prediction_correctness(self.overall_pred_correctness_obj_merged[EQUATIONS],
                                             merged_imgs_num, 'equation merged')

    def calculate_merged_evaluation(self, i, j):
        prediction_matrix = self.get_merged_predictions()
        approx_pred = compare_predictions(
            str(i) + '_' + str(j), prediction_matrix, self.pred_incorrect_merged_file)
        self.pred_incorrect_merged_file.write(
            ' ' + str(self.objects_number[0]) + ' ' + str(self.objects_number[1]) + '\n')

        self.overall_pred_correctness_merged += print_prediction_percent(
            approx_pred[EQUATIONS] + approx_pred[HANDWRITTEN],
            self.objects_number[HANDWRITTEN] + self.objects_number[EQUATIONS],
            self.pred_file_merged, 'merged')

        self.overall_pred_correctness_obj_merged[EQUATIONS] += print_prediction_percent(
            approx_pred[EQUATIONS], self.objects_number[EQUATIONS],
            self.pred_file_equation_merged, 'merged equation')

        self.overall_pred_correctness_obj_merged[HANDWRITTEN] += print_prediction_percent(
            approx_pred[HANDWRITTEN], self.objects_number[HANDWRITTEN],
            self.pred_file_handwritten_merged, 'merged handwritten')

        print_merged_answer(str(i) + '_' + str(j), prediction_matrix)

    def run_evaluation(self):
        """
        The run of the evaluation
        """
        self.open_evaluation_files()

        if DEBUG_ONE_FILE:
            self.process_one_image(name=DEBUG_FILENAME, img_path=TEST_DATA_PATH + DEBUG_FILENAME + JPG)
            return

        for i in range(0, NUMBER_OF_IMAGES):
            for j in range(0, NUMBER_IMAGES_PODTYPES):
                for k in range(0, NUMBER_OF_IMAGE_TYPES):
                    filename = str(i) + '_' + str(j) + '_' + str(k)
                    self.process_one_image(name=filename, img_path=TEST_DATA_PATH + filename + JPG)

                self.calculate_merged_evaluation(i, j)
                self.pred_matrixes = []

        self.close_evaluation_files()
        self.print_evaluation_results()

    def draw_bounding_boxes(self, image, boxes_classids_pred, are_legitimate, results):
        results_colour = [int(c) for c in self.colors[2]]
        results_indx = 0
        for i in range(0, len(boxes_classids_pred)):
            element = boxes_classids_pred[i]
            box = element[BOX]
            x, y, w, h = box[0], box[1], box[2], box[3]

            class_id = element[CLASS_ID]
            color = [int(c) for c in self.colors[class_id]]
            if are_legitimate[i]:
                prediction = element[PREDICTION]
            else:
                prediction = ''
            draw_rectangle(image, x, y, w, h, color, self.labels[class_id], prediction)
            if class_id == EQUATIONS and results_indx > len(results) > 0:
                put_text(image, get_text_from_result(results[results_indx]), results_colour, x - 20, y + 50, 1.5)
                results_indx += 1

        return image

    def calculate_expression_prediction_accuracy(self, pred_matrix, name):
        pred_matrix_row_count = len(pred_matrix[0])
        self.objects_number = [int(pred_matrix_row_count / 3) * len(pred_matrix),
                               int(pred_matrix_row_count / 3) * 2 * len(pred_matrix)]

        right_count = compare_predictions(name, pred_matrix, self.pred_incorrect_file)
        self.pred_incorrect_file.write(' ' + str(self.objects_number[0]) + ' ' + str(self.objects_number[1]) + '\n')
        self.overall_pred_correctness += print_prediction_percent(right_count[EQUATIONS] + right_count[HANDWRITTEN],
                                                                  self.objects_number[EQUATIONS] + self.objects_number[
                                                                      HANDWRITTEN],
                                                                  self.pred_file, '')
        self.overall_pred_correctness_obj[EQUATIONS] += print_prediction_percent(right_count[EQUATIONS],
                                                                                 self.objects_number[EQUATIONS],
                                                                                 self.pred_file_equation, 'equation')
        self.overall_pred_correctness_obj[HANDWRITTEN] += print_prediction_percent(right_count[HANDWRITTEN],
                                                                                   self.objects_number[HANDWRITTEN],
                                                                                   self.pred_file_handwritten,
                                                                                   'handwritten')

    def add_prediction_matrix(self, boxes, predictions, class_ids, are_legitimate, name):
        """
        Creates matrix out of predicted objects
        similar to the way the objects are located in the image
        :param boxes: the list of bounding boxes
        :param predictions: the list of text predictions
        :param class_ids: the list of class ids
        :param are_legitimate: the list of legitimate equations
        :param name: the name of the image
        :return: the list of results for each equation, bounding boxes, class ids, text predictions
        """
        lines = divide_by_lines(boxes, predictions, class_ids, are_legitimate)
        lines = sort_by_x_coordinate(lines)
        lines = sort_by_y_coordinate(lines)
        boxes_classids_pred = [item for sublist in lines for item in sublist]

        matrix = create_matrix_from_lines(lines)
        if self.mode == EVALUATIONAL_MODE:
            self.calculate_expression_prediction_accuracy(matrix, name)
        pred_matrix_is_empty = False
        if len(self.pred_matrixes) == 0:
            pred_matrix_is_empty = True
        for i in range(0, len(matrix)):
            if pred_matrix_is_empty:
                self.pred_matrixes.append([])
            for j in range(0, len(matrix[i])):
                if pred_matrix_is_empty:
                    self.pred_matrixes[i].append({})
                add_to_dictionary(self.pred_matrixes[i][j], matrix[i][j])
        results = confirm_results(matrix)

        return results, boxes_classids_pred

    def run_object_localisation(self, image):
        """
        Localise the objects
        :param image: the image for the localisation
        :return: predicted bounding boxes, class ids and confidence of the prediction for each object
        """
        height, width = image.shape[:2]

        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.layer_names)
        return extract_boxes_confidences_classids(outputs, width, height)

    def make_prediction(self, image, filename):
        """
        Recognises objects in the image
        :param image: the image for recognition
        :param filename: the name of the image
        :return: the list of predicted bounding boxes,
         class ids, text predictions, correctness of the solved equations
        """
        # in case image has only one channel
        if image.shape[2] == 1:
            image = cv2.merge((image, image, image))

        boxes, confidences, classIDs = self.run_object_localisation(image)
        boxes, classIDs = run_nms_algorithm(boxes, confidences, classIDs)

        predictions, are_legitimate = run_object_detection(image, boxes, classIDs)
        if self.mode == EVALUATIONAL_MODE:
            filename = filename[:-2]
        results, boxes_classids_pred = self.add_prediction_matrix(boxes, predictions, classIDs, are_legitimate,
                                                                  filename)
        return boxes_classids_pred, are_legitimate, results

    def clear_labels(self):
        """
        Removes labels from the aplication window
        :return:
        """
        for label in self.showed_labels:
            label.clear()
        self.showed_labels = []

    def create_text_label(self, text, label_type, x, y):
        """
        Displays a text label
        :param text: the text to display
        :param label_type: the type of the label
        :param x: the x-coordinate of the label
        :param y: the y-coordinate of the label
        :return:
        """
        label = QLabel(text, self)
        if label_type == EXPRESSIONS_LBL or label_type == FINAL_TEXT_LBL:
            label.setStyleSheet("color: #003286;"
                                "font-family: Tekton Pro;"
                                "font-size: 30pt")
        elif label_type == TEXT_RESULTS_LBL:
            label.setStyleSheet("color: #b31c48;"
                                "font-family: Tekton Pro;"
                                "font-size: 20pt")
        label.move(x, y)
        self.showed_labels.append(label)
        label.show()

    def get_text_labels_positions(self, column_number, rows_number):
        """
        Gets the position of expressions in the application window
        :param column_number: the number of expressions columns
        :param rows_number: the number of expressions rows
        :return: the list with x and y coordinates
        """
        xs, ys = [100] * column_number * 3, [50] * rows_number
        column_size = int(WINDOW_WIDTH / column_number) - 10
        small_column_size = int(column_size / 3) - 3
        for i in range(1, column_number * 3):
            if i % 3 == 0:
                xs[i] = xs[i - 3] + column_size
            elif i % 3 == 2:
                xs[i] = xs[i - 1] + 25
            else:
                xs[i] = xs[i - 1] + small_column_size

        row_size = int(WINDOW_HEIGHT / rows_number) - 10
        for i in range(1, rows_number):
            ys[i] = ys[i - 1] + row_size
        return xs, ys

    def show_expressions(self, results, matrix):
        """
        Displays mathematical expressions and the result text
        :param results: the list of results for each equation
        :param matrix: the matrix of predicted objects
        :return:
        """
        self.clear_labels()
        column_number = int(len(matrix[0]) / 3)
        rows_number = len(matrix)
        xs, ys = self.get_text_labels_positions(column_number, rows_number)

        k = 0
        for i in range(0, rows_number):
            new_j = 0
            for j in range(0, column_number * 3):
                if new_j > j:
                    continue
                if len(matrix[i][j]) != 0:
                    self.create_text_label(matrix[i][j], EXPRESSIONS_LBL, xs[j], ys[i])
                elif j % 3 == 0:
                    new_j = j + 3
                if j % 3 == 0:
                    self.create_text_label(get_text_from_result(results[k]), TEXT_RESULTS_LBL, xs[j], ys[i] - 30)
                    k += 1
        QGuiApplication.processEvents()

    def calculate_box_prediction_accuracy(self, boxes_classids_pred, name):
        """
        Calculates accuracy of bounding box prediction
        :param boxes_classids_pred: the list of bounding boxes, class ids and predictions
        :param name: the name of the image for recognition
        :return:
        """
        prediction_compare, len_annotation_boxes, incorrect_box_positions_count = compare_box_predictions(
            boxes_classids_pred, name)
        percent = prediction_compare / len_annotation_boxes
        print('Prediction box comparecent percent: ' + str(percent))
        self.pred_boxes_file.write(str(percent) + '\n')
        missing_boxes_count = len_annotation_boxes - prediction_compare
        print('Incorrect box predicitons: ' + str(incorrect_box_positions_count) + ' ' + str(missing_boxes_count) +
              ' ' + str(len_annotation_boxes))
        self.boxes_incorrect_file.write(str(incorrect_box_positions_count) + ' ' + str(missing_boxes_count) + ' ' +
                                        str(len_annotation_boxes) + '\n')
        self.overall_box_pred_correctness += percent

    def process_one_image(self, name, image=None, img_path=None):
        """
        Recognises objects on one image, draw bounding boxes and saves the image
        :param name: the name of the image
        :param image: the image for recognition
        :param img_path: the path to the image  for recognition
        :return:
        """
        if self.mode == EVALUATIONAL_MODE:
            print(name)

        if image is None:
            image = scan(to_crop=False, file_name=img_path)
        else:
            image = scan(to_crop=True, image=rotate_image(image, 180))

        boxes_classids_pred, are_legitimate, results = self.make_prediction(image, name)

        if self.mode == EVALUATIONAL_MODE:
            self.calculate_box_prediction_accuracy(boxes_classids_pred, name)

        if BOOL_SHOW or BOOL_SAVE:
            image = self.draw_bounding_boxes(image, boxes_classids_pred, are_legitimate, results)

        if BOOL_SHOW:
            cv2.imshow('Output image', image)
            cv2.waitKey(0)

        if BOOL_SAVE:
            cv2.imwrite(OUTPUT_PATH + name + '.jpg', image)


def application():
    """
    Starts application
    :return:
    """
    app = QApplication(sys.argv)
    ex = Application()
    sys.exit(app.exec_())
