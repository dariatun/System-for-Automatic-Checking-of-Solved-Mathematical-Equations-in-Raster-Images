import argparse
import sys
import time
import cv2
import numpy as np

from application.functions import print_prediction_percent, divide_by_lines, sort_by_x_coordinate, sort_by_y_coordinate, \
    create_matrix_from_lines, print_overall_prediction_correctness, confirm_results, get_emotion, \
    get_modified_by_indxes, extract_boxes_confidences_classids, print_merged_answer, get_text_from_result, \
    prediction_object
from application.scaner import scan
from application.expimental_part_functions import compare_predictions, compare_box_predictions
from utils.utils import add_to_dictionary, get_element_with_the_biggest_value, rotate_img_opencv
from application.outlined_label import OutlinedLabel
from PyQt5.QtCore import pyqtSlot, Qt
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QMenuBar, QAction
from PyQt5.QtGui import QPixmap, QGuiApplication, QLinearGradient, QGradient, QColor, QFont

from application.constants import *


class App(QWidget):

    def __init__(self):
        super().__init__()
        self.title = 'Application'
        self.left = 10
        self.top = 10
        self.width = WINDOW_WIDTH
        self.height = WINDOW_HEIGHT
        self.widgets()
        self.mode = SIMPLE_MODE

        parser = argparse.ArgumentParser()
        parser.add_argument('-p', '--path', type=str, default=DEFAULT_PROGRAM_IMGS_PATH,
                            help='Path to the images for recognition')
        args = parser.parse_args()
        self.search_dir = args.path
        # QFontDatabase.addApplicationFont("../fonts/BubblegumSans-Regular.ttf")
        # Get the labels
        self.labels = open(LABELS_PATH).read().strip().split('\n')

        # Create a list of colors for the labels
        self.colors = np.random.randint(0, 255, size=(len(self.labels) + 1, 3), dtype='uint8')

        # Load weights using OpenCV
        self.net = cv2.dnn.readNetFromDarknet(CONFIG_PATH, WEIGHTS_PATH)

        if BOOL_USE_GPU:
            print('Using GPU')
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        if BOOL_SAVE:
            os.makedirs('../output', exist_ok=True)

        # Get the ouput layer names
        self.layer_names = self.net.getLayerNames()
        self.layer_names = [self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        self.font = cv2.FONT_HERSHEY_PLAIN
        self.starting_time = time.time()
        self.frame_id = 0

        self.fps_label = None

        self.start_button = QPushButton("START", self)
        self.start_button.setFont(QFont("Tekton Pro", 50))
        self.start_button.setStyleSheet("background-color: #b31c48;"
                                        "border: 5px solid;"
                                        "border-color: #003286;"
                                        "color: #ffda00")
        self.start_button.clicked.connect(self.start_tryout)

        self.create_start_button()

        self.pred_matrixes = []

        self.objects_number = [0, 0]

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

        self.show()

    def show_image(self, x, y, width, height, image=None, path='', to_save=True):
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

    def setup_fps_label(self):
        self.fps_label = QLabel('FPS: ---------------------------', self)
        self.fps_label.move(0, 0)

    def setup_corner_images(self):
        corner_image = cv2.imread(CORNER_IMAGE_PATH, cv2.IMREAD_UNCHANGED)
        x = 0
        y = 0
        width = int(WINDOW_WIDTH / 6)
        height = int(WINDOW_HEIGHT / 6)
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
        self.start_button.setGeometry(WINDOW_WIDTH / 2 - 100, WINDOW_HEIGHT / 2 - 50, WINDOW_WIDTH / 6,
                                      WINDOW_HEIGHT / 8)
        self.start_button.setText('START')
        self.start_button.show()

    def create_next_button(self):
        self.start_button.setGeometry(WINDOW_WIDTH / 2 - 100, WINDOW_HEIGHT - 60, WINDOW_WIDTH / 8,
                                      WINDOW_HEIGHT / 14)
        self.start_button.setText('Next')

    def widgets(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        menu = QMenuBar(self)

        file = menu.addMenu("File")
        expModeAction = QAction("Experimental mode", self)
        expModeAction.triggered.connect(self.turn_on_experimental_mode)
        file.addAction(expModeAction)

        simpleModeAction = QAction("Simple mode", self)
        simpleModeAction.triggered.connect(self.turn_on_simple_mode)
        file.addAction(simpleModeAction)

        allModeAction = QAction("All in one mode", self)
        allModeAction.triggered.connect(self.turn_on_all_mode)
        file.addAction(allModeAction)

        exitAction = QAction("Exity", self)
        exitAction.triggered.connect(self.client_exit)
        file.addAction(exitAction)

    def turn_on_experimental_mode(self):
        self.mode = EXPERIMENTAL_MODE
        self.clear_labels()
        self.setup_fps_label()
        self.create_start_button()
        self.set_background()

    def turn_on_simple_mode(self):
        self.mode = SIMPLE_MODE
        self.clear_labels()
        self.create_start_button()
        self.set_background()

    def turn_on_all_mode(self):
        self.mode = ALL_IN_ONE_MODE
        self.clear_labels()
        self.create_start_button()
        self.set_background()

    def set_background(self):
        if self.mode == EXPERIMENTAL_MODE:
            self.setStyleSheet("background-color: teal;")
        elif self.mode == SIMPLE_MODE:
            self.setStyleSheet("background-color: pink;")
            self.setup_corner_images()
        elif self.mode == ALL_IN_ONE_MODE:
            self.setStyleSheet("background-color: white;")
            self.setup_corner_images()
        self.show()
        QGuiApplication.processEvents()

    @pyqtSlot()
    def client_exit(self):
        exit()

    @pyqtSlot()
    def start_tryout(self):
        self.create_next_button()
        self.start_loop()

    def start_loop(self):
        if self.mode == EXPERIMENTAL_MODE:
            self.go_through_phone_imgs()
        else:
            self.create_filename_list()
            if len(self.filenames_list) == self.filename_indx:
                self.start_button.deleteLater()
                self.clear_labels()
                self.show_emotion(confirm_results(self.get_approximate_prediction()), show_text=True)
            else:
                img_path = self.filenames_list[self.filename_indx]
                print(img_path)
                self.clock(name=DEFAULT_IMG_NAME, img_path=img_path)
                prediction_matrix = self.get_approximate_prediction()
                results = confirm_results(prediction_matrix)
                if self.mode == SIMPLE_MODE:
                    self.show_emotion(results)
                else:
                    self.show_expressions(results, prediction_matrix)

                self.filename_indx += 1

    def create_filename_list(self):
        if len(self.filenames_list) == 0:
            #os.chdir(self.search_dir)
            for filename in os.listdir(self.search_dir):
                re, ext = os.path.splitext(filename)
                if ext == '.jpg' or ext == '.JPG':
                    self.filenames_list.append(self.search_dir + filename)
            self.filenames_list.sort(key=lambda x: os.path.getmtime(x))
        else:
            self.add_filenames_to_list()

    def add_filenames_to_list(self):
        for filename in os.listdir(self.search_dir):
            re, ext = os.path.splitext(filename)
            path = self.search_dir + filename
            if not path in self.filenames_list and (ext == '.jpg' or ext == '.JPG'):
                self.filenames_list.append(path)
        self.filenames_list.sort(key=lambda x: os.path.getmtime(x))

    def get_approximate_prediction(self):
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
        text = str(success_count) + ' out of ' + str(defined_count) + ' are correct!'
        label = QLabel(text, self)
        label.setStyleSheet("color: #003286;"
                            "font-family: Tekton Pro;"
                            "font-size: 30pt")
        label.move(WINDOW_WIDTH / 3 + 50, WINDOW_HEIGHT / 16 * 15)
        self.showed_labels.append(label)
        label.show()

    def show_emotion(self, results, show_text=False):
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

    def go_through_phone_imgs(self):
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

        if DEBUG_ONE_FILE:
            self.clock(name=DEBUG_FILENAME, img_path=TEST_DATA_PATH + DEBUG_FILENAME + JPG)
        else:

            for i in range(0, NUMBER_OF_IMAGES):
                for j in range(0, NUMBER_IMAGES_PODTYPES):
                    for k in range(0, NUMBER_OF_IMAGE_TYPES):
                        filename = str(i) + '_' + str(j) + '_' + str(k)
                        self.clock(name=filename, img_path=TEST_DATA_PATH + filename + JPG)

                    prediction_matrix = self.get_approximate_prediction()
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
                    self.pred_matrixes = []

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

        self.end_loop()

    def draw_rectangle(self, image, x, y, w, h, colour, class_id, prediction):
        cv2.rectangle(image, (x, y), (x + w, y + h), colour, 2)
        text = "{} ({})".format(self.labels[class_id], prediction)
        self.put_text(image, text, colour, x, y - 5, font_scale=0.9)

    def put_text(self, image, text, colour, x, y, font_scale):
        cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, color=colour, thickness=2)

    def draw_bounding_boxes(self, image, boxes_classids_pred, confidences, are_legitimate, results):
        results_colour = [int(c) for c in self.colors[2]]
        results_indx = 0
        for i in range(0, len(boxes_classids_pred)):
            # extract bounding box coordinates
            element = boxes_classids_pred[i]
            box = element[BOX]
            x, y, w, h = box[0], box[1], box[2], box[3]

            class_id = element[CLASS_ID]
            # draw the bounding box and label on the image
            color = [int(c) for c in self.colors[class_id]]
            # if self.mode == EXPERIMENTAL_MODE:
            if are_legitimate[i]:
                prediction = element[PREDICTION]
            else:
                prediction = ''
            self.draw_rectangle(image, x, y, w, h, color, class_id, prediction)
            if class_id == EQUATIONS and results_indx > len(results) > 0:
                self.put_text(image, get_text_from_result(results[results_indx]), results_colour, x - 20, y + 50, 1.5)
                results_indx += 1

        return image

    def add_pred_to_file(self, pred_matrix, name):
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
        lines = divide_by_lines(boxes, predictions, class_ids, are_legitimate)
        lines = sort_by_x_coordinate(lines)
        lines = sort_by_y_coordinate(lines)
        boxes_classids_pred = [item for sublist in lines for item in sublist]

        matrix = create_matrix_from_lines(lines)
        if self.mode == EXPERIMENTAL_MODE:
            self.add_pred_to_file(matrix, name)
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

        if self.mode == EXPERIMENTAL_MODE:  # or self.mode == ALL_IN_ONE_MODE:
            self.show_expressions(results, matrix)
        return results, boxes_classids_pred

    def make_prediction(self, image, filename):
        if image.shape[2] == 1:
            image = cv2.merge((image, image, image))
        height, width = image.shape[:2]

        # Create a blob and pass it through the model
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.layer_names)

        # Extract bounding boxes, confidences and classIDs
        boxes, confidences, classIDs = extract_boxes_confidences_classids(outputs, width, height)

        # Apply Non-Max Suppression
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, THRESHOLD)

        boxes, confidences, classIDs = get_modified_by_indxes(boxes, confidences, classIDs, idxs)

        predictions, are_legitimate = prediction_object(image, boxes, classIDs)
        results, boxes_classids_pred = self.add_prediction_matrix(boxes, predictions, classIDs, are_legitimate,
                                                                  filename[:-2])
        return boxes_classids_pred, confidences, are_legitimate, results

    """
    def paintEvent(self, e):
        painter = QPainter(self)
        painter.setPen(QPen(Qt.black, 5, Qt.SolidLine))
        # painter.setBrush(QBrush(Qt.red, Qt.SolidPattern))
        painter.setBrush(QBrush(Qt.white, Qt.SolidPattern))
        painter.drawRect(50, 50, WINDOW_WIDTH - 100, WINDOW_HEIGHT - 200)
    """

    def clear_labels(self):
        for label in self.showed_labels:
            label.clear()
        self.showed_labels = []

    def show_expressions(self, results, matrix):
        self.clear_labels()
        # self.draw_board()
        column_number = int(len(matrix[0]) / 3)
        rows_number = len(matrix)
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

        k = 0
        for i in range(0, rows_number):
            new_j = 0
            for j in range(0, column_number * 3):
                if new_j > j:
                    continue
                if len(matrix[i][j]) != 0:
                    label = QLabel(matrix[i][j], self)
                    label.setStyleSheet("color: #003286;"
                                        "font-family: Tekton Pro;"
                                        "font-size: 30pt")
                    label.move(xs[j], ys[i])
                    self.showed_labels.append(label)
                    label.show()
                elif j % 3 == 0:
                    new_j = j + 3
                if j % 3 == 0:
                    results_label = QLabel(get_text_from_result(results[k]), self)
                    results_label.move(xs[j], ys[i] - 30)
                    results_label.setStyleSheet("color: #b31c48;"
                                                "font-family: Tekton Pro;"
                                                "font-size: 20pt")
                    results_label.show()
                    self.showed_labels.append(results_label)
                    k += 1
        QGuiApplication.processEvents()

    def clock(self, name, image=None, img_path=None):
        self.frame_id += 1
        print(name)

        if image is None:
            image = scan(to_crop=False, file_name=img_path)
        else:
            image = scan(to_crop=True, image=rotate_img_opencv(image, 180))

        boxes_classids_pred, confidences, are_legitimate, results = self.make_prediction(image, name)

        if self.mode == EXPERIMENTAL_MODE:
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

        image = self.draw_bounding_boxes(image, boxes_classids_pred, confidences, are_legitimate, results)

        # show the output image
        if BOOL_SHOW:
            cv2.imshow('YOLO Object Detection', image)
            cv2.waitKey(0)

        if BOOL_SAVE:
            cv2.imwrite(OUTPUT_PATH + name + '.jpg', image)
        if self.mode == EXPERIMENTAL_MODE:
            elapsed_time = time.time() - self.starting_time
            fps = self.frame_id / elapsed_time

            self.fps_label.setText("FPS: " + str(fps))
            self.fps_label.show()

        # if self.mode == EXPERIMENTAL_MODE:
        #    self.show_image(image=image)
        #    QGuiApplication.processEvents()

        # self.var.set("FPS: " + str(fps))


def application():
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
