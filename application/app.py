import sys
import time
import cv2
import numpy as np

from application.functions import print_prediction_percent, divide_by_lines, sort_by_x_coordinate, sort_by_y_coordinate, \
    create_matrix_from_lines, print_overall_prediction_correctness, confirm_results, get_emotion, \
    get_modified_by_indxes, extract_boxes_confidences_classids, print_merged_answer
from application.scaner import scan
from application.expimental_part_functions import compare_predictions, compare_box_predictions
from utils.utils import add_to_dictionary, get_element_with_the_biggest_value, rotate_img_opencv, resize_image
from application.image_manipulation import recognise_object

from PIL import Image
import threading
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QMenuBar, QAction
from PyQt5.QtGui import QIcon, QPixmap, QGuiApplication


from application.constants import *


class App(QWidget):

    def __init__(self):
        super().__init__()
        self.title = 'Application'
        self.left = 10
        self.top = 10
        self.width = 800
        self.height = 600

        self.widgets()
        self.mode = SIMPLE_MODE
        self.set_background()

        # Get the labels
        self.labels = open(LABELS_PATH).read().strip().split('\n')

        # Create a list of colors for the labels
        self.colors = np.random.randint(0, 255, size=(len(self.labels), 3), dtype='uint8')

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

        self.fps_label = QLabel('FPS: ---------------------------', self)
        self.fps_label.move(0, 0)

        leave_button = QPushButton("End loop", self)
        leave_button.move(10, 550)
        leave_button.clicked.connect(self.end_loop)

        start_button = QPushButton("Start loop", self)
        start_button.move(10, 500)
        start_button.clicked.connect(self.start_tryout)

        self.leave_loop = False

        self.objects = []

        self.rows = 0
        self.columns = 0

        self.pred_matrixes = []
        self.FROM_PHONE = True

        self.objects_number = [0, 0]

        self.pred_boxes_file = None
        self.pred_file_merged = None
        self.pred_file = None
        self.pred_file_equation = None
        self.pred_file_handwritten = None
        self.pred_file_equation_merged = None
        self.pred_file_handwritten_merged = None

        self.overall_pred_correctness_merged = 0
        self.overall_pred_correctness = 0
        self.overall_box_pred_correctness = 0
        self.overall_pred_correctness_obj = [0, 0]
        self.overall_pred_correctness_obj_merged = [0, 0]
        self.cap = cv2.VideoCapture(0)

        self.show()

    def show_image(self, image=None, path=''):
        label = QLabel(self)
        filename = path
        if len(path) == 0:
            filename = "{}.png".format(os.getpid())
            cv2.imwrite(filename, image)

        pixmap = QPixmap(filename)
        label.setPixmap(pixmap.scaledToWidth(300))
        label.move(100, 100)
        if len(path) == 0:
            os.remove(filename)
        label.show()

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

        exitAction = QAction("Exit", self)
        exitAction.triggered.connect(self.client_exit)
        file.addAction(exitAction)

    def turn_on_experimental_mode(self):
        self.mode = EXPERIMENTAL_MODE
        self.set_background()

    def turn_on_simple_mode(self):
        self.mode = SIMPLE_MODE
        self.set_background()

    def turn_on_all_mode(self):
        self.mode = ALL_IN_ONE_MODE
        self.set_background()

    def set_background(self):
        if self.mode == EXPERIMENTAL_MODE:
            self.setStyleSheet("background-color: teal;")
        elif self.mode == SIMPLE_MODE:
            self.setStyleSheet("background-color: pink;")
        elif self.mode == ALL_IN_ONE_MODE:
            self.setStyleSheet("background-color: white;")

    def client_exit(self):
        exit()

    @pyqtSlot()
    def end_loop(self):
        self.leave_loop = True

    @pyqtSlot()
    def start_tryout(self):
        threading.Thread(target=self.start_loop()).start()

    def start_loop(self):
        self.leave_loop = True
        # self.clock()
        if self.mode == EXPERIMENTAL_MODE:
            self.go_through_phone_imgs()
        else:
            self.camera_images_loop()

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

    def camera_images_loop(self):
        #cap = cv2.VideoCapture(0)

        # self.clock(name=DEBUG_FILENAME, img_path=TEST_DATA_PATH + DEBUG_FILENAME + JPG)
        time.sleep(5)
        _, frame = self.cap.read()
        self.clock(name='capture', image=frame)

        prediction_matrix = self.get_approximate_prediction()
        results = confirm_results(prediction_matrix)

        if self.mode == SIMPLE_MODE:
            emotion = get_emotion(results)
            if emotion == SMILE:
                img_path = SMILE_PATH
            elif emotion == NEUTRAL:
                img_path = NEUTRAL_PATH
            else:
                img_path = SAD_PATH
            self.show_image(path=img_path)
            self.pred_matrixes = []

    def go_through_phone_imgs(self):
        self.pred_boxes_file = open(OUTPUT_PATH + 'pred_boxes_percent' + TXT, "w+")
        self.pred_file_merged = open(OUTPUT_PATH + 'pred_percent_merged' + TXT, "w+")
        self.pred_file = open(OUTPUT_PATH + 'pred_percent' + TXT, "w+")
        self.pred_file_equation = open(OUTPUT_PATH + 'pred_percent_equation' + TXT, "w+")
        self.pred_file_handwritten = open(OUTPUT_PATH + 'pred_percent_handwritten' + TXT, "w+")
        self.pred_file_equation_merged = open(OUTPUT_PATH + 'pred_percent_equation_merged' + TXT, "w+")
        self.pred_file_handwritten_merged = open(OUTPUT_PATH + 'pred_percent_handwritten_merged' + TXT, "w+")

        if DEBUG_ONE_FILE:
            self.clock(name=DEBUG_FILENAME, img_path=TEST_DATA_PATH + DEBUG_FILENAME + JPG)
        else:

            for i in range(0, NUMBER_OF_IMAGES):
                for j in range(0, NUMBER_IMAGES_PODTYPES):
                    for k in range(0, NUMBER_OF_IMAGE_TYPES):
                        filename = str(i) + '_' + str(j) + '_' + str(k)
                        self.clock(name=filename, img_path=TEST_DATA_PATH + filename + JPG)

                    prediction_matrix = self.get_approximate_prediction()
                    approx_pred = compare_predictions(str(i) + '_' + str(j), prediction_matrix)

                    print_prediction_percent(approx_pred[EQUATIONS] + approx_pred[HANDWRITTEN],
                                             self.objects_number[HANDWRITTEN] + self.objects_number[EQUATIONS],
                                             self.overall_pred_correctness_merged, self.pred_file_merged, 'merged')

                    print_prediction_percent(approx_pred[EQUATIONS], self.objects_number[EQUATIONS],
                                             self.overall_pred_correctness_obj_merged[EQUATIONS],
                                             self.pred_file_equation_merged, 'merged equation')

                    print_prediction_percent(approx_pred[HANDWRITTEN], self.objects_number[HANDWRITTEN],
                                             self.overall_pred_correctness_obj_merged[HANDWRITTEN],
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

    def get_prediction(self, image, x, y, w, h, class_id):
        filename = "{}.jpg".format(os.getpid())
        cv2.imwrite(filename, image)
        prediction, box, is_legitimate = recognise_object(np.array(Image.open(filename)), x, y, w, h, class_id,
                                                          self.FROM_PHONE)
        os.remove(filename)
        return prediction, box, is_legitimate

    def prediction_object(self, image, boxes, class_ids):
        predictions = [''] * len(boxes)
        are_legitimate = [True] * len(boxes)
        for indx in range(0, len(boxes)):
            box = boxes[indx]
            predictions[indx], boxes[indx], are_legitimate[indx] = self.get_prediction(image, box[0], box[1], box[2],
                                                                                       box[3], class_ids[indx])

        return predictions, are_legitimate

    def draw_rectangle(self, image, x, y, w, h, colour, class_id, prediction):
        cv2.rectangle(image, (x, y), (x + w, y + h), colour, 2)
        text = "{} ({})".format(self.labels[class_id], prediction)
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.9, color=colour, thickness=2)

    def draw_bounding_boxes(self, image, boxes, confidences, classIDs, colors, predictions, are_legitimate):
        for i in range(0, len(boxes)):
            # extract bounding box coordinates
            x, y = boxes[i][0], boxes[i][1]
            w, h = boxes[i][2], boxes[i][3]

            # draw the bounding box and label on the image
            color = [int(c) for c in colors[classIDs[i]]]
            if DEBUG_WITHOUT_PREDICTIONS:
                prediction = ''
            else:
                if are_legitimate[i]:
                    prediction = predictions[i]
                else:
                    prediction = ''
            self.draw_rectangle(image, x, y, w, h, color, classIDs[i], prediction)

        return image

    def add_pred_to_file(self, pred_matrix, name):
        pred_matrix_row_count = len(pred_matrix[0])
        self.objects_number = [int(pred_matrix_row_count / 3) * len(pred_matrix),
                               int(pred_matrix_row_count / 3) * 2 * len(pred_matrix)]
        right_count = compare_predictions(name, pred_matrix)
        print_prediction_percent(right_count[EQUATIONS] + right_count[HANDWRITTEN],
                                 self.objects_number[EQUATIONS] + self.objects_number[HANDWRITTEN],
                                 self.overall_pred_correctness, self.pred_file, '')
        print_prediction_percent(right_count[EQUATIONS], self.objects_number[EQUATIONS],
                                 self.overall_pred_correctness_obj[EQUATIONS], self.pred_file_equation, 'equation')
        print_prediction_percent(right_count[HANDWRITTEN], self.objects_number[HANDWRITTEN],
                                 self.overall_pred_correctness_obj[HANDWRITTEN], self.pred_file_handwritten,
                                 'handwritten')

    def add_prediction_matrix(self, boxes, predictions, class_ids, are_legitimate, name):
        lines = divide_by_lines(boxes, predictions, class_ids, are_legitimate)
        lines = sort_by_x_coordinate(lines)
        lines = sort_by_y_coordinate(lines)
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

    def make_prediction(self, image, filename):
        if image.shape[2] == 1:
            image = cv2.merge((image, image, image))
        height, width = image.shape[:2]

        # Create a blob and pass it through the model
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.layer_names)

        # Extract bounding boxes, confidences and classIDs
        boxes, confidences, classIDs = extract_boxes_confidences_classids(outputs, CONFIDENCE, width, height)

        # Apply Non-Max Suppression
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, THRESHOLD)

        boxes, confidences, classIDs = get_modified_by_indxes(boxes, confidences, classIDs, idxs)
        if DEBUG_WITHOUT_PREDICTIONS:
            predictions = None
        else:
            predictions, are_legitimate = self.prediction_object(image, boxes, classIDs)
            self.add_prediction_matrix(boxes, predictions, classIDs, are_legitimate, filename[:-2])
        return boxes, confidences, classIDs, predictions, are_legitimate

    def clock(self, name, image=None, img_path=None):
        self.frame_id += 1
        print(name)

        if image is None:
            image = scan(to_crop=False, file_name=img_path)
        else:
            image = scan(to_crop=True, image=rotate_img_opencv(image, 180))

        boxes, confidences, classIDs, predictions, are_legitimate = self.make_prediction(image, name)

        if self.mode == EXPERIMENTAL_MODE:
            prediction_compare = compare_box_predictions(boxes, classIDs, name)
            print('Prediction box comparecent percent: ' + str(prediction_compare))
            self.pred_boxes_file.write(str(prediction_compare))
            self.pred_boxes_file.write('\n')
            self.overall_box_pred_correctness += prediction_compare

        image = self.draw_bounding_boxes(image, boxes, confidences, classIDs, self.colors, predictions, are_legitimate)

        # show the output image
        if BOOL_SHOW:
            cv2.imshow('YOLO Object Detection', image)
            cv2.waitKey(0)

        if BOOL_SAVE:
            cv2.imwrite(OUTPUT_PATH + name + '.jpg', image)
        elapsed_time = time.time() - self.starting_time
        fps = self.frame_id / elapsed_time
        # var = StringVar()
        # label = Label(root, textvariable=var, relief=RAISED)
        self.fps_label.setText("FPS: " + str(fps))
        self.fps_label.show()

        if self.mode == EXPERIMENTAL_MODE:
            self.show_image(image=image)
            QGuiApplication.processEvents()
        # self.var.set("FPS: " + str(fps))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
