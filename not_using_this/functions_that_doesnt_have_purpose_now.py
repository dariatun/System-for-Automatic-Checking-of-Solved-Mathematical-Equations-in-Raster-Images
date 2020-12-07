import json
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from application.handwritten_recogniser import recognise_handwritten_image
from application.recognise_text import recognise_text
from utils import get_xy_wh, cut_image


SAVE_EACH_NUMBER = False
INPUTS_FROM_STDIN = False


def recognises_all_digits(objects, init_img_arr, filename, is_from_phone):
    """ Prediction is done by taking all of the objects from one image

    :param objects: array of objects in the image
    :param init_img_arr: initial image
    :param filename: name of the image file
    :return:
    """
    handwrttn_imgs = None
    handwrttn_xy_coords = []
    equations_xy_coords = []
    equations_predictions = []
    init_img = Image.fromarray(init_img_arr)
    #init_img = cv2.cvtColor(np.float32(init_img), cv2.COLOR_BGR2RGB)
    draw = ImageDraw.Draw(init_img)

    for obj in objects:
        if obj['confidence'] < 0.5:
            continue
        xy, w, h = get_xy_wh(obj['relative_coordinates'], init_img_arr.shape)
        image = cut_image(xy[0] - 12, xy[1] - 12, w + 24, h + 24, init_img_arr)
        #image = cut_image(xy[0] , xy[1] , w , h , init_img_arr)
        if image.size == 0:
            continue
        image = rbg_image_to_grey(image)
        if obj['class_id'] == 0:
            prediction = recognise_text(image, is_from_phone)
            if len(prediction) == 0: continue
            draw.rectangle([xy, (w + xy[0], h + xy[1])], outline="green")
            equations_xy_coords.append([xy, h, w])
            equations_predictions.append(prediction)
        else:
            draw.rectangle([xy, (w + xy[0], h + xy[1])], outline="blue")

            handwrttn_xy_coords.append([xy, h])
            image = prepare_handwritten_image(image)
            if handwrttn_imgs is None:
                handwrttn_imgs = image
            else:
                handwrttn_imgs = np.append(handwrttn_imgs, image, axis=0)
    predictions = None
    if handwrttn_imgs is not None:
        _, predictions = recognise_handwritten_image(handwrttn_imgs)
        plot_full_image(predictions, handwrttn_xy_coords, draw)
    if len(equations_predictions) > 0:
        plot_full_image(equations_predictions, equations_xy_coords, draw)
    init_img.save('out/' + filename + '_rec.jpg')
    print('added ' + filename + '.jpg')
    return predictions, handwrttn_xy_coords, equations_predictions, equations_xy_coords

def recognise_one_image_at_a_time(objects, img, is_from_phone):
    """ Prediction is done by one image at a time
    :param objects: array of objects in the image
    :param img: initial image
    :return:
    """
    for obj in objects:

        xy, w, h = get_xy_wh(obj['relative_coordinates'], img.shape)
        image = cut_image(xy[0], xy[1], w, h, img)
        image = rbg_image_to_grey(image)
        if obj['class_id'] == 0:
            prediction = recognise_text(image, is_from_phone)
            if len(prediction) == 0:
                continue
            plot_single_digit1(image, prediction)

        else:
            image, prediction = recognise_handwritten_image(prepare_handwritten_image(image))
            plot_single_digit(image, prediction)


def plot_single_digit1(image, prediction):
    """ Plots image with the predicted label

    :param image:
    :param prediction:
    :return:
    """
    plt.title('Predicted Label: {0}'.format(prediction))
    plt.imshow(image, interpolation='nearest', cmap='gray')
    plt.show()


def plot_single_digit(image, prediction):
    """ Plots image with the predicted label

    :param image:
    :param prediction:
    :return:
    """
    two_d = (np.reshape(image, (28, 28))).astype(np.uint8)
    plt.title('Predicted Label: {0}'.format(prediction))
    plt.imshow(two_d, interpolation='nearest', cmap='gray')
    plt.show()


def plot_full_image(predictions, xy_coords, draw):
    """ Plots initial image with written on it labels and bounding boxes around objects

    :param predictions: predicted labels
    :param xy_coords: coordinates of the predictions
    :param draw: ImageDraw module, that allows putting label on an image
    :return:
    """
    for i in range(0, len(predictions)):
        font = ImageFont.truetype("fonts/arial.ttf", int(xy_coords[i][1] * 0.5))
        draw.text(xy=xy_coords[i][0], text=str(predictions[i]), fill=(255), font=font)


def handwritten_recogniser():
    if INPUTS_FROM_STDIN:
        path_to_image_folder = input('Enter path to the folder with images: ')
        path_to_json_file = input('Enter path to the json file: ')
    else:
        path_to_image_folder = '/phone_images/'  #'/Users/dariatunina/mach-lerinig/' #'/Users/dariatunina/mach-lerinig/mLStuff/'
        path_to_json_file = '/result.json'  #'/Users/dariatunina/mach-lerinig/mLStuff/result.json'
    with open(path_to_json_file) as json_file:
        data = json.load(json_file)
        for file in data:
            filename = file['filename'].split('/')[-1]
            try:

                img = np.array(Image.open(path_to_image_folder + filename))
            except Exception as e:
                continue
            if SAVE_EACH_NUMBER:
                recognise_one_image_at_a_time(file['objects'], img, False)
            else:
                return recognises_all_digits(file['objects'], img, filename[0:-4], False)
    print('Finished')


"""height, width, channels = frame.shape

# Detecting objects
blob = cv2.dnn.blobFromImage(image=frame, scalefactor=1 / 255.0, size=(416, 416), mean=0, swapRB=True,
                             crop=False)

self.net.setInput(blob)
outs = self.net.forward(self.output_layers)

# Showing information on the screen
objects = []
new_pred_matrix = []
prev_coord = [0, 0]
prev_row_indx = 0
prev_col_indx = 0
prev_class_id = -1
first = True
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        # print(class_id, confidence)
        if confidence > 0.2:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            box = [x, y, w, h]
            objects.append(Object(box, class_id, float(confidence), '', UNDECIDED_ANSWER))

objects = sorted(objects, key=lambda k: [k.box[1], k.box[0]])
prev_obj = None
"""
"""
for obj in objects:
    class_id = obj.class_id
    box = obj.box
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]
    filename = "{}.jpg".format(os.getpid())
    cv2.imwrite(filename, frame)
    prediction = recognise_object(np.array(Image.open(filename)), x, y, w, h, class_id, self.FROM_PHONE)
    os.remove(filename)
    if prediction is None or prediction == "":
        objects.remove(obj)
        continue
    else:
        # answers.append(UNDECIDED_ANSWER)
        box = [x, y, w, h]
        if class_id == 0:
            obj.prediction = prediction
            if first:
                new_pred_matrix.append([prediction, ''])
                prev_col_indx = 1
                first = False
            else:
                if abs(prev_coord[0] - box[0]) > abs(prev_coord[1] - box[1]):
                    new_pred_matrix.append([prediction, ''])
                    prev_row_indx += 1
                    prev_col_indx = 1
                else:
                    new_pred_matrix[prev_row_indx].append(prediction)
                    new_pred_matrix[prev_row_indx].append('')
                    prev_col_indx += 2
            prev_coord = [box[0] + box[2], box[1]]
            prev_class_id = 0
            prev_obj = obj

        else:
            # if self.mode is TEACHER_MODE:
            if prev_row_indx == 0 or prev_col_indx == 0 or prev_class_id == -1:
                objects.remove(obj)
                continue
            if 0 < prev_coord[0] - box[0] < 50 and abs(prev_coord[1] - box[1]) < 10:
                if prev_class_id == 0:
                    obj.prediction = prediction

                    new_pred_matrix[prev_row_indx][prev_col_indx] = prediction
                    answer = correctness_one(new_pred_matrix[prev_row_indx][prev_col_indx - 1], prediction)
                    obj.corr_answer = (CORRECT_ANSWER if answer else INCORRECT_ANSWER)
                    prev_coord = [box[0] + box[2], box[1]]
                    prev_obj = obj
                    prev_class_id = 1
                else:
                    prev_obj.box = self.append_box(obj.box, box)
                    prev_obj.prediction += prediction

                    new_pred_matrix[prev_row_indx][prev_col_indx] = prev_obj.prediction
                    answer = correctness_one(new_pred_matrix[prev_row_indx][prev_col_indx - 1],
                                             prev_obj.prediction)
                    prev_obj.corr_answer = (CORRECT_ANSWER if answer else INCORRECT_ANSWER)
                    objects.remove(obj)
            else:
                objects.remove(obj)
        # else:
        #    obj.prediction = prediction

indexes = cv2.dnn.NMSBoxes(self.get_boxes(objects), self.get_conf(objects), 0.4, 0.3)

self.objects = objects

for i in range(len(objects)):
    if i in indexes:
        x, y, w, h = objects[i].box
        if self.mode is not self.TEACHER_MODE:
            color = self.colors[objects[i].class_id]
            prediction = objects[i].prediction
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.rectangle(frame, (x, y), (x + w, y + 30), color, -1)
            cv2.putText(frame, str(prediction), (x, y + 30), self.font, 3, (255, 255, 255), 3)
"""
"""if objects[i].class_id == 1:
            color = self.colors[2]
            cv2.rectangle(frame, (x + w, y), (x + w * 2, y + h), color, 2)
            cv2.rectangle(frame, (x + w, y), (x + w * 2, y + 30), color, -1)
            cv2.putText(frame, objects[i].corr_answer, (x + w, y + 30), self.font, 3, (255, 255, 255), 3)
        """
"""
elapsed_time = time.time() - self.starting_time
fps = self.frame_id / elapsed_time
cv2.imwrite("out_images/" + name + ".jpg", frame)

load = Image.open("out_images/" + name + ".jpg")
load = load.resize((600, 300), Image.ANTIALIAS)
render = ImageTk.PhotoImage(load)

images = Label(image=render)
images.image = render
images.place(x=0, y=100)

if self.mode is not self.TEACHER_MODE:
    var = StringVar()
    label = Label(root, textvariable=var, relief=RAISED)
    self.var.set("FPS: " + str(fps))

if not self.leave_loop:
    root.after(10000, self.clock)  # run itself again after 1000 ms
"""

"""
def predict_object(self, objects, image):
    for obj in objects:
        class_id = obj.class_id
        box = obj.box
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        filename = "{}.jpg".format(os.getpid())
        cv2.imwrite(filename, image)
        prediction = recognise_object(np.array(Image.open(filename)), x, y, w, h, class_id, self.FROM_PHONE)
        os.remove(filename)
        if prediction is None or prediction == "":
            objects.remove(obj)
            continue
        else:
            # answers.append(UNDECIDED_ANSWER)
            box = [x, y, w, h]
            if class_id == 0:
                obj.prediction = prediction
                if first:
                    new_pred_matrix.append([prediction, ''])
                    prev_col_indx = 1
                    first = False
                else:
                    if abs(prev_coord[0] - box[0]) > abs(prev_coord[1] - box[1]):
                        new_pred_matrix.append([prediction, ''])
                        prev_row_indx += 1
                        prev_col_indx = 1
                    else:
                        new_pred_matrix[prev_row_indx].append(prediction)
                        new_pred_matrix[prev_row_indx].append('')
                        prev_col_indx += 2
                prev_coord = [box[0] + box[2], box[1]]
                prev_class_id = 0
                prev_obj = obj

            else:
                # if self.mode is TEACHER_MODE:
                if prev_row_indx == 0 or prev_col_indx == 0 or prev_class_id == -1:
                    objects.remove(obj)
                    continue
                if 0 < prev_coord[0] - box[0] < 50 and abs(prev_coord[1] - box[1]) < 10:
                    if prev_class_id == 0:
                        obj.prediction = prediction

                        new_pred_matrix[prev_row_indx][prev_col_indx] = prediction
                        answer = correctness_one(new_pred_matrix[prev_row_indx][prev_col_indx - 1], prediction)
                        obj.corr_answer = (CORRECT_ANSWER if answer else INCORRECT_ANSWER)
                        prev_coord = [box[0] + box[2], box[1]]
                        prev_obj = obj
                        prev_class_id = 1
                    else:
                        prev_obj.box = self.append_box(obj.box, box)
                        prev_obj.prediction += prediction

                        new_pred_matrix[prev_row_indx][prev_col_indx] = prev_obj.prediction
                        answer = correctness_one(new_pred_matrix[prev_row_indx][prev_col_indx - 1],
                                                 prev_obj.prediction)
                        prev_obj.corr_answer = (CORRECT_ANSWER if answer else INCORRECT_ANSWER)
                        objects.remove(obj)
                else:
                    objects.remove(obj)
"""

# else:
#    obj.prediction = prediction




def append_box(prev_b, b):
    new_b = [0] * 4
    new_b[0] = prev_b[0]
    new_b[1] = prev_b[1] if prev_b[1] < b[1] else b[1]
    new_b[2] = prev_b[2] + b[2]
    new_b[3] = prev_b[3] if prev_b[3] < b[3] else b[3]
    return new_b


def get_boxes(objects):
    boxes = []
    for obj in objects:
        boxes.append(obj.box)
    return boxes


def get_conf(objects):
    boxes = []
    for obj in objects:
        boxes.append(obj.confidence)
    return boxes


def prediction_makes_sense(prediction):
    if len(prediction) == 0: return False
    numbers = re.split('[=+-]', prediction)
    delimeter = re.split('[0-9 ]', prediction)
    number_1 = int(numbers[0])
    number_2 = int(numbers[1])

    if delimeter[0] == '-' and number_1 < number_2: return False
    if delimeter[0] == '+' and number_1 + number_2 >= 100: return False

    return True




@dataclass
class Object:
    box: []
    class_id: int
    confidence: float
    prediction: []
    corr_answer: UNDECIDED_ANSWER
    indx: int


@dataclass
class TextFileObject:
    box: []
    prediction: []



def recognise_text(image, from_phone):
    lines, lines_size = get_lines_predictions(image)
    # if from_phone:    lines_size = 2

    first = {}
    second = {}
    delimiter = {}

    for i in range(0, lines_size):
        lines[i] = lines[i].replace('\n', '')

        symbol = get_delimeter(lines[i])

        numbers = re.split('[=+-]', lines[i])
        numbers = list(filter(None, numbers))
        size = len(numbers)
        if size == 0:
            continue
        elif size == 1:
            if is_digit(lines[i][0]):
                add_number(numbers[0], first, second, symbol)
            else:
                find_second(numbers[0], second, symbol)
        elif size >= 2:
            find_first(numbers[0], first)
            find_second(numbers[1], second, symbol)

        add_delimiter_to_dict(delimiter, symbol)

    if DEBUG:
        print(lines)
        print(first, delimiter, second)
    text = ''
    f = [k for k, v in first.items() if v == max(first.values())]
    s = [k for k, v in second.items() if v == max(second.values())]
    if len(first) > 0 and len(delimiter) > 0 and len(second) > 0:
        text = find_max_list(f) + max(delimiter, key=delimiter.get) + find_max_list(s) + '='
    if DEBUG:
        print(text)
    return text


def scan(file_name):
    image = cv2.imread(file_name)
    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image = imutils.resize(image, height=500)

    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grey = cv2.GaussianBlur(grey, (5, 5), 0)
    edged = cv2.Canny(grey, 75, 200)

    print('Step 1: Edge Detection')
    #cv2.imshow("Image", image)
    #cv2.imshow("Edged", edged)
    cv2.imwrite('edged.jpg', edged)
    #cv2.waitKey(0)
    cv2.destroyAllWindows()

    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    screenCnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02*peri, True)

        if len(approx) == 4:
            screenCnt = approx
            break
    if screenCnt is None:
        return 0
    print('Step 2: Find contours of paper')
    cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
    #cv2.imshow("Outline", image)
    cv2.imwrite('countour.jpg', image)
    #cv2.waitKey(0)
    cv2.destroyAllWindows()

    warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

    # warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    # T = threshold_local(warped, 11, offset=10, method="gaussian")
    # warped = (warped > T).astype("uint8") * 255

    print('Step 3: Apply perspective transform')
    #cv2.imshow("Original", imutils.resize(orig, height=650))
    #cv2.imshow("Scanned", imutils.resize(warped, height=650))
    cv2.imwrite('capture.jpg', warped)
    #cv2.waitKey(0)
    return 1

    def check_around(self, image, obj, prediction):
        pred_len = len(prediction)
        symbol_width = int(obj.box[2] / pred_len)
        left_symbol_pred = self.get_prediction(image, obj.box[0] - 2 * symbol_width, obj.box[1],
                                               obj.box[2] + 2 * symbol_width,
                                               obj.box[3], EQUATIONS)
        right_symbol_pred = self.get_prediction(image, obj.box[0], obj.box[1], obj.box[2] + symbol_width, obj.box[3],
                                                EQUATIONS)
        return obj.box[0] - symbol_width, obj.box[1], obj.box[2] + symbol_width, obj.box[3], left_symbol_pred, obj.box[
            0], obj.box[1], obj.box[2] + symbol_width, obj.box[3], right_symbol_pred

    def check_around_xywh(self, image, x, y, w, h, prediction):
        pred_len = len(prediction)
        symbol_width = DEFAULT_SYMBOL_WIDTH
        symbol_width_multiplier = 2
        if not pred_len == 0: symbol_width = int(w / pred_len)
        left_x = x - symbol_width_multiplier * symbol_width
        if left_x < 0: left_x = 0

        left_symbol_pred = self.get_prediction(image, left_x, y, w + x - left_x, h, EQUATIONS)
        if len(left_symbol_pred) > pred_len:
            return [left_x, y, w + symbol_width * symbol_width_multiplier, h], left_symbol_pred

        right_symbol_pred = self.get_prediction(image, x, y, w + symbol_width * symbol_width_multiplier, h, EQUATIONS)
        if len(right_symbol_pred) > pred_len:
            return [x, y, w + symbol_width * symbol_width_multiplier, h], right_symbol_pred

        return [x, y, w, h], prediction
