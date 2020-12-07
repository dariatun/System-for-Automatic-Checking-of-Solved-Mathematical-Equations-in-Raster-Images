import os

CORRECT_ANSWER = 'Y'
INCORRECT_ANSWER = 'N'
UNDECIDED_ANSWER = 'U'

# object types
EQUATIONS = 0
HANDWRITTEN = 1

# variables in a bounding box
X = 0
Y = 1
W = 2
H = 3

DEFAULT_SYMBOL_WIDTH = 160

# variables for experimental part
EXPERIMENTAL_PART = False

COMPARE_POSSIBILITIES = True

TEST_DATA_ANNOTATIONS_PATH = '../test_data_annotations/'
TEST_DATA_BOX_ANNOTATIONS_PATH = '../test_data_annotations/000/'
OUTPUT_PATH = '../output/'
TEST_DATA_PATH = '../test_data/'

NUMBER_OF_IMAGE_TYPES = 5
NUMBER_OF_IMAGES = 20
NUMBER_IMAGES_PODTYPES = 2

DEBUG_ONE_FILE = False
DEBUG_FILENAME = '19_1_4'
DEBUG_WITHOUT_PREDICTIONS = False

# file extensions
JPG = '.jpg'
TXT = '.txt'

# model variables
LABELS_PATH = '../model/obj.names'
CONFIG_PATH = '../model/yolov3.cfg'
WEIGHTS_PATH = '../model/yolov3.weights'
BOOL_USE_GPU = False
BOOL_SAVE = True
BOOL_SHOW = False
CONFIDENCE = 0.1
THRESHOLD = 0.05

SAME_LINE_CHARACTER_SPACE = 100

BOX = 0
PREDICTION = 1
CLASS_ID = 2
IS_LEGITIMATE = 3


POSSIBLE_OVERLAP_WIDTH = 50
POSSIBLE_WIDTH_BETWEEN_CLOSE_OBJECTS = 100

# application modes
EXPERIMENTAL_MODE = 0
SIMPLE_MODE = 1
ALL_IN_ONE_MODE = 2

SUCCESS_PROCENT = 0.5

SMILE = 0
NEUTRAL = 1
SAD = 2

EMOTION_PATH = '../images/emotions/'
SMILE_PATH = EMOTION_PATH + 'happy.png'
NEUTRAL_PATH = EMOTION_PATH + 'neutral.png'
SAD_PATH = EMOTION_PATH + 'sad.png'
