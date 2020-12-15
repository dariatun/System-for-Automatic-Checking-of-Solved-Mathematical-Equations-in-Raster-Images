import re
from application.constants import TXT, OUTPUT_PATH
from utils.utils import add_to_dictionary


def add_to_count(count, numbers):
    for i in range(0, len(count)):
        count[i] += numbers[i]


def calculate_fo_file(file, len):
    count = [0] * len
    for line in file.readlines():
        numbers = re.split('[/" /"/\n]', line)
        numbers = list(filter(None, numbers))
        numbers = list(map(int, numbers))
        add_to_count(count, numbers)
    print(count)


def add_numbers_to_dict(numbers, points):
    sum = 0
    for i in range(0, len(numbers)):
        sum += numbers[i] * points[i]
    if len(numbers) == 3:
        return sum / numbers[2]
    else:
        return sum / (numbers[-1] + numbers[-2])


def get_numbers(line):
    numbers = re.split('[/" /"/\n]', line)
    numbers = list(filter(None, numbers))
    numbers = list(map(int, numbers))
    return numbers


def calculate_for_one(file_box, points_box, file_pred, points_pred):
    dict = {}
    lines_box = file_box.readlines()
    lines_pred = file_pred.readlines()
    for i in range(0, len(lines_pred)):
        numbers_box = get_numbers(lines_box[i])
        numbers_pred = get_numbers(lines_pred[i])
        sum_box = add_numbers_to_dict(numbers_box, points_box)
        sum_pred = add_numbers_to_dict(numbers_pred, points_pred)
        add_to_dictionary(dict, sum_box + sum_pred)
    new_dict = {0: 0, 1: 0, 2: 0, 3: 0}
    for k, v in sorted(dict.items()):
        if k < 1:
            new_dict[0] += v
        elif k < 2:
            new_dict[1] += v
        elif k < 3:
            new_dict[2] += v
        else:
            new_dict[3] += v

    for k, v in sorted(new_dict.items()):
        print(k, v)


def print_dict(file_box, file_pred, file_pred_merged):
    calculate_for_one(file_box, [5, 5, 0], file_pred, [1, 2, 3, 3, 4, 3, 0, 0])
    # calculate_for_one(file_box, [5, 5, 0], file_pred_merged, [1, 2, 3, 3, 4, 3, 0, 0])


def print_overall(file_box, file_pred, file_pred_merged):
    calculate_fo_file(file_box, 3)
    calculate_fo_file(file_pred, 7)
    calculate_fo_file(file_pred_merged, 7)


if __name__ == "__main__":
    file_box = open(OUTPUT_PATH + "boxes_incorrect" + TXT, 'r')
    file_pred = open(OUTPUT_PATH + "pred_incorrect" + TXT, 'r')
    file_pred_merged = open(OUTPUT_PATH + "pred_incorrect_merged" + TXT, 'r')
    print_dict(file_box, file_pred, file_pred_merged)
    print('Finished')
