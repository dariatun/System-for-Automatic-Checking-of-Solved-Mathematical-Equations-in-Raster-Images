import re
from application.constants import TXT, OUTPUT_PATH


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


def print_overall(file_box, file_pred, file_pred_merged):
    calculate_fo_file(file_box, 3)
    calculate_fo_file(file_pred, 7)
    calculate_fo_file(file_pred_merged, 7)


if __name__ == "__main__":
    file_box = open(OUTPUT_PATH + "boxes_incorrect" + TXT, 'r')
    file_pred = open(OUTPUT_PATH + "pred_incorrect" + TXT, 'r')
    file_pred_merged = open(OUTPUT_PATH + "pred_incorrect_merged" + TXT, 'r')
    print_overall(file_box, file_pred, file_pred_merged)
    print('Finished')
