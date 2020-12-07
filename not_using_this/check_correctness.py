import re
#from handwritten_recogniser import handwritten_recogniser


def correctness(digit_predictions, digit_xy_coords, equations_predictions, equations_xy_coords):

    eq_ranges = [None] * len(equations_xy_coords)
    i = 0
    for eq in equations_xy_coords:
        eq_ranges[i] = [[eq[0] + eq[2], eq[0] + eq[2] + 50], [eq[1] - 10, eq[1] + 10 + eq[3]]]
        i += 1

    together = [None] * len(equations_xy_coords)
    i = 0
    for eq_range in eq_ranges:
        j = 0
        for xy in digit_xy_coords:
            if eq_range[0][0] <= xy[0] <= eq_range[0][1] and eq_range[1][0] <= xy[1] <= eq_range[1][1]:
                together[i] = j
            j += 1
        i += 1

    for i in range(0, len(together)):
        if together[i] is None: continue
        if correctness_one(equations_predictions[i], digit_predictions[together[i]]):
            print('Correct')
        else:
            print('Incorrect')


def correctness_one(equation, digit):
    numbers = list(filter(None, re.split('[=+-]', equation)))
    symbols = list(filter(None, re.split('[0-9]', equation)))
    if symbols[0] == '+':
        return int(numbers[0]) + int(numbers[1]) == int(digit)
    else:
        return int(numbers[0]) - int(numbers[1]) == int(digit)


if __name__ == "__main__":
    digit_predictions, digit_xy_coords, equations_predictions, equations_xy_coords = handwritten_recogniser()

    print('Finished')
