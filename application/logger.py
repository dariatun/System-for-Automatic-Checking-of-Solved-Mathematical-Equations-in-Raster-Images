from application.constants import OUTPUT_PATH, TXT


def print_prediction_percent(approx_pred, all, file, string):
    """
    Prints prediction accuracy percent

    :param approx_pred: count of correctly predicted predictions
    :param all: count of all of the prediction
    :param file: file that accuracy percent is added to
    :param string: type of the prediction
    :return: prediction accuracy percent
    """
    if all == 0:
        pred_percent = 0
    else:
        pred_percent = approx_pred / all
    print('Prediction string accuracy percent ' + string + ': ' + str(pred_percent))
    file.write(str(pred_percent))
    file.write('\n')
    return pred_percent


def print_overall_prediction_correctness(value, dividend, name):
    """
    Prints overall prediction accuracy percent
    :param value: the count of correctly predicted objects
    :param dividend: the count of all objects
    :param name: type of the prediction
    :return:
    """
    print('overall prediction correctness' + name + ': ' + str(value / dividend))


def print_merged_answer(name, matrix):
    """
    Add predicted objects to the file
    :param name: type of the prediction
    :param matrix: matrix of objects predictions
    :return:
    """
    file = open(OUTPUT_PATH + name + '_answer' + TXT, 'w+')
    for i in range(0, len(matrix)):
        for j in range(0, len(matrix[0])):
            file.write(str(matrix[i][j]))
            file.write(' ')
        file.write('\n')
    file.close()