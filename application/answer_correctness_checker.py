from application.constants import UNDECIDED_ANSWER, CORRECT_ANSWER, INCORRECT_ANSWER, NEUTRAL, SUCCESS_PERCENT, SMILE, \
    SAD
from application.utils import get_numbers_and_delimiter


def get_handwritten_answer(handwritten_number_1, handwritten_number_2):
    """
    Calculates handwritten answer
    :param handwritten_number_1: first handwritten digit
    :param handwritten_number_2: second handwritten digit
    :return: calculated answer
    """
    if handwritten_number_1 == '' and handwritten_number_2 == '':
        return -1
    elif handwritten_number_2 == '':
        answer = int(handwritten_number_1)
    elif handwritten_number_1 == '':
        answer = int(handwritten_number_2)
    else:
        answer = int(handwritten_number_1) * 10 + int(handwritten_number_2)
    return answer


def get_correct_answer(equation):
    """
    Calculate the answer to the equation
    :param equation: equation to calculate
    :return: answer to the equation
    """
    numbers, symbol = get_numbers_and_delimiter(equation)

    if symbol == '+':
        correct_answer = int(numbers[0]) + int(numbers[1])
    else:
        correct_answer = int(numbers[0]) - int(numbers[1])

    return correct_answer


def check_answer_of_one_equation(equation, handwritten_number_1, handwritten_number_2):
    """
    Check if the given answer is correct answer for the equation
    :param equation: equation to check
    :param handwritten_number_1: first handwritten digit
    :param handwritten_number_2: second handwritten digit
    :return: the decided correctness of the answer
    """
    if len(equation) == 0:
        return UNDECIDED_ANSWER

    correct_answer = get_correct_answer(equation)
    answer = get_handwritten_answer(handwritten_number_1, handwritten_number_2)

    if answer == -1:
        return UNDECIDED_ANSWER
    elif correct_answer == answer:
        return CORRECT_ANSWER
    else:
        return INCORRECT_ANSWER


def confirm_results(prediction_matrix):
    """
    Check if the given answers are correct answers for all of the equation
    :param prediction_matrix: matrix of the predicted objects
    :return: list of the results
    """
    result_list = []
    for i in range(0, len(prediction_matrix)):
        for j in range(0, len(prediction_matrix[i]), 3):
            answer_result = check_answer_of_one_equation(prediction_matrix[i][j],
                                                         prediction_matrix[i][j + 1],
                                                         prediction_matrix[i][j + 2])
            result_list.append(answer_result)

    return result_list


def get_emotion(results):
    """
    Get emotion that will be shown based on the calculated results
    :param results: list of all of the expressions results
    :return: type of the emotion
             number of the correct answers
             number of the expressions with an answer
    """
    success_count = 0
    undefined_count = 0
    defined_count = 0
    for i in range(0, len(results)):
        if results[i] == CORRECT_ANSWER:
            success_count += 1
            defined_count += 1
        elif results[i] == UNDECIDED_ANSWER:
            undefined_count += 1
        else:
            defined_count += 1

    overall = len(results)
    if overall == 0 or defined_count == 0:
        return NEUTRAL, success_count, defined_count
    elif success_count / defined_count > SUCCESS_PERCENT:
        return SMILE, success_count, defined_count
    else:
        return SAD, success_count, defined_count


def get_text_from_result(result):
    """
    Get text that will be displayed near the expression
    :param result: expression's result
    :return: chosen text
    """
    if result == CORRECT_ANSWER:
        text = 'Answer is correct!'
    elif result == INCORRECT_ANSWER:
        text = 'Check this, one more time!'
    else:
        text = 'Keep going, you\'re doing great!'
    return text
