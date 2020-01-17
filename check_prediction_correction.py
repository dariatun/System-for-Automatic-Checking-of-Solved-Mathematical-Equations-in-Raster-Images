import json


if __name__ == "__main__":
    prob_eq = 0
    prob_handwrttn = 0
    count_eq = 0
    count_handwrttn = 0
    with open('result.json') as json_file:
        data = json.load(json_file)
        for file in data:
            for obj in file['objects']:
                if obj['class_id'] == 0:
                    prob_eq += obj['confidence']
                    count_eq += 1
                else:
                    prob_handwrttn += obj['confidence']
                    count_handwrttn += 1
    prob_eq /= count_eq
    prob_handwrttn /= count_handwrttn
    prob_eq *= 100
    prob_handwrttn *= 100
    print('Overall probability equation: {}%, probability handwritten: {}%'.format(prob_eq, prob_handwrttn))
    print('Finished')
