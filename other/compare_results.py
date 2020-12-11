file_1 = open(
    '../different_results/before_merging_two_dictionaries_of_handwritten/pred_percent_handwritten_merged.txt',
    #"../different_results/after_using_github_rep_and_blur/pred_percent_handwritten_merged.txt",
    'r')
file_2 = open(
    "/Users/dariatunina/mach-lerinig/mLStuff/output/pred_percent_handwritten_merged.txt",
    #'/different_results/before_doing_postprocessing_not_using_empty_strings/pred_percent_handwritten_merged.txt',
    'r')

lines_1 = file_1.readlines()
lines_2 = file_2.readlines()

count_better = 0
count_worse = 0
count_eq = 0
for i in range(0, len(lines_1)):
    if abs(float(lines_1[i]) - float(lines_2[i])) < 0.05:
        count_eq += 1
    elif float(lines_1[i]) < float(lines_2[i]):
        count_better += 1
    else:
        count_worse += 1

file_1.close()
file_2.close()

print('Better: ' + str(count_better) + ', equal: ' + str(count_eq) + ', worse: ' + str(count_worse))
