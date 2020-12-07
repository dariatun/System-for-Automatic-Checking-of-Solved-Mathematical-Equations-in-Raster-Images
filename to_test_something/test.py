import re
import os
import numpy as np

prediction_matrix = np.zeros((5, 6))

"""
line = '= 6 6+9='
numbers = re.split('[=+-/" /"]', line)
print(numbers)

numbers = list(filter(None, numbers))
print(numbers)


symbols = re.split('[0-9 ]', line)
print(symbols)

symbols = list(filter(None, symbols))
print(symbols)
"""

os.rename('/Users/dariatunina/mach-lerinig/mLStuff/out_images/capture.jpg', 'herapture.jpg')
