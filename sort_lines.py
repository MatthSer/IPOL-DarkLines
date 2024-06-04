import os

import numpy as np


def sort_lines(input):
    lines = np.loadtxt(input)

    if os.path.exists('./output/sorted_lines.txt'):
        os.remove('./output/sorted_lines.txt')

    # First sort the line by length
    # length_sorted_lines = lines[lines[:, 4].argsort()]
    length_sorted_lines = np.flip(lines[lines[:, -4].argsort()], axis=0)

    sorted_lines_list = []
    sorted_lines_list.append(length_sorted_lines[0])
    for line_test in length_sorted_lines:
        is_double = False
        for line in sorted_lines_list:
            a = line[5]
            b = line[6]
            a_test = line_test[5]
            b_test = line_test[6]
            if a - 0.2 < a_test < a + 0.2 and b - 100 <= b_test <= b + 100:
                is_double = True
        if not is_double:
            sorted_lines_list.append(line_test)

    return sorted_lines_list

