import os

import numpy as np


def sort_lines(input):
    lines = np.loadtxt(input)
    nb_lines = sum(1 for _ in open(input))

    if os.path.exists('./output/sorted_lines_NFA.txt'):
        os.remove('./output/sorted_lines_NFA.txt')
    if os.path.exists('./output/sorted_lines_length.txt'):
        os.remove('./output/sorted_lines_length.txt')

    if nb_lines == 1:
        sorted_lines_list_length = []
        sorted_lines_list_NFA = []
        sorted_lines_list_length.append(lines)
        sorted_lines_list_NFA.append(lines)
        return sorted_lines_list_length, sorted_lines_list_NFA

    # First sort the line by length
    # length_sorted_lines = lines[lines[:, 4].argsort()]
    length_sorted_lines = np.flip(lines[lines[:, 4].argsort()], axis=0)     # Sorted by length
    NFA_sorted_lines = lines[lines[:, 7].argsort()]     # Sorted by logNFA

    sorted_lines_list_length = []
    sorted_lines_list_length.append(length_sorted_lines[0])
    for line_test in length_sorted_lines:
        is_double = False
        for line in sorted_lines_list_length:
            a = line[5]
            b = line[6]
            a_test = line_test[5]
            b_test = line_test[6]
            if a - 0.2 < a_test < a + 0.2 and b - 10 <= b_test <= b + 10:
                is_double = True
        if not is_double:
            sorted_lines_list_length.append(line_test)

    sorted_lines_list_NFA = []
    sorted_lines_list_NFA.append(NFA_sorted_lines[0])
    for line_test in NFA_sorted_lines:
        is_double = False
        for line in sorted_lines_list_NFA:
            a = line[5]
            b = line[6]
            a_test = line_test[5]
            b_test = line_test[6]
            if a - 0.2 < a_test < a + 0.2 and b - 10 <= b_test <= b + 10:
                is_double = True
        if not is_double:
            sorted_lines_list_NFA.append(line_test)

    return sorted_lines_list_length, sorted_lines_list_NFA

