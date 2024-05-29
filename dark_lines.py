import iio
import numpy as np
import scipy
from math import floor
import os
import time

import argparse

import cv2
from numba import njit
import numba as nb


def convert_to_grey(img):
    grey_value = [1 / 3, 1 / 3, 1 / 3]
    # grey_value = [0.2125, 0.7154, 0.0721 ]
    grey_scale = np.matmul(img, grey_value)
    return grey_scale


def is_local_minimum(img, x, y):
    """Check if the element at position (x, y) is a local minimum."""
    rows = len(img)
    cols = len(img[0])

    # Get the value at the current position
    current_value = img[x, y]

    # List of relative positions of all possible neighbors
    neighbors = [
        (-1, -1), (-1, 0), (-1, 1),  # Top-left, Top, Top-right
        (0, -1), (0, 1),  # Left,          Right
        (1, -1), (1, 0), (1, 1)  # Bottom-left, Bottom, Bottom-right
    ]

    # neighbors = [(0, -1), (0, 1)]
    # neighbors = [(-1, 0), (0, -1), (0, 1), (1, 0)]

    for dx, dy in neighbors:
        nx, ny = x + dx, y + dy
        if 0 <= nx < rows and 0 <= ny < cols:
            if img[nx][ny] <= current_value:
                return False
    return True


def find_local_minimum(img):
    """Find all local minima in the 2D array."""
    local_minimum = np.zeros_like(img)
    list_local_min = []
    rows = len(img)
    cols = len(img[0])

    for i in range(rows):
        for j in range(cols):
            if is_local_minimum(img, i, j):
                local_minimum[i][j] = True
                list_local_min.append((i, j))

    return local_minimum, list_local_min


@njit
def interpolate(img, x, y):
    xx = floor(x)
    yy = floor(y)
    cx = x - floor(x)
    cy = y - floor(y)
    a00 = img[xx, yy]
    a01 = img[xx + 1, yy]
    a10 = img[xx, yy + 1]
    a11 = img[xx + 1, yy + 1]

    # Bilinear interpolation
    return (a00 * (1.0 - cx) * (1.0 - cy)
            + a01 * cx * (1.0 - cy)
            + a10 * (1.0 - cx) * cy
            + a11 * cx * cy)


@njit
def log_nfa_dark_lines(img, sigma, rho, x1, y1, x2, y2):
    X = img.shape[0]
    Y = img.shape[1]
    k = 0
    """" 
        number of test (NT) - every possible line segment in the image
        up to a precision sigma in the endpoints
    """
    log_nt = (2.0 * np.log10(X) - 2.0 * np.log10(sigma)
              + 2.0 * np.log10(Y) - 2.0 * np.log10(sigma))

    # Compute the tilt of the vector
    dx = x2 - x1
    dy = y2 - y1
    length = np.sqrt(dx * dx + dy * dy)
    if length <= 0.0:
        return log_nt
    dx = dx / length
    dy = dy / length

    """
    The line is evaluated with the following schema:

      A   A   A   A   A   A   A   A   A   A
      |   |   |   |   |   |   |   |   |   |
      C---C---C---C---C---C---C---C---C---C
      |   |   |   |   |   |   |   |   |   |
      B   B   B   B   B   B   B   B   B   B

     The C corresponds to the places along the line in which it is evaluated.
     The A and B correspond to the lateral measures to which the central
     values (Cs) are compared.  One central value is counted as supporting
     the line when C<A and C<B.  The lines | or --- correspond to a distance
     2sigma.
    """
    n = floor(length / sigma / 2.0)
    for i in range(n):
        """evaluate A-C-B, k count the number of cases in which
         C is darker than its two neighbors"""
        xc = x1 + i * 2.0 * sigma * dx
        yc = y1 + i * 2.0 * sigma * dy
        xa = x1 + i * 2.0 * sigma * dx - 2.0 * sigma * dy
        ya = y1 + i * 2.0 * sigma * dy + 2.0 * sigma * dx
        xb = x1 + i * 2.0 * sigma * dx + 2.0 * sigma * dy
        yb = y1 + i * 2.0 * sigma * dy - 2.0 * sigma * dx

        C = interpolate(img, xc, yc)
        A = interpolate(img, xa, ya)
        B = interpolate(img, xb, yb)

        if C < A and C < B:
            k += 1

    """
    compute NFA = NT x P( K >= k ), where K is a Binomial random variable
                                 of parameters n and rho
                                 
    probability term: when k/n < rho is not an interesting case, p=1
                   when k=n then the binomial tail is rho^n
                   the other cases are bounded by Hoeffding's inequality:
                   p <= (rho*n/k)^k * ( (n-n*rho) / (n-k) )^(n-k)
                   
    actually the code computes log10(NFA), which is easily obtained
    """
    kk = k
    nn = n
    nk = (n - k)
    if nn == 0:
        log_p = 0
    elif kk / nn < rho:
        log_p = 0.0
    elif n == k:
        log_p = nn * np.log10(rho)
    else:
        log_p = (kk * np.log10(rho) + kk * np.log10(nn) - kk * np.log10(kk)
                 + nk * np.log10(1.0 - rho) + nk * np.log10(nn) - nk * np.log10(nk))

    log_nfa = log_nt + log_p

    return log_nfa


@njit
def test_points(blurred_img, sigma, rho, list_local_min):
    list_line = []
    for i in nb.prange(len(list_local_min)):
        x1, y1 = list_local_min[i]
        for x2, y2 in list_local_min:
            if x1 != x2 or y1 != y2:
                log_nfa = log_nfa_dark_lines(blurred_img, sigma, rho, x1, y1, x2, y2)
                if log_nfa < 0.0:
                    list_line.append((x1, y1, x2, y2))
    return list_line


def main(input, sigma, noise_level, rho):

    # Read input image and convert to grey scale
    img = iio.read(input)
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, img.shape)
        img = img + noise
    if img.shape[2] == 3:
        grey_scale = convert_to_grey(img)
    else:
        grey_scale = img

    # Apply gaussian blur
    start_local_minimum = time.time()
    blurred_img = scipy.ndimage.gaussian_filter(grey_scale, sigma=sigma)

    # Compute the local minimum of the image
    local_minimum, list_local_min = find_local_minimum(blurred_img)
    time_local_minimum = time.time() - start_local_minimum

    # Overwrite output file
    if not os.path.exists('./output'):
        os.mkdir('./output')
    if os.path.exists('./output/lines.txt'):
        os.remove('./output/lines.txt')

    # Create output image with detected lines
    output = np.copy(img)

    # Compute log NFA
    start = time.time()
    with open('./output/lines.txt', 'a') as file:
        list_lines = test_points(blurred_img, sigma, rho, list_local_min)
        for x1, y1, x2, y2 in list_lines:
            cv2.line(output, (y1, x1), (y2, x2), (255, 0, 0), 2)
            file.write(f'{y1} {x1} {y2} {x2}\n')
    compute_time = time.time() - start

    # Number of lines found
    nb_lines = sum(1 for _ in open('./output/lines.txt'))

    # Print computation times
    print(f'Computation time for local minima: {time_local_minimum:.2f} s')
    print(f'Computation time for searching lines: {compute_time:.2f} s')
    print(f'Number of lines found: {nb_lines} lines')

    # Write outputs
    iio.write('./output/local_minimum.png', (local_minimum * 255).astype(np.uint8))
    iio.write('./output/output.png', output.astype(np.uint8))

    return exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='./inputs/test.png')
    parser.add_argument('-s', '--sigma', type=float, required=False, default=4.5)
    parser.add_argument('-n', '--noise_level', type=int, required=False, default=0)
    parser.add_argument('-r', '--rho', type=float, required=False, default=1 / 3)
    args = parser.parse_args()
    main(args.input, args.sigma, args.noise_level, args.rho)

    # img_path = './inputs/test.png'
    # rho = 1 / 3
    # main(img_path, rho)
