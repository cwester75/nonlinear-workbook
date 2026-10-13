"""Transcribed mathematical/data fixtures, with printed book page references."""
import numpy as np

FIVE_PATTERNS = np.array([[1, 1, 1, 1, -1], [-1, 1, -1, 1, 1]])  # p.354

# pp.360-361, row-major 8x5, +1 black and -1 white; no inferred glyph redraw.
DIGIT_PATTERNS = np.array([
    [-1,-1,1,-1,-1, -1,1,1,-1,-1, -1,-1,1,-1,-1, -1,-1,1,-1,-1,
     -1,-1,1,-1,-1, -1,-1,1,-1,-1, -1,-1,1,-1,-1, -1,-1,-1,-1,-1],
    [-1,1,1,1,-1, 1,-1,-1,-1,1, -1,-1,-1,-1,1, -1,-1,1,1,-1,
     -1,1,-1,-1,-1, 1,-1,-1,-1,-1, 1,1,1,1,1, -1,-1,-1,-1,-1],
    [1,-1,-1,1,-1, 1,-1,-1,1,-1, 1,-1,-1,1,-1, 1,1,1,1,1,
     -1,-1,-1,1,-1, -1,-1,-1,1,-1, -1,-1,-1,1,-1, -1,-1,-1,-1,-1]
])
DIGIT_LABELS = (1, 2, 4)
DIGIT_CORRUPTED = np.array([  # p.361 starting configuration
    1,1,-1,1,-1, -1,1,-1,1,1, 1,-1,-1,1,-1, 1,-1,1,1,1,
    -1,-1,-1,-1,-1, -1,-1,1,1,-1, -1,-1,1,1,-1, -1,-1,-1,-1,-1
])

KOHONEN_INPUTS = np.array([[1,1,0,0], [0,0,0,1], [1,0,0,0], [0,0,1,1.]])  # p.374
KOHONEN_INITIAL = np.array([[.2,.6,.5,.9], [.8,.4,.7,.3]])  # neuron rows, book columns

CITIES = np.array([[10.1,20.2], [4,7], [.1,2], [.5,10.7], [10.5,.6],
                   [8,10], [12,16], [18,8.]])  # p.383
BOOK_TOUR = np.array([0,3,5,1,2,4,7,6,0])

BOOLEAN_INPUTS = np.array([[0,0], [0,1], [1,0], [1,1.]])
XOR_TARGETS = np.array([[0], [1], [1], [0.]])
# p.414: book's first column is a bias weight, here represented separately.
XOR_HIDDEN_WEIGHTS = np.array([[.5,-.5], [-.3,.7]])
XOR_HIDDEN_BIAS = np.array([-.2,-.3])
XOR_OUTPUT_WEIGHTS = np.array([[.3,.5]])
XOR_OUTPUT_BIAS = np.array([-.2])
