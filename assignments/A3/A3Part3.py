import numpy as np
from scipy.fftpack import fft, fftshift
import math

"""
A3-Part-3: Symmetry properties of the DFT

Write a function to check if the input signal is real and even using the symmetry properties of its
DFT. The function will return the result of this test, the zerophase windowed version of the input 
signal (dftbuffer), and the DFT of the dftbuffer. 

Given an input signal x of length M, do a zero phase windowing of x without any zero-padding (a 
dftbuffer, on the same lines as the fftbuffer in sms-tools). Then compute the M point DFT of the 
zero phase windowed signal and use the symmetry of the computed DFT to test if the input signal x 
is real and even. Return the result of the test, the dftbuffer computed, and the DFT of the dftbuffer. 

The input argument is a signal x of length M. The output is a tuple with three elements 
(isRealEven, dftbuffer, X), where 'isRealEven' is a boolean variable which is True if x is real 
and even, else False. dftbuffer is the M length zero phase windowed version of x. X is the M point 
DFT of the dftbuffer. 

To make the problem easier, we will use odd length input sequence in this question (M is odd). 

Due to the precision of the FFT computation, the zero values of the DFT are not zero but very small
values < 1e-12 in magnitude. For practical purposes, all values with absolute value less than 1e-6 
can be considered to be zero. Use an error tolerance of 1e-6 to compare if two floating point arrays 
are equal. 

Caveat: Use the imaginary part of the spectrum instead of the phase to check if the input signal is 
real and even.

Test case 1: If x = np.array([ 2, 3, 4, 3, 2 ]), which is a real and even signal (after zero phase 
windowing), the function returns (True, array([ 4., 3., 2., 2., 3.]), array([14.0000+0.j, 2.6180+0.j, 
0.3820+0.j, 0.3820+0.j, 2.6180+0.j])) (values are approximate)

Test case 2: If x = np.array([1, 2, 3, 4, 1, 2, 3]), which is not a even signal (after zero phase 
windowing), the function returns (False,  array([ 4.,  1.,  2.,  3.,  1.,  2.,  3.]), array([ 16.+0.j, 
2.+0.69j, 2.+3.51j, 2.-1.08j, 2.+1.08j, 2.-3.51j, 2.-0.69j])) (values are approximate)
"""


def testRealEven(x):
    """
    Inputs:
        x (numpy array)= input signal of length M (M is odd)
    Output:
        The function should return a tuple (isRealEven, dftbuffer, X)
        isRealEven (boolean) = True if the input x is real and even, and False otherwise
        dftbuffer (numpy array, possibly complex) = The M point zero phase windowed version of x 
        X (numpy array, possibly complex) = The M point DFT of dftbuffer 
    """
    # Your code here
    M = x.shape[0]
    N = M
    hN = (N//2)+1
    hM1 = (M+1)//2
    hM2 = int(math.floor(M/2))
    xw = x[0:M]
    fftbuffer = np.zeros(N)
    fftbuffer[:hM1] = xw[hM2:]
    fftbuffer[-hM2:] = xw[:hM2]

    X = fft(fftbuffer)
    #X = abs(X)
    shifted_hX1 = fftshift(X[1:hN])
    hX2 = X[-hM1+1:]
    
    isRealEven = True

    for hx1, hx2 in zip(shifted_hX1, hX2):
        if abs(abs(hx1) - abs(hx2)) > 1e-6 or np.imag(hx1) > 1e-6 or np.imag(hx2) > 1e-6:
            isRealEven = False
            break

    #print(shifted_hX1)
    #print(hX2)

    return isRealEven, fftbuffer, X


def test_case():
    a = 0.5
    f = 100
    fs = 1000
    t = np.arange(0, 1, 1.0/fs)
    x = a * np.cos(2*np.pi*f*t)
    x = np.array([2, 3, 4, 3, 2])
    #x = np.array([1, 2, 3, 4, 1, 2, 3])
    isRealEven, buff, X = testRealEven(x)
    print(isRealEven, buff, X)

#test_case()
