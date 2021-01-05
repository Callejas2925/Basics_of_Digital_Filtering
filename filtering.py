"""
Exploring Digital Filters
    Circular Buffers
    1) Circular Buffer is a technique used for emulating a shift register
    2) Circular Buffer uses an array instead of a shift register which makes
       shifting values in the buffer more difficult.
    3) A moving index is used to keep track of the current sample in the buffer
       and samples before the index are considered past samples.
"""
import numpy as np
from matplotlib import pyplot as plt


def lpf(samples, wc):
    taps = [0] * len(samples.tolist())
    for i in range(len(samples)):
        if i == int((len(samples.tolist()) / 2)):
            taps[i] = wc / np.pi
        else:
            taps[i] = np.sin(wc * samples[i]) / (np.pi * samples[i])
    return taps


def fast_fourier_transform(function):
    fft = np.fft.fft(function)
    fft_shift = np.fft.fftshift(fft)
    fft_mag = np.abs(fft_shift)
    return fft_mag


def circular_convolution(input_signal, impulse_response_coefficients):
    cir_buffer = [0] * len(impulse_response_coefficients)
    unfolded_buffer = [0] * len(impulse_response_coefficients)
    mac_buffer = []

    cir_index = 0
    for i in range(len(input_signal)):
        cir_buffer[cir_index] = input_signal[i]

        buffer_len = int(len(cir_buffer))
        for j in range(len(cir_buffer)):
            if cir_index - j < 0:
                unfolded_buffer[j] = cir_buffer[buffer_len - 1] * impulse_response_coefficients[j]
                buffer_len -= 1
            else:
                unfolded_buffer[j] = cir_buffer[cir_index - j] * impulse_response_coefficients[j]

        print(unfolded_buffer)
        mac_buffer.append(sum(unfolded_buffer))

        if cir_index >= len(cir_buffer) - 1:
            cir_index = 0
        else:
            cir_index += 1
    return mac_buffer


def rect(array, size=0.5):
    return np.where(abs(array) <= size, 1, 0).tolist()


def rect_window(size):
    return np.ones(shape=size, dtype=int).tolist()


def plot(x_axis, y_axis):
    plt.plot(x_axis, y_axis)
    plt.show()


fs = 1000  # Sampling Freq
fc = 100  # Cut-off Freq
w = 2 * np.pi * (fc / fs)

t = np.linspace(start=-20, stop=20, num=41, dtype=int)
f = np.linspace(start=0, stop=40, num=41, dtype=int)
ilpf = lpf(samples=t, wc=w)
fft_ilpf = fast_fourier_transform(ilpf)

points = np.linspace(start=-10, stop=10, num=201)

rect_function = rect(array=points)
coeff = rect_window(size=10)

x = circular_convolution(input_signal=rect_function, impulse_response_coefficients=coeff)

plot(x_axis=points, y_axis=x)
