"""
Exploring Digital Filters
    Circular Buffers
    1) Circular Buffer is a technique used for emulating a shift register
    2) Circular Buffer uses an array instead of a shift register which makes
       shifting values in the buffer more difficult.
    3) A moving index is used to keep track of the current sample in the buffer
       and samples before the index are considered past samples.
"""
import math
import numpy as np
from matplotlib import pyplot as plt


def lpf_taps(samples, wc):
    taps = [0] * len(samples)
    for i in range(len(samples)):
        if i == int(len(samples) / 2):
            taps[i] = wc / math.pi
        else:
            taps[i] = math.sin(wc * samples[i]) / (math.pi * samples[i])
    return taps


def fft_db(function):
    fft = np.fft.fft(function)
    fft_shift = np.fft.fftshift(fft)
    fft_mag = np.abs(fft_shift)

    db_fft = []
    for i in range(len(fft_mag)):
        db_fft.append(10*math.log10(fft_mag[i]))
    return db_fft


def freq_taps(function, sample_freq):
    taps = []
    for i in range(len(function)):
        taps.append((-sample_freq / 2) + (i * (sample_freq / len(function))))
    return taps


def circular_convolution(input_signal, impulse_response_coefficients, transient=True):
    """

    :param input_signal: Signal that will be flipped and shifted to perform convolution.
    :param impulse_response_coefficients: Static Signal to perform convolution.
    :param transient: Transient Response. Adds zeros to allow the circular buffer to flush properly.
    :return: Convolution Result
    """
    cir_buffer = [0] * len(impulse_response_coefficients)
    unfolded_buffer = [0] * len(impulse_response_coefficients)
    mac_buffer = []

    if transient:
        input_signal = input_signal + [0] * len(cir_buffer)

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

        mac_buffer.append(sum(unfolded_buffer))

        if cir_index >= len(cir_buffer) - 1:
            cir_index = 0
        else:
            cir_index += 1
    return mac_buffer


def sample_taps(size, odd=True):
    if odd:
        if size % 2 == 0:
            size += 1
    taps = [0] * size
    for i in range(size):
        taps[i] = i - int(size / 2)
    return taps


def plot(x_axis, y_axis):
    plt.plot(x_axis, y_axis)
    plt.show()


fs = 1000  # Sampling Freq
fc = 100   # Cut-off Freq
w = 2 * math.pi * (fc / fs)

N = 40
t = sample_taps(size=N)
impulse = [0]*1001
impulse[int(len(impulse)/2)] = 1

ilpf = lpf_taps(samples=t, wc=w)
con_ilpf = circular_convolution(input_signal=impulse, impulse_response_coefficients=ilpf)
fft_ilpf = fft_db(con_ilpf)

f = freq_taps(function=fft_ilpf, sample_freq=fs)

plt.plot(f, fft_ilpf)
plt.show()
