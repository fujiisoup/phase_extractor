import sys
from configparser import ConfigParser
import PIL.Image
import numpy as np
from scipy import signal, optimize


def get_phase(image, config, reference=None):
    """
    retrieve phase from image.
    image: 2d array
    """
    image = image - np.mean(image)
    fft = np.fft.rfft2(image)
    kx = np.fft.fftfreq(image.shape[0])
    ky = np.fft.rfftfreq(image.shape[1])
    
    # ------
    # find the modulation frequency 
    # ------
    # make some low frequency component zero
    fft = np.where((np.abs(kx)[:, np.newaxis] < 0.1) * (np.abs(ky) < 0.2),
                   0, np.abs(fft))
    # maximum of the fourier transform
    idx = np.unravel_index(np.argmax(fft), shape=fft.shape)
    kx_max = kx[idx[0]]
    ky_max = ky[idx[1]]

    roi_fraction = float(config['parameters']['roi_fraction'])
    sl_x = slice(
        int(image.shape[0] * (0.5 - 0.5 * roi_fraction)),
        int(image.shape[0] * (0.5 + 0.5 * roi_fraction))
    )
    sl_y = slice(
        int(image.shape[1] * (0.5 - 0.5 * roi_fraction)),
        int(image.shape[1] * (0.5 + 0.5 * roi_fraction))
    )
    x = np.arange(image.shape[0])[:, np.newaxis]
    y = np.arange(image.shape[1])
    
    parameters = {}

    # maximize the modulation frequency
    def func(p):
        kx, ky = p
        wave = np.exp(2j * np.pi * (kx * x + ky * y))
        return np.mean(np.abs(image * wave)[sl_x, sl_y])
    
    kx_max, ky_max = optimize.minimize(func, x0=(kx_max, ky_max)).x
    wave = np.exp(2j * np.pi * (kx_max * x + ky_max * y))
    parameters['kx'] = kx_max
    parameters['ky'] = ky_max
    
    # convolute the window function
    n_waves = float(config['parameters']['n_waves'])
    n = int(n_waves / np.sqrt(kx_max**2 + ky_max**2))
    window = getattr(signal, config['parameters']['window'])(n)
    window = window[:, np.newaxis] * window
    window = window * n**2 / np.sum(window)

    convolved = signal.convolve(
        image * wave, window, mode='valid', method='direct')
    
    return parameters, np.abs(convolved), np.angle(convolved)


def save_array(array, src_path, suffix):
    # squish array into 0-255 range
    vmin = np.min(array)
    vmax = np.max(array)
    image = PIL.Image.fromarray(
        ((array - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    )
    outpath = src_path[:src_path.rfind('.')] + suffix + '.bmp'
    image.save(outpath, format='bmp')


if __name__ == '__main__':
    if len(sys.argv) not in [2, 3]:
        raise ValueError(
            'Usage: get_phase.py [path/to/image]'
            'or \nget_phase.py [path/to/image] [path/to/reference image]'
        )
    
    config = ConfigParser()
    config.read('settings.ini')

    if len(sys.argv) == 2:
        path = sys.argv[1]
        image = PIL.Image.open(path)
        _, amplitude, phase = get_phase(np.array(image), config, reference=None)
        save_array(amplitude, path, '_amp')
        save_array(phase, path, '_phase')
