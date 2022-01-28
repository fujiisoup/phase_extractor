import sys
from configparser import ConfigParser
import os
import glob
import PIL.Image
import numpy as np
from scipy import signal, optimize


def get_phase(image, config, parameters=None):
    """
    retrieve phase from image.
    image: 2d array
    """
    image = image - np.mean(image)
    x = np.arange(image.shape[0])[:, np.newaxis]
    y = np.arange(image.shape[1])

    if parameters is None:
        fft = np.fft.rfft2(image)
        kx = np.fft.fftfreq(image.shape[0])
        ky = np.fft.rfftfreq(image.shape[1])

        # ------
        # find the modulation frequency 
        # ------
        # make some low frequency component zero
        fft = np.where(
            (np.abs(kx)[:, np.newaxis] < 0.1) * (np.abs(ky) < 0.1),
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
        
        parameters = {}

        # maximize the modulation frequency
        def func(p):
            kx, ky = p
            wave = np.exp(2j * np.pi * (kx * x + ky * y))
            return -np.mean(np.abs(image * wave)[sl_x, sl_y])
        
        optimize_method = config['parameters']['optimize_method'].strip()
        if optimize_method != 'none':
            result = optimize.minimize(
                func, x0=(kx_max, ky_max),
                method=optimize_method)
            kx_max, ky_max = result.x
        parameters['kx'] = kx_max
        parameters['ky'] = ky_max
    
    kx_max, ky_max = parameters['kx'], parameters['ky']
    wave = np.exp(2j * np.pi * (kx_max * x + ky_max * y))
    
    # convolute the window function
    n_waves = float(config['parameters']['n_waves'])
    n = int(n_waves / np.sqrt(kx_max**2 + ky_max**2))
    window = getattr(signal, config['parameters']['window'])(n)
    window = window[:, np.newaxis] * window
    window = window * n**2 / np.sum(window)

    convolved = signal.convolve(
        image * wave, window, mode='valid', method='direct')
    
    return parameters, convolved


def get_phase_from_reference(image, source):
    """
    Retrieve phase from an image with the aid of the source image.
    """
    return np.angle(image * np.exp(-1j * np.angle(source)))


def save_array(array, src_path, suffix, image_format='bmp'):
    if isinstance(image_format, (list, tuple)):
        for img_format in image_format:
            save_array(array, src_path, suffix, image_format=img_format)
        return
        
    outpath = src_path[:src_path.rfind('.')] + suffix + '.' + image_format
    if image_format in ['bmp', 'png']:
        # squish array into 0-255 range
        vmin = np.min(array)
        vmax = np.max(array)
        image = PIL.Image.fromarray(
            ((array - vmin) / (vmax - vmin) * 255).astype(np.uint8)
        )
        image.save(outpath, format=image_format)
    elif image_format == 'csv':
        np.savetxt(outpath, array, fmt='%10.5f', delimiter=',')


def save_video(arrays, src_path, suffix, video_format='avi'):
    raise NotImplementedError

def _video2images(video, config):
    raise NotImplementedError

__VIDEO_EXTENSIONS__ = ['.avi', '.mp4']

if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise ValueError(
            'Usage: get_phase.py [path/to/image] '
            'or \nget_phase.py [path/to/reference_image] [path/to/target_image1] ... '
            '\n  [NOTE] target_image1 can include the wild cards.'
            '\n         e.g. step[0-9][0-9].bmp will cover step01.bmp, step02.bmp, ..., step99.bmp'
            'or \nget_phase.py [path/to/video] reference_frames(e.g. 1,2 or 1-10)'
        )
    
    config = ConfigParser()
    config.read('settings.ini')
    image_format = config['output']['format'].strip()
    if ',' in image_format:
        image_format = [img_format.strip() for img_format in image_format.split(',')]

    path = sys.argv[1]
    video_images = []
    if any(ex in path for ex in __VIDEO_EXTENSIONS__):  # video
        image, video_images = _video2images(path)
    image = PIL.Image.open(path)
    parameters, convolved = get_phase(np.array(image), config, parameters=None)
    amplitude, phase = np.abs(convolved), np.angle(convolved)

    if len(sys.argv) == 2:
        save_array(amplitude, path, '_amp', image_format=image_format)
        save_array(phase, path, '_phase', image_format=image_format)

    elif len(video_images) > 1:  # video
        image_format = config['output']['video_format'].strip()
        target_amplitudes = []
        target_phases = []
        for image in video_images:
            image = PIL.Image.open(path)
            _, target_conv = get_phase(np.array(image), config, parameters=parameters)

            target_amplitudes.append(np.abs(target_conv))
            target_phases.append(
                get_phase_from_reference(target_conv, convolved)
            )
        save_video(target_amplitudes, path, '_amp', image_format=image_format)
        save_video(target_phases, path, '_phase', image_format=image_format)
    
    else:
        paths = []
        for path in sys.argv[2:]:
            paths += glob.glob(path)
        print("converting ", paths)
            
        for path in paths:
            image = PIL.Image.open(path)
            _, target_conv = get_phase(np.array(image), config, parameters=parameters)

            target_amplitude = np.abs(target_conv)
            target_phase = get_phase_from_reference(
                target_conv, convolved)
            save_array(target_amplitude, path, '_amp', image_format=image_format)
            save_array(target_phase, path, '_phase', image_format=image_format)
