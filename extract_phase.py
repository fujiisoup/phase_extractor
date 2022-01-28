import sys
from configparser import ConfigParser
import os
import glob
import PIL.Image
import numpy as np
from scipy import signal, optimize

try:
    import cv2
    __HAS_OPENCV__ = True
except ImportError:
    __HAS_OPENCV__ = False


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
    outpath = src_path[:src_path.rfind('.')] + suffix + '.' + video_format
    print(arrays[0].shape)
    out = cv2.VideoWriter(outpath, cv2.VideoWriter_fourcc(*'DIVX'), 15, arrays[0].shape) 
    for array in arrays:
        vmin = np.min(array)
        vmax = np.max(array)
        array = ((array - vmin) / (vmax - vmin) * 255).astype(np.uint8)
        out.write(array)
    out.release()


def _video2images(video):
    if not __HAS_OPENCV__:
        raise ImportError(
            "Open cv must be installed to read video.\n"
        )

    vidcap = cv2.VideoCapture(video)
    success, image = vidcap.read()
    count = 0
    images = []
    while success:
        success, image = vidcap.read()
        image = np.array(image)
        # here we just use the r channel. Maybe we need something here
        if image.ndim == 3:
            image = image[..., 0]
        if success:
            images.append(image)
    return images[0], images


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
    video_format = config['output'].get('video_format').strip()
    if ',' in image_format:
        image_format = [img_format.strip() for img_format in image_format.split(',')]

    path = sys.argv[1]
    video_images = None
    if any(ex in path for ex in __VIDEO_EXTENSIONS__):  # video
        image, video_images = _video2images(path)
    else:
        image = PIL.Image.open(path)
    parameters, convolved = get_phase(np.array(image), config, parameters=None)
    amplitude, phase = np.abs(convolved), np.angle(convolved)

    if len(sys.argv) == 2 and video_images is None:
        save_array(amplitude, path, '_amp', image_format=image_format)
        save_array(phase, path, '_phase', image_format=image_format)

    elif video_images is not None:  # video
        image_format = config['output']['video_format'].strip()
        target_amplitudes = []
        target_phases = []
        for image in video_images:
            _, target_conv = get_phase(np.array(image), config, parameters=parameters)

            target_amplitudes.append(np.abs(target_conv))
            target_phases.append(
                get_phase_from_reference(target_conv, convolved)
            )
        save_video(target_amplitudes, path, '_amp', video_format='avi')
        save_video(target_phases, path, '_phase', video_format='avi')
    
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
