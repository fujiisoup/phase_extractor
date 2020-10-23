from .extract_phase import get_phase
from configparser import ConfigParser
import numpy as np
import PIL.Image


def test_getphase():
    # image = PIL.Image.open('data/wo_step.bmp')
    # image = PIL.Image.open('data/step.bmp')
    image = PIL.Image.open('data/wostep_nopinhole_0th1st.bmp')
    config = ConfigParser()
    config.read('settings.ini')
    parameters, convolved = get_phase(np.array(image), config)

    amplitude, phase = np.abs(convolved), np.angle(convolved)

    image = PIL.Image.open('data/wo_step.bmp')
    paramters, convolved = get_phase(
        np.array(image), config, parameters)
