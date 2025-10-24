from radmc3dPy import image
from radmc3d_tools import simpleread
import matplotlib.pyplot as plt
import numpy as np

im = simpleread.read_image()
im.fwhm = []

image.radmc3dImage.writeFits(im, 'test.fits')