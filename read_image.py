import numpy as np
from constants import au


def read_image(fname='image.out'):
    """
    Parse a RADMC-3D image.out file.

    Returns a dict with:
      iformat  : 1 (intensity only) or 3 (full Stokes I,Q,U,V)
      nx, ny   : pixel counts
      nwav     : number of wavelength channels
      dpx, dpy : pixel size in cm
      wav      : wavelengths in microns, shape (nwav,)
      data     : intensity array in erg/s/cm²/Hz/sr
                   iformat 1 -> shape (nwav, ny, nx)
                   iformat 3 -> shape (nwav, ny, nx, 4)  [I, Q, U, V]
      x_au, y_au : pixel-centre coordinates in au, shapes (nx,) and (ny,)
    """
    with open(fname, 'r') as f:
        iformat = int(f.readline())
        nx, ny = map(int, f.readline().split())
        nwav = int(f.readline())
        dpx, dpy = map(float, f.readline().split())
        wav = np.array([float(f.readline()) for _ in range(nwav)])
        f.readline()  # blank separator
        raw = f.read()

    nstokes = 4 if iformat == 3 else 1
    data = np.fromstring(raw, sep=' ', dtype=float)
    data = data[:nwav * ny * nx * nstokes].reshape(nwav, ny, nx, nstokes)

    if nstokes == 1:
        data = data[..., 0]

    x_au = (np.arange(nx) - nx / 2 + 0.5) * dpx / au
    y_au = (np.arange(ny) - ny / 2 + 0.5) * dpy / au

    return dict(
        iformat=iformat, nx=nx, ny=ny, nwav=nwav,
        dpx=dpx, dpy=dpy, wav=wav, nstokes=nstokes,
        data=data, x_au=x_au, y_au=y_au,
    )


def image_extent_au(im):
    dpx_au = im['dpx'] / au
    dpy_au = im['dpy'] / au
    x = im['x_au']
    y = im['y_au']
    return [x[0] - dpx_au / 2, x[-1] + dpx_au / 2,
            y[0] - dpy_au / 2, y[-1] + dpy_au / 2]


def au_to_arcsec(x_au, dist_pc):
    return x_au / dist_pc
