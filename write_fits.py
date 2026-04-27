#!/usr/bin/env python3
"""
Convert a RADMC-3D image.out to a FITS file.

Handles both single-wavelength images (scattered light) and multi-channel
line cubes. For Stokes output (iformat 3) the Stokes axis is included.

WCS headers are written in au by default. Pass --dist to convert to arcsec.
For line cubes the spectral axis is written as a velocity axis (km/s) relative
to the rest wavelength of CO J=2-1 (1300.4 μm); override with --lam0.

Usage
-----
    python write_fits.py                          # image.out -> test.fits
    python write_fits.py image.out -o test.fits
    python write_fits.py image.out --dist 160
    python write_fits.py co_image.out --lam0 1300.4 --dist 160 -o co.fits
"""

import argparse
import numpy as np
from astropy.io import fits

from read_image import read_image

C_KMS = 2.998e5  # km/s


def write_fits(fname='image.out', outfile='test.fits',
               dist_pc=None, lam0_um=1300.4):

    im = read_image(fname)
    data = im['data']   # (nwav, ny, nx) or (nwav, ny, nx, 4)
    wav  = im['wav']
    nwav = im['nwav']
    nx, ny = im['nx'], im['ny']
    dpx, dpy = im['dpx'], im['dpy']

    from constants import au
    dpx_au = dpx / au
    dpy_au = dpy / au

    if dist_pc is not None:
        scale_x = -dpx_au / dist_pc / 3600.0   # degrees, negative (RA increases left)
        scale_y =  dpy_au / dist_pc / 3600.0
        cunit_spatial = 'deg'
        ctype_x, ctype_y = 'RA---TAN', 'DEC--TAN'
    else:
        scale_x = -dpx_au   # au, negative to match sky orientation
        scale_y =  dpy_au
        cunit_spatial = 'au'
        ctype_x, ctype_y = 'x', 'y'

    crpix_x = nx / 2.0 + 0.5
    crpix_y = ny / 2.0 + 0.5

    hdr = fits.Header()
    hdr['BUNIT'] = ('erg/s/cm2/Hz/sr', 'Specific intensity')
    hdr['ORIGIN'] = 'RADMC-3D / warp_radmc'

    if im['iformat'] == 3 and nwav == 1:
        # Scattered light Stokes image: axes are (Stokes, y, x)
        cube = data[0].transpose(2, 0, 1)   # (4, ny, nx)
        hdr['NAXIS'] = 3
        _add_spatial_wcs(hdr, 1, ctype_x, scale_x, crpix_x, cunit_spatial)
        _add_spatial_wcs(hdr, 2, ctype_y, scale_y, crpix_y, cunit_spatial)
        hdr['CTYPE3'] = 'STOKES'
        hdr['CRPIX3'] = 1.0
        hdr['CRVAL3'] = 1.0   # 1=I, 2=Q, 3=U, 4=V (FITS Stokes convention)
        hdr['CDELT3'] = 1.0
        hdr['CUNIT3'] = ''
        hdr['COMMENT'] = 'Stokes planes: 1=I 2=Q 3=U 4=V'
        hdr['WAVELEN'] = (float(wav[0]), 'Wavelength [micron]')

    elif im['iformat'] == 3 and nwav > 1:
        # Line cube with Stokes: (Stokes, nwav, ny, nx)
        cube = data[:, :, :, 0].transpose(...)   # take Stokes I only for simplicity
        # Actually write Stokes I only for line cubes
        cube = data[..., 0]   # (nwav, ny, nx)
        vel = C_KMS * (wav - lam0_um) / lam0_um
        dvel = float(np.mean(np.diff(vel)))
        hdr['NAXIS'] = 3
        _add_spatial_wcs(hdr, 1, ctype_x, scale_x, crpix_x, cunit_spatial)
        _add_spatial_wcs(hdr, 2, ctype_y, scale_y, crpix_y, cunit_spatial)
        _add_velocity_wcs(hdr, 3, vel[0], dvel, lam0_um)
        cube = cube   # (nwav, ny, nx)

    elif nwav > 1:
        # Scalar line cube: (nwav, ny, nx)
        cube = data
        vel = C_KMS * (wav - lam0_um) / lam0_um
        dvel = float(np.mean(np.diff(vel)))
        hdr['NAXIS'] = 3
        _add_spatial_wcs(hdr, 1, ctype_x, scale_x, crpix_x, cunit_spatial)
        _add_spatial_wcs(hdr, 2, ctype_y, scale_y, crpix_y, cunit_spatial)
        _add_velocity_wcs(hdr, 3, vel[0], dvel, lam0_um)

    else:
        # Single-wavelength scalar image: (ny, nx)
        cube = data[0]
        hdr['NAXIS'] = 2
        _add_spatial_wcs(hdr, 1, ctype_x, scale_x, crpix_x, cunit_spatial)
        _add_spatial_wcs(hdr, 2, ctype_y, scale_y, crpix_y, cunit_spatial)
        hdr['WAVELEN'] = (float(wav[0]), 'Wavelength [micron]')

    fits.writeto(outfile, cube.astype(np.float32), header=hdr, overwrite=True)
    print(f'Written {outfile}  shape={cube.shape}  dtype=float32')


def _add_spatial_wcs(hdr, n, ctype, cdelt, crpix, cunit):
    hdr[f'CTYPE{n}'] = ctype
    hdr[f'CRPIX{n}'] = crpix
    hdr[f'CRVAL{n}'] = 0.0
    hdr[f'CDELT{n}'] = cdelt
    hdr[f'CUNIT{n}'] = cunit


def _add_velocity_wcs(hdr, n, crval, cdelt, lam0_um):
    hdr[f'CTYPE{n}'] = 'VRAD'
    hdr[f'CRPIX{n}'] = 1.0
    hdr[f'CRVAL{n}'] = float(crval)
    hdr[f'CDELT{n}'] = float(cdelt)
    hdr[f'CUNIT{n}'] = 'km/s'
    hdr[f'RESTFRQ'] = (C_KMS / lam0_um * 1e4 * 1e9, 'Rest frequency [Hz]')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('image', nargs='?', default='image.out')
    p.add_argument('-o', '--outfile', default='test.fits')
    p.add_argument('--dist', type=float, default=None, metavar='PC',
                   help='Source distance in pc; writes WCS in degrees')
    p.add_argument('--lam0', type=float, default=1300.4, metavar='UM',
                   help='Rest wavelength in μm for line cubes (default: 1300.4 = CO J=2-1)')
    args = p.parse_args()
    write_fits(fname=args.image, outfile=args.outfile,
               dist_pc=args.dist, lam0_um=args.lam0)


if __name__ == '__main__':
    main()
