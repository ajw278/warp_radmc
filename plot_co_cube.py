#!/usr/bin/env python3
"""
Plot channel maps from a RADMC-3D CO line image.out.

Wavelengths are converted to line-of-sight velocity using the rest wavelength
of the observed transition (default: CO J=2-1 at 1300.4 μm / 230.538 GHz).

Usage
-----
    python plot_co_cube.py [image.out] [options]

Examples
--------
    python plot_co_cube.py
    python plot_co_cube.py image.out --dist 160 --ncols 8 --outfile co_cube.png
    python plot_co_cube.py image.out --lam0 1300.4 --dist 160 --vrange -6 6
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

from read_image import read_image, image_extent_au, au_to_arcsec
from mpl_setup import *

# CO rest wavelengths in μm for common transitions
CO_REST_WAV = {
    '1-0':  2600.8,
    '2-1':  1300.4,
    '3-2':   866.96,
    '6-5':   433.56,
}
C_KMS = 2.998e5  # km/s


def wav_to_velocity(wav_um, lam0_um):
    """Doppler velocity in km/s relative to rest wavelength."""
    return C_KMS * (wav_um - lam0_um) / lam0_um


def plot_co_cube(fname='image.out', lam0_um=CO_REST_WAV['2-1'],
                 dist_pc=None, ncols=6, vrange=None,
                 log=False, outfile=None, dpi=150):

    im = read_image(fname)

    if im['nwav'] < 2:
        raise SystemExit('image.out has only 1 wavelength — run RADMC-3D with linenlam > 1.')

    vel = wav_to_velocity(im['wav'], lam0_um)  # km/s, shape (nwav,)

    # Optionally restrict to a velocity range
    if vrange is not None:
        vmin_sel, vmax_sel = vrange
        sel = (vel >= vmin_sel) & (vel <= vmax_sel)
        if not sel.any():
            raise SystemExit(f'No channels in velocity range {vrange} km/s.')
        vel = vel[sel]
        data = im['data'][sel]  # (nchan, ny, nx) or (nchan, ny, nx, 4)
    else:
        data = im['data']

    # For Stokes output take Stokes I
    if im['iformat'] == 3:
        data = data[..., 0]

    nchan = data.shape[0]
    ext = image_extent_au(im)
    xlabel = 'x [au]'
    ylabel = 'y [au]'
    if dist_pc is not None:
        ext = [au_to_arcsec(v, dist_pc) for v in ext]
        xlabel = r'$\Delta\alpha$ [arcsec]'
        ylabel = r'$\Delta\delta$ [arcsec]'

    # Global colour limits
    finite = data[np.isfinite(data)]
    if finite.size and finite.max() > 0:
        vmin_plot = 0.0
        vmax_plot = np.nanpercentile(finite[finite > 0], 99) if (finite > 0).any() else 1.0
    else:
        vmin_plot, vmax_plot = 0.0, 1.0

    ncols = min(ncols, nchan)
    nrows = int(np.ceil(nchan / ncols))
    fig, axs = plt.subplots(nrows, ncols,
                            figsize=(2.8 * ncols, 2.8 * nrows),
                            constrained_layout=True)
    axs = np.atleast_2d(axs)

    for k in range(nrows * ncols):
        ax = axs[k // ncols, k % ncols]
        if k >= nchan:
            ax.axis('off')
            continue

        channel = data[k]
        norm = plt.matplotlib.colors.LogNorm(vmin=max(vmin_plot, 1e-40), vmax=vmax_plot) if log else None
        im_plot = ax.imshow(channel, origin='lower', extent=ext,
                            cmap='inferno', norm=norm,
                            vmin=(None if log else vmin_plot),
                            vmax=(None if log else vmax_plot),
                            interpolation='nearest')
        ax.set_title(f'{vel[k]:+.2f} km/s', fontsize=8)
        if k % ncols == 0:
            ax.set_ylabel(ylabel, fontsize=7)
        if k // ncols == nrows - 1:
            ax.set_xlabel(xlabel, fontsize=7)
        ax.tick_params(labelsize=6)

    cbar = fig.colorbar(im_plot, ax=axs, fraction=0.015, pad=0.02)
    cbar.set_label(r'$I_\nu$ [erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$ sr$^{-1}$]', fontsize=9)

    dist_str = f', d = {dist_pc} pc' if dist_pc else ''
    fig.suptitle(rf'CO channel maps  ($\lambda_0 = {lam0_um:.1f}\ \mu$m{dist_str})', fontsize=11)

    if outfile:
        fig.savefig(outfile, dpi=dpi, bbox_inches='tight')
        print(f'Saved {outfile}')
    else:
        plt.show()

    return fig, axs


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('image', nargs='?', default='image.out',
                   help='RADMC-3D image.out file (default: image.out)')
    p.add_argument('--lam0', type=float, default=CO_REST_WAV['2-1'], metavar='UM',
                   help=f'Rest wavelength in μm (default: {CO_REST_WAV["2-1"]} = CO J=2-1)')
    p.add_argument('--dist', type=float, default=None, metavar='PC',
                   help='Source distance in pc; converts axes to arcsec')
    p.add_argument('--ncols', type=int, default=6,
                   help='Columns per row in the channel map grid (default: 6)')
    p.add_argument('--vrange', type=float, nargs=2, default=None, metavar=('VMIN', 'VMAX'),
                   help='Velocity range to display in km/s, e.g. --vrange -6 6')
    p.add_argument('--log', action='store_true',
                   help='Log colour scale')
    p.add_argument('--outfile', default=None,
                   help='Save to this file instead of showing interactively')
    p.add_argument('--dpi', type=int, default=150)
    args = p.parse_args()

    plot_co_cube(fname=args.image, lam0_um=args.lam0,
                 dist_pc=args.dist, ncols=args.ncols,
                 vrange=args.vrange, log=args.log,
                 outfile=args.outfile, dpi=args.dpi)


if __name__ == '__main__':
    main()
