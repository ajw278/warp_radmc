#!/usr/bin/env python3
"""
Plot a RADMC-3D scattered-light image (iformat 1 or 3).

For Stokes output (iformat 3) the script produces three panels:
  - Total intensity  I
  - Polarised intensity  PI = sqrt(Q² + U²)
  - Polarisation fraction  PI / I

For scalar output (iformat 1) a single intensity panel is shown.

Usage
-----
    python plot_scattered_light.py [image.out] [options]

Examples
--------
    python plot_scattered_light.py
    python plot_scattered_light.py image.out --dist 160 --log --outfile scat_2p2um.png
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from read_image import read_image, image_extent_au, au_to_arcsec
from mpl_setup import *


def plot_scattered_light(fname='image.out', dist_pc=None, log=True,
                         vmin=None, vmax=None, outfile=None, dpi=200):

    im = read_image(fname)
    wav0 = im['wav'][0]

    if im['nwav'] > 1:
        print(f"Warning: {im['nwav']} wavelengths found; plotting the first (λ = {wav0:.3f} μm).")

    ext = image_extent_au(im)
    xlabel = 'x [au]'
    ylabel = 'y [au]'
    if dist_pc is not None:
        ext = [au_to_arcsec(v, dist_pc) for v in ext]
        xlabel = r'$\Delta\alpha$ [arcsec]'
        ylabel = r'$\Delta\delta$ [arcsec]'

    data = im['data']  # (nwav, ny, nx) or (nwav, ny, nx, 4)

    if im['iformat'] == 3:
        I  = data[0, :, :, 0]
        Q  = data[0, :, :, 1]
        U  = data[0, :, :, 2]
        PI = np.sqrt(Q**2 + U**2)
        with np.errstate(invalid='ignore', divide='ignore'):
            PF = np.where(I > 0, PI / I, np.nan)
        panels = [
            (I,  r'Stokes $I$',  'inferno', False),
            (PI, r'Pol. intensity $\sqrt{Q^2+U^2}$', 'viridis', False),
            (PF, r'Pol. fraction $\sqrt{Q^2+U^2}/I$', 'plasma', True),
        ]
        ncols = 3
    else:
        I = data[0]
        panels = [(I, r'Intensity $I$', 'inferno', False)]
        ncols = 1

    fig, axs = plt.subplots(1, ncols, figsize=(5 * ncols, 4.5), constrained_layout=True)
    if ncols == 1:
        axs = [axs]

    for ax, (Z, title, cmap, is_fraction) in zip(axs, panels):
        if log and not is_fraction:
            pos = Z > 0
            if pos.any():
                hi = vmax if vmax is not None else np.percentile(Z[pos], 98)
                lo = vmin if vmin is not None else hi / 100.0  # 2 dex dynamic range
                norm = mcolors.LogNorm(vmin=lo, vmax=hi)
            else:
                norm = None
        else:
            norm = None

        imkw = dict(origin='lower', extent=ext, cmap=cmap,
                    interpolation='nearest')
        if norm is not None:
            imkw['norm'] = norm
        elif is_fraction:
            imkw['vmin'] = 0.0
            imkw['vmax'] = 1.0
        else:
            imkw['vmin'] = vmin
            imkw['vmax'] = vmax
        im_plot = ax.imshow(Z, **imkw)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        cb = fig.colorbar(im_plot, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label('' if is_fraction else r'erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$ sr$^{-1}$')

    dist_str = f', d = {dist_pc} pc' if dist_pc else ''
    fig.suptitle(rf'Scattered light  $\lambda = {wav0:.2f}\ \mu$m{dist_str}', fontsize=12)

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
    p.add_argument('--dist', type=float, default=None, metavar='PC',
                   help='Source distance in pc; converts axes to arcsec')
    p.add_argument('--log', action='store_true', default=True,
                   help='Log colour scale (default)')
    p.add_argument('--linear', dest='log', action='store_false',
                   help='Linear colour scale')
    p.add_argument('--vmin', type=float, default=None)
    p.add_argument('--vmax', type=float, default=None)
    p.add_argument('--outfile', default=None,
                   help='Save to this file instead of showing interactively')
    p.add_argument('--dpi', type=int, default=200)
    args = p.parse_args()

    plot_scattered_light(fname=args.image, dist_pc=args.dist, log=args.log,
                         vmin=args.vmin, vmax=args.vmax,
                         outfile=args.outfile, dpi=args.dpi)


if __name__ == '__main__':
    main()
