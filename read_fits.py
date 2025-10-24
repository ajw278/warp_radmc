#!/usr/bin/env python3

import argparse
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits


def find_spectral_axis(header):
    naxis = header.get("NAXIS", 0)
    for i in range(1, naxis + 1):
        ctype = str(header.get(f"CTYPE{i}", "")).upper()
        if any(k in ctype for k in ("FREQ", "VRAD", "VELO", "VOPT", "WAVE", "AWAV", "ENER")):
            return i
    # common default: CTYPE3 is spectral
    return min(3, naxis)


def spectral_world_value(header, fits_axis_i, chan_zero_based):
    key = lambda k: header.get(f"{k}{fits_axis_i}")
    crval = float(key("CRVAL") or 0.0)
    cdelt = float(key("CDELT") or 1.0)
    crpix = float(key("CRPIX") or 1.0)

    pix = chan_zero_based + 1.0
    world = crval + (pix - crpix) * cdelt
    unit = header.get(f"CUNIT{fits_axis_i}", "").strip()
    ctype = header.get(f"CTYPE{fits_axis_i}", "").strip()
    return world, unit, ctype


def main():
    p = argparse.ArgumentParser(description="Plot channel maps from a FITS spectral cube.")
    p.add_argument("fitsfile", help="Path to FITS cube (e.g., warp.fits)")
    p.add_argument("--hdu", type=int, default=None, help="HDU index to use (default: first with data)")
    p.add_argument("--max-panels", type=int, default=24, help="Max number of channels to show")
    p.add_argument("--step", type=int, default=1, help="Take every Nth channel")
    p.add_argument("--perchannel", action="store_true",
                   help="Autoscale each panel independently (default: global scaling)")
    p.add_argument("--outfile", default=None, help="Save figure to this PNG instead of showing")
    p.add_argument("--dpi", type=int, default=150, help="Figure DPI when saving")
    args = p.parse_args()

    hdul = fits.open(args.fitsfile)

    if args.hdu is not None:
        hdu = hdul[args.hdu]
    else:
        hdu = next((h for h in hdul if isinstance(getattr(h, "data", None), np.ndarray)
                    and h.data.ndim >= 3), hdul[0])

    data = np.asarray(hdu.data, dtype=float)
    hdr = hdu.header
    naxis = data.ndim
    if naxis < 3:
        raise SystemExit(f"Need a 3D cube, got ndim={naxis}")

    fits_spec_i = find_spectral_axis(hdr)                  # 1..N
    np_spec_axis = naxis - fits_spec_i                     # FITS->NumPy index mapping

    axes = list(range(naxis))
    spatial_axes = [ax for ax in axes if ax != np_spec_axis]
    transpose_order = [np_spec_axis] + spatial_axes
    cube = np.transpose(data, transpose_order)
    nchan, ny, nx = cube.shape[0], cube.shape[-2], cube.shape[-1]

    chan_indices = np.arange(0, nchan, args.step)
    if chan_indices.size > args.max_panels:
        chan_indices = np.linspace(0, nchan - 1, args.max_panels, dtype=int)

    nplot = chan_indices.size
    ncols = min(6, nplot)
    nrows = int(np.ceil(nplot / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.7*ncols, 2.7*nrows), squeeze=False)

    if args.perchannel:
        vmin_global = vmax_global = None
    else:
        finite = np.isfinite(cube)
        if np.any(finite):
            vmin_global = np.nanpercentile(cube[finite], 1)
            vmax_global = np.nanpercentile(cube[finite], 99)
        else:
            vmin_global = vmax_global = None

    for k, ch in enumerate(chan_indices):
        ax = axes[k // ncols, k % ncols]
        img = cube[ch, :, :]

        if args.perchannel:
            if np.isfinite(img).any():
                vmin = np.nanpercentile(img, 1)
                vmax = np.nanpercentile(img, 99)
            else:
                vmin = vmax = None
        else:
            vmin, vmax = vmin_global, vmax_global

        im = ax.imshow(img, origin="lower", aspect="equal",
                       vmin=vmin, vmax=vmax, interpolation="nearest")
        # Title with spectral world value
        world, unit, ctype = spectral_world_value(hdr, fits_spec_i, int(ch))
        # Tidy unit for display
        unit_disp = f" {unit}" if unit else ""
        ax.set_title(f"ch {ch}  ({ctype}: {world:.5g}{unit_disp})", fontsize=9)
        ax.set_xlabel("x [pix]")
        ax.set_ylabel("y [pix]")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cbar.ax.tick_params(labelsize=8)

    # Hide unused panels
    for k in range(nplot, nrows * ncols):
        axes[k // ncols, k % ncols].axis("off")

    fig.suptitle(f"{args.fitsfile} — {nchan} channels (showing {nplot}, step={args.step})",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if args.outfile:
        fig.savefig(args.outfile, dpi=args.dpi, bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":
    main()
