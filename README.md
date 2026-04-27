# warp_radmc

Code to generate warped accretion disc models and all input files required for [RADMC-3D](https://www.ita.uni-heidelberg.de/~dullemond/software/radmc-3d/) radiative transfer simulations. The default configuration models the disc around **MWC 758** using an observed warp profile, but the framework is general and can be adapted to any disc with a spatially varying inclination and position angle.

---

## Overview

A warped disc has a tilt (inclination and position angle) that varies with radius. This code:

1. Reads an observed warp profile (δi, δPA as a function of radius)
2. Extends the profile to unconstrained radii via cubic spline interpolation
3. Computes density and Keplerian velocity fields on a spherical grid, accounting for the local disc orientation at each radius
4. Writes all RADMC-3D input files (grid, density, velocity, stellar parameters, opacity, etc.)
5. Generates diagnostic plots of the warp profile, density structure, and velocity fields

The vertical density structure follows a Gaussian (hydrostatic equilibrium) profile, and the velocity field is purely Keplerian in each annulus's local frame, rotated into the global Cartesian frame using rotation matrices derived from the warp profile.

---

## Repository Structure

```
warp_radmc/
├── build_warp.py           # Main script — run this to generate all RADMC-3D inputs
├── write_radmc.py          # Writes RADMC-3D input files in required formats
├── extend_warpprof.py      # Extends observed warp profile via cubic spline interpolation
├── plot_funcs.py           # Diagnostic visualisation functions
├── utils.py                # Coordinate transforms, rotation matrices, density profile
├── constants.py            # Physical constants in CGS units
├── misc_funcs.py           # Disc surface-finding utility
├── mpl_setup.py            # Matplotlib global styling
├── read_fits.py            # FITS spectral cube viewer (channel maps)
├── write_fits.py           # Converts RADMC-3D image.out → FITS
└── mwc758_warpprofile.txt  # Input: observed MWC 758 warp profile (r, δi, δPA)
```

---

## Dependencies

- Python 3
- `numpy`
- `scipy`
- `matplotlib`
- `astropy` (for `read_fits.py` / `write_fits.py`)
- [`radmc3dPy`](https://www.ita.uni-heidelberg.de/~dullemond/software/radmc-3d/radmc3dpy/) (for `write_fits.py`)

Install Python dependencies with:

```bash
pip install numpy scipy matplotlib astropy
```

`radmc3dPy` must be installed separately following the RADMC-3D documentation.

---

## Quick Start

To generate all RADMC-3D input files with the default MWC 758 configuration:

```bash
python build_warp.py
```

This will create the following files in the working directory:

| File | Description |
|---|---|
| `amr_grid.inp` | Spherical grid cell edges (r, θ, φ) |
| `dust_density.inp` | Dust mass density on the grid |
| `gas_velocity.inp` | Gas velocity field (3 Cartesian components) |
| `numberdens_co.inp` | CO molecular number density |
| `wavelength_micron.inp` | Wavelength grid for radiative transfer |
| `stars.inp` | Stellar parameters and spectrum |
| `lines.inp` | Molecular line configuration (CO) |
| `dustopac.inp` | Dust opacity control file |
| `radmc3d.inp` | Main RADMC-3D configuration |

It also produces diagnostic plots:

| File | Description |
|---|---|
| `warp_profile_extrapolated.pdf` | Warp profile (δi, δPA) with spline and extension |
| `density_rtheta_bipolar.png` | Bipolar (r, θ) density cross-section |
| `velocity_slice_zOverR0p10.png` | Velocity components at z/R = 0.1 |

---

## Model Parameters

Key parameters are set at the top of `build_warp.py`:

### Star

| Parameter | Default | Description |
|---|---|---|
| `Mstar` | 1.4 M☉ | Stellar mass |
| `Rstar` | 2 R☉ | Stellar radius |
| `Tstar` | 7600 K | Effective temperature |

### Grid

| Parameter | Default | Description |
|---|---|---|
| `nr` | 200 | Number of radial cells |
| `ntheta` | 200 | Number of polar angle cells |
| `nphi` | 200 | Number of azimuthal cells |
| `r_in` | 10 au | Inner grid boundary |
| `r_out` | 150 au | Outer grid boundary |

### Dust density

The dust surface density follows a power-law with a Gaussian vertical profile:

```
ρ(r, z) = ρ₀ · (r / r₀)^(-α) · exp(−z² / (2 H²))
```

| Parameter | Default | Description |
|---|---|---|
| `RHO0` | 2 × 10⁻¹⁵ g cm⁻³ | Midplane density at r₀ |
| `r0` | 10 au | Scale radius |
| `alpha` | 1 | Surface density power-law exponent |
| `flare` | 0.2 | Flaring exponent (H/r ∝ r^flare) |
| `h0` | 0.1 | Aspect ratio H/r at r₀ |

### CO abundance

CO number density is derived from dust density assuming a fixed dust-to-gas ratio and CO abundance:

| Parameter | Default | Description |
|---|---|---|
| `d2g` | 0.01 | Dust-to-gas mass ratio |
| `co_abundance` | 1 × 10⁻⁴ | CO-to-H₂ number ratio |

---

## Warp Profile

The warp profile is read from `mwc758_warpprofile.txt`, a whitespace-delimited file with three columns:

```
# radius [au]   delta_i [rad]   delta_PA [rad]
46.77            0.032           -0.11
...
```

`delta_i` is the inclination offset from the global disc plane and `delta_PA` is the position-angle offset, both as a function of radius.

The profile is extended to the inner (`r_in`) and outer (`r_out`) grid boundaries using cubic spline interpolation (`extend_warpprof.py`). Extension boundary values are currently set by hand to physically motivated constants at the edges.

---

## Code Modules

### `build_warp.py`

The main entry point. Defines all model parameters, loads and extends the warp profile, computes density and velocity on the grid, generates plots, and calls `write_radmc.py` to write all input files.

### `write_radmc.py`

Contains one function per RADMC-3D input file. All arrays are written in the Fortran column-major order expected by RADMC-3D. Key functions:

- `write_amr_grid_spherical(r, theta, phi)` — grid edges in each coordinate
- `write_density_spherical(density)` — dust density array
- `write_gas_velocity(vx, vy, vz)` — three velocity components
- `write_co_number_density(density)` — CO number density derived from dust
- `write_wavelength_grid()` — three logarithmic wavelength segments
- `write_stars(Rstar, Mstar, Tstar)` — stellar parameters and blackbody spectrum
- `write_line_input()` — molecular data file configuration (CO)
- `write_opacity_control()` — dust opacity file pointer
- `write_radmc3d_inp()` — photon count and optical depth settings

### `extend_warpprof.py`

`extend_warp_profile(r_obs, delta_i_obs, delta_PA_obs, r_in, r_out)` returns interpolant functions `f_di(r)` and `f_dPA(r)` valid across `[r_in, r_out]`. Uses `scipy.interpolate.CubicSpline` with clamped boundary conditions.

### `utils.py`

Core geometry functions:

- `l_vector(i, delta_i, delta_PA)` — angular momentum unit vector for a warped annulus
- `rotation_from_z_to_l(l)` — rotation matrix mapping ẑ → l̂
- `vertical_density(z, H)` — Gaussian vertical profile
- `interpolate_to_cartesian(r, theta, phi, density)` — spherical → Cartesian interpolation
- `compute_cell_walls(centres)` — cell edge positions from cell centres

### `plot_funcs.py`

Advanced visualisation of the 3D disc structure:

- `plot_warp_profile()` — δi and δPA vs radius with data, spline, and extension points
- `plot_bipolar_r_theta_slice()` — r–θ density cross-section (upper/lower hemispheres)
- `plot_velocity_slice()` — three-panel velocity component profile at fixed z/R
- `plot_velocity_slice_map()` — 2D (R, φ) map of velocity components

### `read_fits.py`

Stand-alone FITS spectral cube viewer. Reads a FITS cube, identifies the spectral axis from the header, and produces multi-panel channel maps.

```bash
python read_fits.py image.fits --ncols 5 --dpi 150 --scale global
```

Arguments:

| Flag | Default | Description |
|---|---|---|
| `filename` | — | Path to FITS file |
| `--ncols` | 4 | Columns per row in channel map |
| `--dpi` | 100 | Output figure DPI |
| `--scale` | `per_channel` | Colour scale: `per_channel` or `global` |

### `read_image.py`

Standalone parser for RADMC-3D `image.out` files. Requires no external packages beyond numpy. Supports both scalar (iformat 1) and full-Stokes (iformat 3) outputs.

```python
from read_image import read_image
im = read_image('image.out')
# im['data'] shape: (nwav, ny, nx) or (nwav, ny, nx, 4) for Stokes
# im['x_au'], im['y_au']: pixel-centre coordinates in au
# im['wav']: wavelengths in microns
```

### `plot_scattered_light.py`

Plots a single-wavelength scattered-light `image.out`. For Stokes output (iformat 3) produces three panels: total intensity I, polarised intensity PI = √(Q²+U²), and polarisation fraction PI/I.

```bash
python plot_scattered_light.py                               # show interactively
python plot_scattered_light.py image.out --dist 160 --outfile scat_2p2um.png
python plot_scattered_light.py image.out --linear --dist 160
```

| Flag | Default | Description |
|---|---|---|
| `image` | `image.out` | Input file |
| `--dist PC` | — | Source distance in pc; converts axes from au to arcsec |
| `--log` / `--linear` | log | Colour scale |
| `--vmin`, `--vmax` | auto | Colour limits |
| `--outfile` | — | Save to file instead of interactive display |
| `--dpi` | 200 | Output DPI |

### `plot_co_cube.py`

Plots channel maps from a CO line `image.out`. Wavelengths are converted to line-of-sight velocity using a specified rest wavelength (default: CO J=2–1 at 1300.4 μm).

```bash
python plot_co_cube.py                                             # show interactively
python plot_co_cube.py image.out --dist 160 --ncols 8 --outfile co_cube.png
python plot_co_cube.py image.out --dist 160 --vrange -6 6 --outfile co_cube.png
python plot_co_cube.py image.out --lam0 866.96                     # CO J=3-2
```

| Flag | Default | Description |
|---|---|---|
| `image` | `image.out` | Input file |
| `--lam0 UM` | 1300.4 | Rest wavelength in μm (CO J=2–1) |
| `--dist PC` | — | Source distance in pc; converts axes to arcsec |
| `--ncols` | 6 | Columns per row in channel map grid |
| `--vrange VMIN VMAX` | all channels | Velocity range to display in km/s |
| `--log` | off | Log colour scale |
| `--outfile` | — | Save to file instead of interactive display |
| `--dpi` | 150 | Output DPI |

Common CO rest wavelengths: J=1–0 → 2600.8 μm, J=2–1 → 1300.4 μm, J=3–2 → 867.0 μm, J=6–5 → 433.6 μm.

### `write_fits.py`

Converts RADMC-3D `image.out` to a FITS file (`test.fits`) using `radmc3dPy`:

```bash
python write_fits.py
```

---

## Workflow

```
mwc758_warpprofile.txt
        │
        ▼
extend_warpprof.py       ←── cubic spline extension to [r_in, r_out]
        │
        ▼
build_warp.py
  ├── compute density(r, θ, φ) using warp-aware Gaussian profile
  ├── compute velocity(r, θ, φ) using Keplerian + rotation matrices
  ├── generate diagnostic plots
  └── write_radmc.py  ──►  amr_grid.inp
                            dust_density.inp
                            gas_velocity.inp
                            numberdens_co.inp
                            wavelength_micron.inp
                            stars.inp
                            lines.inp
                            dustopac.inp
                            radmc3d.inp
        │
        ▼
   RADMC-3D  ──►  dust_temperature.dat
                  image.out
        │
        ▼
   write_fits.py  ──►  test.fits
        │
        ▼
   read_fits.py   ──►  channel map PNG
```

---

## Running RADMC-3D

After generating the input files, run the RADMC-3D thermal Monte Carlo first to compute the dust temperature, then ray-trace to produce images.

```bash
radmc3d mctherm
```

> **Note on inclination:** `build_warp.py` encodes the disc inclination (`i0 = 21°`) directly into the 3D density and velocity fields by rotating the disc midplane in the grid. The RADMC-3D coordinate z-axis is therefore already the observer's line of sight for a 21°-inclined disc. Use `incl 0` at image time — specifying `incl 21` would double-count it.

### Example 1: scattered light total intensity at 2.2 μm

```bash
radmc3d image lambda 2.2 incl 0 posang 0 nphot_scat 10000000 npix 500 sizeau 120 stokes
```

| Flag | Meaning |
|---|---|
| `lambda 2.2` | Wavelength in microns |
| `incl 0` | Observer along grid z-axis; disc inclination already in density structure |
| `posang 0` | Position angle of the projected disc major axis on sky |
| `nphot_scat 10000000` | Scattering Monte Carlo photons (controls noise, not temperature) |
| `npix 500` | Square image pixel count |
| `sizeau 120` | Image half-width in au |
| `stokes` | Output full Stokes vector (I, Q, U, V); total intensity is Stokes I |

This writes `image.out`. Convert to FITS with:

```bash
python write_fits.py
```

### Example 2: CO J=2–1 line cube

```bash
radmc3d image iline 2 widthkms 10 linenlam 40 incl 0 posang 0 npix 500 sizeau 300
```

| Flag | Meaning |
|---|---|
| `iline 2` | Line index in the CO Leiden data file (1-based; line 2 = J=2–1 at 230.538 GHz) |
| `widthkms 10` | Total velocity width of the cube in km/s |
| `linenlam 40` | Number of frequency channels across that window |
| `incl 0` | Observer along grid z-axis; disc inclination already in density structure |
| `posang 0` | Position angle of the projected disc major axis on sky |
| `npix 500` | Square image pixel count |
| `sizeau 300` | Image half-width in au |

This produces `image.out` with 40 channels. Convert and view with:

```bash
python write_fits.py
python read_fits.py test.fits --step 1 --outfile co_channels.png
```

Refer to the [RADMC-3D documentation](https://www.ita.uni-heidelberg.de/~dullemond/software/radmc-3d/) for the full list of command-line options.

---
