
from mpl_setup import *
import matplotlib.pyplot as plt
from constants import *
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import utils as ut

def plot_warp_profile(r_warp, dinc, dpa, r_ext, dinc_ext, dpa_ext, r_test, f_inc, f_pa, mask_new_points):
    # Create the figure
    plt.figure(figsize=(6, 4))

    # Plot delta inclination
    plt.plot(r_warp/au, dinc, 'k.', label='$\delta i$ (fit)', markersize=6)
    plt.plot(r_test/au, f_inc(r_test), 'C0-', label='$\delta i$ (CubicSpline)')

    # Plot delta PA
    plt.plot(r_warp/au, dpa, 'k^', label='$\delta$PA (fit)', markersize=6)
    plt.plot(r_test/au, f_pa(r_test), 'C1--', label='$\delta$PA (CubicSpline)')

    # Highlight extension points only
    plt.scatter(r_ext[mask_new_points]/au, dinc_ext[mask_new_points], 
                c='C0', marker='x', label='$\delta i$ (added)', zorder=5)
    plt.scatter(r_ext[mask_new_points]/au, dpa_ext[mask_new_points], 
                c='C1', marker='s', label='$\delta$PA (added)', zorder=5)

    # Formatting
    plt.axhline(0, color='gray', ls='--', lw=0.5)
    plt.xlim(np.amin(r_ext/au), 267)
    plt.ylim(-0.1, 0.1)
    plt.xlabel('Radius [au]')
    plt.ylabel('$\delta i$ or $\delta$PA [rad]')
    plt.legend()
    plt.tight_layout()
    plt.savefig('warp_profile_extrapolated.pdf', bbox_inches='tight', format='pdf')
    plt.show()



def plot_bipolar_r_theta_slice(rho_sph, r, theta, phi, phi_value=0.0, output='density_rtheta_bipolar.png'):
	"""
	Plot a bipolar (r, theta) slice at phi and phi + pi, with negative r for the back side.

	Parameters:
	-----------
	rho_sph : ndarray
		3D density array (nr, ntheta, nphi)
	r : ndarray
		Radial coordinates (nr,)
	theta : ndarray
		Polar angle coordinates (ntheta,) in radians
	phi : ndarray
		Azimuthal angle array (nphi,) in radians
	phi_value : float
		Azimuthal angle (in radians) for the front side
	output : str
		Filename to save the output plot
	"""
	phi_value = phi_value % (2 * np.pi)
	phi_plus_pi = (phi_value + np.pi) % (2 * np.pi)

	idx_front = np.argmin(np.abs(phi - phi_value))
	idx_back = np.argmin(np.abs(phi - phi_plus_pi))

	# Extract slices
	rho_front = rho_sph[:, :, idx_front]  # (nr, ntheta)
	rho_back = rho_sph[:, :, idx_back]    # (nr, ntheta)
     
	# Find theta of max density at each radius (in degrees - 90)
	theta_deg = theta * 180 / np.pi - 90.0
	theta_max_front = theta_deg[np.argmax(rho_front, axis=1)]
	theta_max_back  = theta_deg[np.argmax(rho_back, axis=1)]

	# Build r arrays
	R, T = np.meshgrid(r / au, theta * 180 / np.pi - 90.0, indexing='ij')        # (nr, ntheta)
	Rneg, Tneg = np.meshgrid(-r / au, theta * 180 / np.pi -90.0, indexing='ij') # mirrored R

	# Plot
	fig, ax = plt.subplots(figsize=(10, 5))
	c1 = ax.pcolormesh(Rneg, Tneg, np.log10(rho_back + 1e-30), cmap='inferno', shading='auto', vmin=-22, vmax=-15.0)
	c2 = ax.pcolormesh(R, T, np.log10(rho_front + 1e-30), cmap='inferno', shading='auto', vmin=-22, vmax=-15.0)

	
	ax.plot(-r / au, theta_max_back, 'c--', lw=1.0)
	ax.plot(r / au, theta_max_front, 'c--', lw=1.0)


	plt.axvline(46.77, color='r', linewidth=1)
	plt.axvline(-46.77, color='r', linewidth=1)
	ax.set_xlabel('Radius: r [au]')
	ax.set_ylabel(r'$\theta$ [deg]')
	cb = fig.colorbar(c1, ax=ax)
	cb.set_label('log Density [g/cm³]')
	ax.set_ylim([-35., 35.])
	plt.tight_layout()
	plt.savefig(output, dpi=150)
	plt.show()
      


def plot_phi_slices(rhod, xc, yc, zc):
	phi_vals = [0.0, np.pi]
	fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
	interp = RegularGridInterpolator((xc, yc, zc), rhod, bounds_error=False, fill_value=np.nan)
	r = np.linspace(5 * au, 250 * au, 256)
	z = np.linspace(-30 * au, 30 * au, 256)
	R, Z = np.meshgrid(r, z, indexing='ij')
	for i, phi in enumerate(phi_vals):
		x = R * np.cos(phi)
		y = R * np.sin(phi)
		points = np.column_stack([x.ravel(), y.ravel(), Z.ravel()])
		rho_slice = interp(points).reshape(r.shape[0], z.shape[0])
		axes[i].imshow(np.log10(rho_slice + 1e-25), origin='lower', aspect='auto',
						extent=[-30, 30, r[0]/au, r[-1]/au], cmap='inferno')
		axes[i].set_title(f"phi = {phi:.1f} rad")
		axes[i].set_xlabel("z [au]")
	axes[0].set_ylabel("R [au]")
	plt.tight_layout()
	plt.savefig("density_phi_slices.png")
	plt.close()

def plot_density_slice(x, z, rho_cart):
	ny = rho_cart.shape[1] // 2
	fig, ax = plt.subplots(figsize=(6,5))
	im = ax.pcolormesh(x[:,0,0]/au, z[0,0,:]/au, np.log10(rho_cart[:, ny, :].T+1e-30), vmin=-18, vmax=np.log10(RHO0), shading='auto')
	ax.set_xlabel('x [au]')
	ax.set_ylabel('z [au]')
	plt.colorbar(im, ax=ax, label='log10(density)')
	plt.tight_layout()
	plt.show()


def _local_rotation(i0, r_val, f_inc, f_pa):
    """Rotation that maps local face-on (+z') to global (+z -> l_vec)."""
    delta_i = f_inc(r_val) if f_inc is not None else 0.0
    delta_pa = f_pa(r_val) if f_pa is not None else 0.0
    l_vec = ut.l_vector(i0, delta_i, delta_pa)
    R = ut.rotation_from_z_to_l(l_vec)  # maps z -> l (column convention)
    return R

def _cart_to_local(vec_global, R):
	"""Row-vector convention: local = global @ R.T"""
	return vec_global @ R.T

def _coords_global(r, theta, phi):
	rr, tt, pp = np.meshgrid(r, theta, phi, indexing='ij')
	x = rr * np.sin(tt) * np.cos(pp)
	y = rr * np.sin(tt) * np.sin(pp)
	z = rr * np.cos(tt)
	return rr, x, y, z

def _spherical_basis_local(xl, yl, zl):
	"""Build local spherical unit vectors (e_r', e_theta', e_phi') at each cell."""
	rloc = np.sqrt(xl**2 + yl**2 + zl**2)
	r_safe = np.maximum(rloc, 1e-30)
	Rcyl = np.sqrt(xl**2 + yl**2)
	Rcyl_safe = np.maximum(Rcyl, 1e-30)

	e_r = np.stack([xl/r_safe, yl/r_safe, zl/r_safe], axis=-1)
	e_phi = np.stack([-yl/Rcyl_safe, xl/Rcyl_safe, np.zeros_like(xl)], axis=-1)
	# e_theta = e_phi × e_r (right-handed; matches standard spherical)
	e_theta = np.cross(e_phi, e_r)
	return e_r, e_theta, e_phi, Rcyl_safe

def _select_slice_mask(z_over_R_target, xl, yl, zl, bandwidth=0.01, min_pixels=16):
	"""Find mask close to target z/R slice in the *local* disc frame."""
	_, _, _, Rcyl_safe = _spherical_basis_local(xl, yl, zl)
	ratio = zl / Rcyl_safe  # z'/R_cyl'
	# Primary band
	mask = np.isfinite(ratio) & (np.abs(ratio - z_over_R_target) <= bandwidth)

	# Fallback if too few pixels: take the best ~min_pixels closest
	if mask.sum() < min_pixels:
		diff = np.abs(ratio - z_over_R_target)
		k = min_pixels
		flat_idx = np.argpartition(diff.ravel(), k-1)[:k]
		mask = np.zeros_like(ratio, dtype=bool)
		mask.ravel()[flat_idx] = True
	return mask

def velocity_slice_profile(vxyz, r, theta, phi, *, z_over_R=0.1, bandwidth=0.01,
                           i0=np.deg2rad(21.0), f_inc=None, f_pa=None,
                           frame='local_sph', min_pixels=16):
	"""
	Build radial profiles of velocity components at a fixed z/R slice in the local disc frame.

	Parameters
	----------
	vxyz : ndarray
		Shape (nr, ntheta, nphi, 3), global Cartesian velocities [cm/s].
	r, theta, phi : 1D arrays
		Grid centers for spherical coordinates [cgs, rad, rad].
	z_over_R : float
		Target z'/R_cyl' slice value in the local (face-on) frame.
	bandwidth : float
		Acceptable |(z'/R_cyl') - z_over_R| band for selecting cells.
	i0 : float
		Base inclination in radians.
	f_inc, f_pa : callables or None
		Warp profiles: functions of radius in *same units as r* that return delta_i, delta_pa (radians).
		If None, assumes a flat disc (no warp offsets).
	frame : str
		'cart' (global Cartesian),
		'local_cart' (local Cartesian after unwarping),
		'local_sph' (local spherical components v_r', v_theta', v_phi').
	min_pixels : int
		Minimum number of cells per radius; falls back to the closest cells if needed.

	Returns
	-------
	r_au : ndarray
		Radii in au.
	v_mean_kms : (3, nr) ndarray
		Mean of each component across selected cells at each radius [km/s].
	v_std_kms : (3, nr) ndarray
		Std dev of each component across selected cells at each radius [km/s].
	counts : (nr,) ndarray
		Number of cells used per radius.
	"""
	nr = len(r)
	rr, x, y, z = _coords_global(r, theta, phi)

	v_mean = np.full((3, nr), np.nan, dtype=float)
	v_std  = np.full((3, nr), np.nan, dtype=float)
	counts = np.zeros(nr, dtype=int)

	for i in range(nr):
		R = _local_rotation(i0, r[i], f_inc, f_pa)
		# Coordinates at this radius
		coords = np.stack([x[i], y[i], z[i]], axis=-1)  # (ntheta, nphi, 3)
		coords_local = coords @ R.T
		xl, yl, zl = coords_local[..., 0], coords_local[..., 1], coords_local[..., 2]

		# Select cells near desired z'/R_cyl' slice
		mask = _select_slice_mask(z_over_R, xl, yl, zl, bandwidth=bandwidth, min_pixels=min_pixels)
		counts[i] = int(mask.sum())
		if counts[i] == 0:
			continue

		# Velocity components according to requested frame
		Vg = vxyz[i]  # global cartesian cm/s, shape (ntheta, nphi, 3)

		if frame == 'cart':
			comps = [Vg[..., 0][mask], Vg[..., 1][mask], Vg[..., 2][mask]]

		else:
			# Move velocities to local frame
			Vl = Vg @ R.T  # global -> local (row vectors)
			if frame == 'local_cart':
				comps = [Vl[..., 0][mask], Vl[..., 1][mask], Vl[..., 2][mask]]
			elif frame == 'local_sph':
				e_r, e_theta, e_phi, _ = _spherical_basis_local(xl, yl, zl)
				# Project local cartesian velocity onto local spherical basis
				v_r     = np.sum(Vl * e_r, axis=-1)
				v_theta = np.sum(Vl * e_theta, axis=-1)
				v_phi   = np.sum(Vl * e_phi, axis=-1)
				comps = [v_r[mask], v_theta[mask], v_phi[mask]]
			else:
				raise ValueError("frame must be one of: 'cart', 'local_cart', 'local_sph'.")

		for k in range(3):
			v_mean[k, i] = np.nanmean(comps[k])
			v_std[k, i]  = np.nanstd(comps[k])

	# Convert to au and km/s
	r_au = r / au
	v_mean_kms = v_mean / 1e5
	v_std_kms  = v_std / 1e5
	return r_au, v_mean_kms, v_std_kms, counts

def plot_velocity_slice(vxyz, r, theta, phi, *,
						z_over_R=0.1, bandwidth=0.01,
						i0=np.deg2rad(21.0), f_inc=None, f_pa=None,
						frame='local_sph', min_pixels=16,
						figsize=(12, 3.6), outfile=None):
	"""
	Convenience wrapper: computes the slice profile and makes a 3-panel plot.
	"""
	r_au, v_mean, v_std, counts = velocity_slice_profile(
		vxyz, r, theta, phi,
		z_over_R=z_over_R, bandwidth=bandwidth,
		i0=i0, f_inc=f_inc, f_pa=f_pa,
		frame=frame, min_pixels=min_pixels
	)

	# Choose labels
	if frame == 'cart':
		comp_labels = [r"$v_x$", r"$v_y$", r"$v_z$"]
	elif frame == 'local_cart':
		comp_labels = [r"$v_{x'}$", r"$v_{y'}$", r"$v_{z'}$"]
	else:
		comp_labels = [r"$v_{r'}$", r"$v_{\theta'}$", r"$v_{\phi'}$"]

	fig, axs = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)
	for k in range(3):
		axs[k].plot(r_au, v_mean[k], lw=1.5)
		axs[k].fill_between(r_au, v_mean[k]-v_std[k], v_mean[k]+v_std[k], alpha=0.2)
		axs[k].set_xlabel("Radius [au]")
		axs[k].set_ylabel(f"{comp_labels[k]} [km s$^{-1}$]")
		axs[k].grid(True, alpha=0.3)

	fig.suptitle(f"Velocity at slice z/R = {z_over_R:.3f} "
					f"(bandwidth={bandwidth:.3f}, frame='{frame}')",
					fontsize=11)

	if outfile:
		fig.savefig(outfile, dpi=200)
	return fig, axs, (r_au, v_mean, v_std, counts)



def plot_velocity_slice_map(
    vxyz, r, theta, phi, *,
    z_over_R=0.10,
    bandwidth=0.01,
    i0=0.0001, #np.deg2rad(21.0),
    f_inc=None, f_pa=None,
    frame='local_sph',            # 'cart' | 'local_cart' | 'local_sph'
    bins_R=128,
    bins_phi=256,
    R_min=None, R_max=None,       # in cm; defaults computed from data if None
    vlim=None,                    # tuple (vmin, vmax) in km/s, or None for auto
    figsize=(13, 3.8),
    outfile=None,
    show_counts=False             # optionally show a small counts panel
):
	"""
	Make 2D colormaps of velocity components at a fixed z'/R' slice that follows the warp.

	Axes are local cylindrical radius R' (au) vs local azimuth φ' (deg). Values are km/s.

	Parameters
	----------
	vxyz : (nr, ntheta, nphi, 3) ndarray
		Global Cartesian velocities [cm/s].
	r, theta, phi : 1D arrays
		Spherical grid centers [cm, rad, rad].
	z_over_R : float
		Target z'/R'_cyl in the local (face-on) frame.
	bandwidth : float
		Acceptable |(z'/R') - target| for slice selection.
	i0 : float
		Base inclination [rad].
	f_inc, f_pa : callables or None
		Warp profiles: delta_i(r), delta_pa(r) in radians. If None → flat.
	frame : str
		Which components to display: 'cart', 'local_cart', or 'local_sph'.
	bins_R, bins_phi : int
		Number of bins in R' and φ' for the 2D map.
	R_min, R_max : float or None
		Cylindrical-radius range (cm) for the histogram; defaults will be inferred.
	vlim : (vmin, vmax) in km/s
		Fixed color limits (symmetric often looks nice); None = autoscale per panel.
	show_counts : bool
		If True, add a small fourth panel with the log10-counts per bin.

	Returns
	-------
	fig, axs, products
		products is a dict with edges and maps for further processing.
	"""
	nr = len(r)
	rr, x, y, z = _coords_global(r, theta, phi)

	# Accumulate per selected cell across all radii
	R_list   = []  # R' [cm]
	Phi_list = []  # φ' [rad]
	comps_lists = [[], [], []]  # three velocity components

	# First pass: gather values
	for i in range(nr):
		Rloc = _local_rotation(i0, r[i], f_inc, f_pa)

		# Coordinates at this radius → local frame
		coords = np.stack([x[i], y[i], z[i]], axis=-1)                 # (ntheta, nphi, 3)
		coords_local = coords @ Rloc.T
		xl, yl, zl = coords_local[..., 0], coords_local[..., 1], coords_local[..., 2]

		# Slice selector in local frame
		_, _, _, Rcyl_safe = _spherical_basis_local(xl, yl, zl)
		ratio = zl / Rcyl_safe
		mask = np.isfinite(ratio) & (np.abs(ratio - z_over_R) <= bandwidth)
		if not np.any(mask):
			continue

		# Velocities → desired frame
		Vg = vxyz[i]  # global cartesian [cm/s]
		if frame == 'cart':
			comps = [Vg[..., 0][mask], Vg[..., 1][mask], Vg[..., 2][mask]]
		else:
			Vl = Vg @ Rloc.T  # global -> local
			if frame == 'local_cart':
				comps = [Vl[..., 0][mask], Vl[..., 1][mask], Vl[..., 2][mask]]
			elif frame == 'local_sph':
				e_r, e_theta, e_phi, _ = _spherical_basis_local(xl, yl, zl)
				v_r     = np.sum(Vl * e_r, axis=-1)
				v_theta = np.sum(Vl * e_theta, axis=-1)
				v_phi   = np.sum(Vl * e_phi, axis=-1)
				comps = [v_r[mask], v_theta[mask], v_phi[mask]]
			else:
				raise ValueError("frame must be 'cart', 'local_cart', or 'local_sph'.")

		# Local polar coordinates of selected cells
		Rsel  = Rcyl_safe[mask]                # cm
		Phisel = np.arctan2(yl[mask], xl[mask])  # rad (-π, π]

		R_list.append(Rsel)
		Phi_list.append(Phisel)
		for k in range(3):
			comps_lists[k].append(comps[k])

	if len(R_list) == 0:
		raise RuntimeError("No cells matched the requested z'/R' slice; try increasing bandwidth.")

	R_vals = np.concatenate(R_list)
	Phi_vals = np.concatenate(Phi_list)
	comp_vals = [np.concatenate(c) / 1e5 for c in comps_lists]  # to km/s

	# Histogram binning ranges
	if R_min is None:
		R_min = np.nanmax([np.nanmin(R_vals), 0.0])
	if R_max is None:
		R_max = np.nanmax(R_vals)
	phi_min, phi_max = -np.pi, np.pi

	# Edges for (R', φ')
	R_edges = np.linspace(R_min, R_max, bins_R + 1)
	phi_edges = np.linspace(phi_min, phi_max, bins_phi + 1)

	# Counts per bin (for averaging)
	counts, _, _ = np.histogram2d(R_vals, Phi_vals, bins=[R_edges, phi_edges])

	# Weighted sums and means per component
	maps = []
	for k in range(3):
		wsum, _, _ = np.histogram2d(R_vals, Phi_vals, bins=[R_edges, phi_edges], weights=comp_vals[k])
		with np.errstate(invalid='ignore', divide='ignore'):
			mean_map = wsum / counts
		maps.append(mean_map)

	# Plot
	comp_labels = {
		'cart':       [r"$v_x$ [km s$^{-1}$]", r"$v_y$ [km s$^{-1}$]", r"$v_z$ [km s$^{-1}$]"],
		'local_cart': [r"$v_{x'}$ [km s$^{-1}$]", r"$v_{y'}$ [km s$^{-1}$]", r"$v_{z'}$ [km s$^{-1}$]"],
		'local_sph':  [r"$v_{r'}$ [km s$^{-1}$]", r"$v_{\theta'}$ [km s$^{-1}$]", r"$v_{\phi'}$ [km s$^{-1}$]"],
	}[frame]

	# Convert R to au and φ to deg for axes; keep data as-is
	R_edges_au = R_edges / au
	phi_edges_deg = np.rad2deg(phi_edges)

	ncols = 4 if show_counts else 3
	fig, axs = plt.subplots(1, ncols, figsize=figsize, constrained_layout=True)

	# Common plotting helper
	def _plot_panel(ax, Z, title):
		# masked array to hide empty bins
		Zm = np.ma.masked_invalid(Z)
		if vlim is None:
			im = ax.pcolormesh(R_edges_au, phi_edges_deg, Zm.T, 	
				shading='nearest',               # avoids subpixel gaps
				edgecolors='none', linewidth=0,  # no edges to show as white lines
				antialiased=False)
		else:
			im = ax.pcolormesh(R_edges_au, phi_edges_deg, Zm.T, 
				shading='nearest',               # avoids subpixel gaps
				edgecolors='none', linewidth=0,  # no edges to show as white lines
				antialiased=False,               # turn off AA on the quads
				vmin=vlim[0], vmax=vlim[1])
  
		ax.set_xlabel("$R$ [au]")
		ax.set_ylabel("$\phi$ [deg]")
		ax.set_title(title, fontsize=10)
		ax.grid(False)
		cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
		cbar.ax.set_ylabel(title, rotation=90)

	def _plot_panel_imshow(ax, Z, title):
		Zm = np.ma.masked_invalid(Z)
		# make masked bins transparent
		cmap = plt.cm.get_cmap(None).copy()
		cmap.set_bad(alpha=0.0)
		vmin, vmax = (None, None) if vlim is None else vlim
		im = ax.imshow(
			Zm.T, origin='lower', aspect='auto', interpolation='nearest',
			extent=(R_edges_au[0], R_edges_au[-1], phi_edges_deg[0], phi_edges_deg[-1]),
			vmin=vmin, vmax=vmax, cmap=cmap
		)
		ax.set_xlabel("$R$ [au]")
		ax.set_ylabel("$\phi$ [deg]")
		ax.set_title(title, fontsize=10)
		cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
		cbar.ax.set_ylabel(title, rotation=90)

	for k in range(3):
		_plot_panel_imshow(axs[k], maps[k], comp_labels[k])

	if show_counts:
		with np.errstate(divide='ignore'):
			logC = np.log10(counts)
		Zm = np.ma.masked_invalid(logC)
		im = axs[3].pcolormesh(R_edges_au, phi_edges_deg, Zm.T, shading='auto')
		axs[3].set_xlabel("R' [au]")
		axs[3].set_ylabel("φ' [deg]")
		axs[3].set_title(r"log$_{10}$(count)")
		fig.colorbar(im, ax=axs[3], fraction=0.046, pad=0.04)

	fig.suptitle(
		f"Velocity maps at slice z'/R' = {z_over_R:.3f} (bandwidth={bandwidth:.3f}, frame='{frame}')",
		fontsize=11
	)

	if outfile:
		fig.savefig(outfile, dpi=200)

	products = dict(
		R_edges_cm=R_edges,
		phi_edges_rad=phi_edges,
		R_edges_au=R_edges_au,
		phi_edges_deg=phi_edges_deg,
		maps=maps,
		counts=counts
	)
	return fig, axs, products
