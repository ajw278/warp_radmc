
from mpl_setup import *
import matplotlib.pyplot as plt
from constants import *
import numpy as np
from scipy.interpolate import RegularGridInterpolator


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