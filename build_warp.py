import numpy as np
from constants import au, ms, rs
import plot_funcs as plf
import extend_warpprof as ewp
import write_radmc as wrm
import utils as ut


nphot = int(2e6)

i0_def = np.deg2rad(19.4)

RHO0 = 2e-15
r0 = 10 * au
H_R0 = 0.05
flang = 0.3
GAS_TO_DUST = 100.0
CO_ABUNDANCE = 1e-5   # n_CO per mean particle; n_CO/n_H2 ~ 2x this value

rin = 5 * au
rout = 250.0*au

mstar, rstar, tstar = 1.4*ms,2.*rs, 7600.0
pstar = np.array([0., 0., 0.])

THETA_OPEN = 0.0
nr, ntheta, nphi = 200,300,200
r_edges = np.geomspace(rin, rout, nr+1)
theta_edges = np.linspace(THETA_OPEN,  np.pi - THETA_OPEN, ntheta+1)
phi_edges = np.linspace(0.0, 2 * np.pi, nphi+1)

r = (r_edges[:-1]+r_edges[1:])/2.
theta = (theta_edges[:-1]+theta_edges[1:])/2.
phi = (phi_edges[:-1]+phi_edges[1:])/2.

WARPFILE = 'mwc758_warpprofile.txt'
warp_data = np.loadtxt(WARPFILE)
r_warp, dinc, dpa = warp_data[:,0]*au, warp_data[:,1], warp_data[:,2]

# Hardcoded extension values at inner boundary — adjust to match physical prior
dinc2= 0.030
dinc1= 0.035

dinc_ext_lower = np.array([dinc1, dinc2])

dpa2 =0.00
dpa1 = -0.10

dpa_ext_lower = np.array([dpa1, dpa2])

f_inc, f_pa = ewp.extend_warp_profile(r_warp, dinc, dpa, plot=True, dinc_ext_lower=dinc_ext_lower, dpa_ext_lower=dpa_ext_lower, r_ext_lower=[rin, 30.*au])


def compute_density_warped(i0=i0_def,  M_star=mstar,G = 6.67430e-8):
	rho = np.zeros((nr, ntheta, nphi), dtype=np.float64)
	vxyz = np.zeros((nr, ntheta, nphi, 3), dtype=np.float64)
	rr, tt, pp = np.meshgrid(r, theta,phi, indexing='ij')

	H = H_R0 * r0*(r/r0)**(1.+flang)
	H = H[:, np.newaxis, np.newaxis]

	rho0 = RHO0 * (r[:, np.newaxis, np.newaxis] / r0) ** -1.0

	x = rr * np.sin(tt) * np.cos(pp)
	y = rr * np.sin(tt) * np.sin(pp)
	z = rr * np.cos(tt)

	rho = np.zeros((nr, ntheta, nphi), dtype=np.float64)

	for i in range(nr):
		delta_i = f_inc(r[i])
		delta_pa = f_pa(r[i])
		l_vec = ut.l_vector(i0, delta_i, delta_pa)
		R = ut.rotation_from_z_to_l(l_vec)

		# Global -> local (disc-face-on): row vectors use right-multiply by R^T
		coords = np.stack([x[i], y[i], z[i]], axis=-1)
		coords_local = coords @ R.T

		x_loc, y_loc, z_loc = coords_local[..., 0], coords_local[..., 1], coords_local[..., 2]
		R_cyl = np.sqrt(x_loc**2 + y_loc**2)
		R_cyl_safe = np.maximum(R_cyl, 1e-8)  # avoid div-by-zero everywhere

		rho0 = RHO0 * (R_cyl_safe / r0) ** -1.0
		rho[i] = rho0 * ut.vertical_density(z_loc, H[i])

		vk = np.sqrt(G * M_star / R_cyl_safe)
		vphi_hat = np.stack([-y_loc / R_cyl_safe, x_loc / R_cyl_safe, np.zeros_like(R_cyl)], axis=-1)
		v_local = vk[..., None] * vphi_hat

		# Local -> global: row vectors use right-multiply by R (inverse of R^T)
		vxyz[i] = v_local @ R

	print(f"Computed density and velocity in warped disc with {nr} radial points.")
	print(f"Warning: velocity computation in warped disc is not yet tested.")
	return rho, vxyz


def run(inc_gas=True):
	print("Computing warped density in spherical coordinates...")
	rho_sph, v_cart = compute_density_warped(i0=i0_def, M_star=mstar)
	fig, axs, data = plf.plot_velocity_slice(
			v_cart, r, theta, phi,
			z_over_R=0.10,
			bandwidth=0.01,
	     i0=i0_def,
	     f_inc=f_inc, f_pa=f_pa,
	     frame='local_sph',      # 'cart' | 'local_cart' | 'local_sph'
	     outfile="velocity_slice_zOverR0p10.png"
	)
	plf.plot_velocity_slice_map(
    v_cart, r, theta, phi,
    z_over_R=0.10,
    bandwidth=0.01,
    i0=i0_def,
    f_inc=None, f_pa=None,
    frame='local_sph',            # 'cart' | 'local_cart' | 'local_sph'
    bins_R=64,
    bins_phi=64
	)
	print("Plotting density slices...")
	plf.plot_bipolar_r_theta_slice(rho_sph,r, theta, phi, phi_value=0.0)

	print("Writing spherical grid...")
	wrm.write_amr_grid_spherical(r_edges, theta_edges, phi_edges)

	print("Writing density...")
	wrm.write_density_spherical(rho_sph)
		
	if inc_gas:
		print('Writing velocity...')
		wrm.write_gas_velocity(v_cart)
		print('Writing CO density...')
		wrm.write_co_number_density(rho_sph, abundance=CO_ABUNDANCE, gas_to_dust=GAS_TO_DUST)

		print('Writing line input...')
		wrm.write_line_input()

	lam = wrm.write_wavelength_grid()
	wrm.write_stars(lam, mstar, rstar, tstar, pstar)

	print('Writing opacity input...')
	wrm.write_opacity_control()

	print('Writing radmc3d input...')
	wrm.write_radmc3d_inp(nphot)
		
if __name__ == '__main__':
	run()
