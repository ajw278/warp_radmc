import numpy as np
from constants import au, ms, rs
import plot_funcs as plf
import extend_warpprof as ewp
import write_radmc as wrm
import utils as ut

# Monte Carlo parameters
nphot = int(2e6)

# Model parameters (dust density)
RHO0 = 2e-15
r0 = 10 * au
H_R0 = 0.05
flang =0.2

# Radial range
rin = 10 * au
rout = 150*au

# Star parameters
mstar, rstar, tstar = 1.4*ms,2.*rs, 7600.0
pstar = np.array([0., 0., 0.])


#Grid parameters
THETA_OPEN = 0.0 #Change if poles not needed
# Number of grid points -- small for testing
nr, ntheta, nphi = 200,200,200
r_edges = np.geomspace(rin, rout, nr+1)
theta_edges = np.linspace(THETA_OPEN,  np.pi - THETA_OPEN, ntheta+1)
phi_edges = np.linspace(0.0, 2 * np.pi, nphi+1)

# Convert edges to centers
r = (r_edges[:-1]+r_edges[1:])/2.
theta = (theta_edges[:-1]+theta_edges[1:])/2.
phi = (phi_edges[:-1]+phi_edges[1:])/2.

# Warp profile file
WARPFILE = 'mwc758_warpprofile.txt' #'hd135344_warpprofile.txt'
warp_data = np.loadtxt(WARPFILE)  
r_warp, dinc, dpa = warp_data[:,0]*au, warp_data[:,1], warp_data[:,2]

#These are harcoded arbitrary extension values  -- change if needed
dinc2= 0.030
dinc1= 0.035
dinc_ext_lower = np.array([dinc1, dinc2])

dpa2 =0.00
dpa1 = -0.10
dpa_ext_lower = np.array([dpa1, dpa2])

#Create warp profile functions
f_inc, f_pa = ewp.extend_warp_profile(r_warp, dinc, dpa, plot=True, dinc_ext_lower=dinc_ext_lower, dpa_ext_lower=dpa_ext_lower, r_ext_lower=[rin, 30.*au])


# Compute the density and velocity in the warped disc
# This is achieved by rotating the spherical coordinates according to the warp profile
def compute_density_warped(i0=np.deg2rad(21.),  M_star=1.4 * 1.98847e33,G = 6.67430e-8):
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

	# Precompute rotation matrix for face-on view
	l0_antivect=  ut.l_vector(-i0, 0.0, 0.0)
	Rx_minus_i0 = ut.rotation_from_z_to_l(l0_antivect)

	for i in range(nr):
		
		delta_i = f_inc(r[i])
		delta_pa = f_pa(r[i])

		# Rotate x,y,z at this radius
		x_i = x[i]
		y_i = y[i]
		z_i = z[i]
		coords = np.stack([x_i, y_i, z_i], axis=-1)  # shape (ntheta, nphi, 3)
			
		l_vec = ut.l_vector(i0, delta_i, delta_pa)
		Rwarp = ut.rotation_from_z_to_l(l_vec)

		# Rotate x, y, z at this radius with warp
		coords = np.stack([x[i], y[i], z[i]], axis=-1)  # shape (ntheta, nphi, 3)
		coords_warped = coords @ Rwarp.T
		coords_faceon = coords_warped @ Rx_minus_i0.T

		x_rot, y_rot, z_rot = coords_faceon[..., 0], coords_faceon[..., 1], coords_faceon[..., 2]
		R_cyl = np.sqrt(x_rot**2 + y_rot**2)
		
		#Assuming a density profile dropping with 1/radius...
		rho0 = RHO0 * (R_cyl / r0) ** -1.0
		rho[i] = rho0*ut.vertical_density(z_rot, H[i])

		# Compute Keplerian azimuthal velocity
		vk = np.sqrt(G * M_star / np.maximum(R_cyl, 1e-5))  # avoid division by zero
				
		# Unit vector in phi direction (tangential)
		vphi_unit = np.stack([-y_rot / R_cyl, x_rot / R_cyl, np.zeros_like(R_cyl)], axis=-1)
		v_local = vk[..., np.newaxis] * vphi_unit  # (ntheta, nphi, 3)

		# Apply inverse warp (i.e., reverse the rotations)
		v_global = v_local @ Rx_minus_i0 @ Rwarp  # note: right to left multiplication
		vxyz[i]=  v_global

	print(f"Computed density and velocity in warped disc with {nr} radial points.")
	print(f"Warning: velocity computation in warped disc is not yet tested.")
	return rho, vxyz


def run():
	print("Computing warped density in spherical coordinates...")
	rho_sph, v_cart = compute_density_warped()
     
	print("Plotting density slices...")
	plf.plot_bipolar_r_theta_slice(rho_sph,r, theta, phi, phi_value=0.0)

	print("Writing spherical grid...")
	wrm.write_amr_grid_spherical(r_edges, theta_edges, phi_edges)

	print("Writing density...")
	wrm.write_density_spherical(rho_sph)
	inc_gas=False
		
	if inc_gas:
		print('Writing velocity...')
		wrm.write_gas_velocity(v_cart)
		print('Writing CO density...')
		wrm.write_co_number_density(rho_sph, abundance=1e-5)

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
