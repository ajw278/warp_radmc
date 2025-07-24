
def find_structure_surface(rho_sph, r, theta, phi, threshold=1e-20, output='structure_faceon.png'):
	"""
	Find and plot the face-on projection of the structure surface where rho > threshold.

	Parameters
	----------
	rho_sph : ndarray
		3D density array with shape (nr, ntheta, nphi)
	r : ndarray
		Radial grid (nr,)
	theta : ndarray
		Polar angle grid in radians (ntheta,)
	phi : ndarray
		Azimuthal angle grid in radians (nphi,)
	threshold : float
		Density threshold to define the surface
	output : str
		Filename to save the face-on plot
	"""
	nr, ntheta, nphi = rho_sph.shape
	x_list, y_list, theta_list = [], [], []
		
	# Precompute θ index of max density for each (r, φ)
	imax_theta = np.argmax(rho_sph, axis=1)  # shape: (nr, nphi)

	for iphi in range(nphi):
		phi_val = phi[iphi]
		for itheta in range(ntheta):
			# Skip if this θ is less than the max density θ for *all* r at this φ
			if not np.any(itheta >= imax_theta[:, iphi]):
				continue

			for ir in range(nr):
				if rho_sph[ir, itheta, iphi] > threshold:
					
					r_val = r[ir]
					theta_val = theta[itheta]

					# Spherical to Cartesian (face-on view)
					sin_theta = np.sin(theta_val)
					x = r_val * sin_theta * np.cos(phi_val)
					y = r_val * sin_theta * np.sin(phi_val)

					if itheta >= np.amax(imax_theta[:min(ir+1,ntheta-1), iphi]):
						x_list.append(x)
						y_list.append(y)
						theta_list.append(theta_val)

						break  # ← crucial! break r-loop after first match


	# Convert lists to arrays
	x_arr = np.array(x_list)
	y_arr = np.array(y_list)
	theta_arr = np.array(theta_list)

	# Plotting
	plt.figure(figsize=(6, 6))
	sc = plt.scatter(1e3*x_arr / au /150.0, 1e3*y_arr / au/150.0, c=np.degrees(theta_arr), s=5, cmap='viridis', alpha=0.8)
	sc = plt.scatter(1e3*x_arr / au /150.0, 1e3*y_arr / au/150.0, c=np.degrees(theta_arr), s=5, cmap='viridis', alpha=0.8)
	plt.xlabel('x [mas]')
	plt.ylabel('y [mas]')
	plt.xlim([-600., 600.0])
	plt.ylim([-600., 600.0])
	plt.title(f'Structure Surface (ρ > {threshold:.1e})')
	plt.colorbar(sc, label=r'$\theta$ [deg]')
	#plt.axis('equal')
	#plt.grid(True)
	plt.tight_layout()
	plt.savefig(output, dpi=150)
	plt.show()