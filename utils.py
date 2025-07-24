import numpy as np
from scipy.interpolate import RegularGridInterpolator

def interpolate_to_cartesian(rho_sph, r, theta, phi, x_grid, y_grid, z_grid):
	# Create interpolator
	interp = RegularGridInterpolator((r, theta, phi), rho_sph, bounds_error=False, fill_value=0)

	# Convert Cartesian grid to spherical coordinates
	rr = np.sqrt(x_grid**2 + y_grid**2 + z_grid**2)
	tt = np.arccos(z_grid / (rr + 1e-30))  # avoid division by zero
	pp = np.arctan2(y_grid, x_grid) % (2 * np.pi)

	# Stack points for interpolation
	interp_points = np.stack([rr, tt, pp], axis=-1)
	rho_cart = interp(interp_points)
	return rho_cart

def vertical_density(z, H):
    return  np.exp(-(z ** 2) / (2 * H ** 2))


def compute_cell_walls(xc):
	dx = np.diff(xc)
	xw = np.zeros(len(xc) + 1)
	xw[1:-1] = 0.5 * (xc[:-1] + xc[1:])
	xw[0] = xc[0] - dx[0]/2
	xw[-1] = xc[-1] + dx[-1]/2
	return xw



def l_vector(i0, delta_i, delta_PA):
	l0 = l0_vector(i0)
	zhat = np.array([0, 0, 1])
	e_i = np.cross(zhat, l0)
	e_i /= np.linalg.norm(e_i)
	e_PA = np.cross(l0, e_i)
	return l0 + delta_i * e_i + delta_PA * np.sin(i0) * e_PA


def l0_vector(i0, PA0=0.0):
	return np.array([
		-np.sin(i0) * np.sin(PA0),
			np.sin(i0) * np.cos(PA0),
			np.cos(i0)
	])


def rotation_from_z_to_l(lvec):
	lvec = lvec / np.linalg.norm(lvec)
	z = np.array([0, 0, 1])
	v = np.cross(z, lvec)
	s = np.linalg.norm(v)
	c = np.dot(z, lvec)
	if s == 0:
		return np.identity(3)  # no rotation needed
	vx = np.array([[0, -v[2], v[1]],
					[v[2], 0, -v[0]],
					[-v[1], v[0], 0]])
	R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))
	return R



def cartesian_grid():
	x = np.linspace(-sizex, sizex, nx)
	y = np.linspace(-sizey, sizey, ny)
	z = np.linspace(-sizez, sizez, nz)
	return np.meshgrid(x, y, z, indexing='ij')