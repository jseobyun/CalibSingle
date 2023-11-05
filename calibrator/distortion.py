import numpy as np
from utils.math_utils import *

class GeneralDistortion():
    def __init__(self):
        self.max_fov = 179.5
        self.make_nan_over_max_fov = False

    def get_projection_expression(self):
        alpha, beta, gamma, cx, cy = get_intrinsic_symbols()
        K = np.array([[alpha, gamma, cx],
                      [0, beta, cy],
                      [0, 0, 1]])
        rx, ry, rz, tx, ty, tz = get_extrinsic_symbols()
        R = euler2matrix((rx, ry, rz), is_symbolic=True)

        T_kg = np.array([[R[0, 0], R[0, 1], R[0, 2], tx],
                         [R[1, 0], R[1, 1], R[1, 2], ty],
                         [R[2, 0], R[2, 1], R[2, 2], tz],
                         [0, 0, 0, 1]])

        X, Y, Z = get_point_symbols()
        pt3d_h = np.array([X, Y, Z, 1]).reshape(-1,1)
        pt3d_h_cam = T_kg @ pt3d_h
        pt3d_cam = from_homogeneous(pt3d_h_cam.T)

        dists = self.get_distortion_symbols()
        uv = self.project(K, dists, pt3d_cam, is_symbolic=True)
        return uv

    def get_distortion_symbols(self):
        return None # this will be overwrited in child class

    def project(self, K, dists, pt3d_cam, is_symbolic=False):

        pt2d_naive = from_homogeneous(pt3d_cam) # projection with identity R, zero t
        pt2d_naive_d = self._distort(pt2d_naive, dists, is_symbolic=is_symbolic)

        pt2d_d = (K[:2, :3] @ to_homogeneous(pt2d_naive_d).T).T
        return pt2d_d

    def _distort(self, pt2d, dists, is_symbolic=False):
        return None # this will be overwrited in child class

    def initialize_distortion(self, K, T_kgs, all_pt2ds, all_pt3ds):
        return None # this will be overwrited in child class


class RadialTangentialDistortion(GeneralDistortion):
    def __init__(self):
        super().__init__()

    def get_distortion_symbols(self):
        return get_radial_tangential_distortion_symbols()

    def _distort(self, pt2d, dists, is_symbolic=False):
        k1, k2, p1, p2, k3 = dists
        if is_symbolic: # only one point because it is symbolic
            xn = pt2d[0, 0]
            yn = pt2d[0, 1]
            r = sympy.sqrt(xn**2 + yn**2)
        else:
            xn = pt2d[:, 0]
            yn = pt2d[:, 1]
            r = np.linalg.norm(pt2d, axis=1)
            if self.make_nan_over_max_fov:
                max_r = np.arctan(np.radians(self.max_fov))
                r [r > max_r] = np.nan

        dist_radial = 1 + k1*(r**2) + k2*(r**4) + k3*(r**6)
        dist_xtangential = 2*p1*xn*yn + p2*(r**2 + 2*(xn**2))
        dist_ytangential = p1*(r**2 + 2*(yn**2)) + 2*p2*xn*yn

        xd = dist_radial * xn + dist_xtangential
        yd = dist_radial * yn + dist_ytangential


        return np.hstack([np.array(xd).reshape(-1,1), np.array(yd).reshape(-1,1)])

    def initialize_distortion(self, K, T_kgs, all_pt2ds, all_pt3ds):
        fx, fy = K[0,0], K[1,1]
        cx, cy = K[0,2], K[1,2]
        num_dists = len(self.get_distortion_symbols())
        D = np.empty([0, num_dists])
        Dd = np.empty([0, 1])

        for i, (pt2ds, pt3ds, T_kg) in enumerate(zip(all_pt2ds, all_pt3ds, T_kgs)):
            for j, (pt2d, pt3d) in enumerate(zip(pt2ds, pt3ds)): # point by point

                pt2d_naive = project2imgplane(np.eye(3), T_kg, pt3d)
                r = np.linalg.norm(pt2d_naive)

                ud, vd = pt2d
                u, v = project2imgplane(K, T_kg, pt3d)

                xn, yn = pt2d_naive.ravel()

                D_ij = np.array([[(u - cx) * r ** 2, (u - cx) * r ** 4, fx * (2 * xn * yn), fx * (r ** 2 + 2 * xn ** 2), (u - cx) * r ** 6],
                                [(v - cy) * r ** 2, (v - cy) * r ** 4, fy * (r ** 2 + 2 * yn ** 2), fy * (2 * xn * yn), (v - cy) * r ** 6]])
                D = np.vstack([D, D_ij])

                Dd_ij = np.array([[ud - u],
                                  [vd - v]])
                Dd = np.vstack([Dd, Dd_ij])
        k = np.linalg.pinv(D) @ Dd
        return tuple(k.ravel())












