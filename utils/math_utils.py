import sympy
import numpy as np
from scipy.spatial.transform import Rotation as Rot

### reformat
def to_homogeneous(v):
    if isinstance(v, tuple) or isinstance(v, list):
        v = np.array(v).reshape(-1, len(v))
    elif len(v.shape) == 1:
        return np.append(v, 1)
    # v is Nx2 or Nx3
    vh = np.hstack((v, np.ones((v.shape[0], 1))))
    return vh

def from_homogeneous(vHom):
    if len(vHom.shape) == 1:
        v = vHom[:-1] / vHom[-1]
    elif len(vHom.shape) == 2:
        v = vHom[:,:-1] / vHom[:,-1:]
    else:
        raise ValueError(f"Unexpected input shape for unhom: {vHom.shape}\n{vHom}")
    return v

def skew(v):
    # v [3,1]
    a = v[:,0]
    vv = np.array([[0, -a[2], a[1]],
                   [a[2], 0, -a[0]],
                   [-a[1], a[0], 0]])
    return vv

def unskew(vHat):
    return np.array([vHat[2,1], vHat[0,2], vHat[1,0]])

### symbolic
def get_point_symbols():
    return tuple(sympy.symbols("X Y Z"))

def get_homography_symbols():
    return tuple(sympy.symbols("H11 H12 H13 H21 H22 H23 H31 H32 H33"))

def get_radial_tangential_distortion_symbols():
    return tuple(sympy.symbols("k1 k2 p1 p2 k3"))

def get_kb4_distortion_symbols():
    return tuple(sympy.symbols("k1 k2 k3 k4"))

def get_intrinsic_symbols():
    return tuple(sympy.symbols("alpha beta gamma cx cy"))

def get_extrinsic_symbols():
    return tuple(sympy.symbols("rx ry rz tx ty tz"))

### rotation and translation

def to_radian(degree):
    return degree / 180.0 * np.pi

def to_degree(radian):
    return radian * 180 / np.pi

def Rt2T(R,t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, -1] = t
    return T

def T2vecs(T):
    R = T[:3, :3]
    t = T[:3, -1]

    rvec = Rot.from_matrix(R).as_rotvec()
    tvec = t
    return rvec, tvec

def apply_T(T, pt3ds):
    if len(np.shape(pt3ds)) == 2:
        return (T[:3, :3] @ pt3ds.T).T + T[:3, -1].reshape(-1,3)
    else: # num 1
        return (T[:3, :3] @ pt3ds) + T[:3, -1]

def exp(w_skewed, is_symbolic=False):
    w = unskew(w_skewed)
    if is_symbolic:
        w_norm = sympy.sqrt(w[0] ** 2 + w[1] ** 2 + w[2] ** 2)
        I = np.eye(3)
        term1 = (w_skewed / w_norm) * sympy.sin(w_norm)
        term2 = ((w_skewed @ w_skewed) / (w_norm ** 2)) * (1 - sympy.cos(w_norm))
    else:
        w_norm = np.linalg.norm(w)

        I = np.eye(3)
        if np.isclose(w_norm, 0):
            term1 = 0
        else:
            term1 = (w_skewed / w_norm) * np.sin(w_norm)
        if np.isclose(w_norm, 0):
            term2 = 0
        else:
            term2 = ((w_skewed @ w_skewed) / (w_norm ** 2)) * (1 - np.cos(w_norm))
    R = I + term1 + term2
    return R

def euler2matrix(euler, is_symbolic=False):
    rx, ry, rz = euler
    wx = np.array([1, 0, 0]).reshape(-1,1)
    wy = np.array([0, 1, 0]).reshape(-1,1)
    wz = np.array([0, 0, 1]).reshape(-1,1)

    if is_symbolic:
        Rx = exp(to_radian(rx) * skew(wx), is_symbolic=is_symbolic)
        Ry = exp(to_radian(ry) * skew(wy), is_symbolic=is_symbolic)
        Rz = exp(to_radian(rz) * skew(wz), is_symbolic=is_symbolic)
    else:
        Rx = exp(np.radians(rx) * skew(wx))
        Ry = exp(np.radians(ry) * skew(wy))
        Rz = exp(np.radians(rz) * skew(wz))
    R = Rz @ Ry @ Rx
    return R

def matrix2euler(matrix):
    R11, R12, R13, R21, R22, R23, R31, R32, R33 = matrix.ravel()
    if not (np.isclose(R31, +1) or np.isclose(R31, -1)):
        theta = -np.arcsin(R31)
        psi = np.arctan2(R32 / np.cos(theta), R33 / np.cos(theta))
        phi = np.arctan2(R21 / np.cos(theta), R11 / np.cos(theta))
    else:
        phi = 0
        if np.isclose(R31, -1):
            theta = np.pi / 2
            psi = phi + np.arctan2(R12, R13)
        else:
            theta = -np.pi / 2
            psi = -phi + np.arctan2(-R12, -R13)

    return tuple(np.degrees((psi, theta, phi)))

### projection : assume pinhole
def transform(T_kg, pt3d):
    bX = from_homogeneous((T_kg @ to_homogeneous(pt3d).T).T)
    return bX

def project2normplane(pt3d):
    pt2d = from_homogeneous(pt3d)
    return pt2d

def project2imgplane(K, T_kg, pt3d):
    pt3d_cam = transform(T_kg, pt3d)
    pt2d_naive = project2normplane(pt3d_cam)
    pt2d = from_homogeneous((K @ to_homogeneous(pt2d_naive).T).T)
    return pt2d