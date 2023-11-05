import numpy as np
from utils.math_utils import Rt2T

def initialize_extrinsics(Hs, K):
    K_inv = np.linalg.inv(K)
    T_kgs = []

    for H in Hs:
        h0, h1, h2 = H[:,0], H[:,1], H[:,2]
        lamb = np.linalg.norm(K_inv @ h0)
        lamb_inv = 1/lamb
        r0 = lamb_inv * K_inv @ h0
        r1 = lamb_inv * K_inv @ h1
        r2 = np.cross(r0, r1)
        t = lamb_inv * K_inv @ h2

        # Q in not in SO3
        Q = np.concatenate([r0[:,None], r1[:,None], r2[:,None]], axis=1)
        R = to_SO3(Q)

        T_kg = Rt2T(R, t)
        T_kgs.append(T_kg)
    return T_kgs

def to_SO3(Q):
    U, S, VT = np.linalg.svd(Q)
    R = U @ VT
    return R

