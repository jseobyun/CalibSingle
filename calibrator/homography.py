import numpy as np
from calibrator.jacobian import HomographyJacobian
from utils.math_utils import to_homogeneous, from_homogeneous

def estimate_homographies(all_pt2ds, all_pt3ds):
    Hs = []
    for pt2ds, pt3ds in zip(all_pt2ds, all_pt3ds):
        H = estimate_homography(pt2ds, pt3ds[:, :2])
        Hs.append(H)
    return Hs


def estimate_homography(uv, xy):
    N_uv = compute_normalization(uv)
    N_xy = compute_normalization(xy)

    num_points = np.shape(uv)[0]
    A = np.zeros([2 * num_points, 9], dtype=np.float32)

    for i in range(num_points):
        ui, vi = from_homogeneous(np.matmul(N_uv, to_homogeneous(uv[i])))
        xi, yi = from_homogeneous(np.matmul(N_xy, to_homogeneous(xy[i])))
        A[2 * i, :] = (-xi, -yi, -1, 0, 0, 0, ui * xi, ui * yi, ui)
        A[2 * i + 1, :] = (0, 0, 0, -xi, -yi, -1, vi * xi, vi * yi, vi)

    U, S, VT = np.linalg.svd(A)
    h = VT[-1]
    H_normed = h.reshape(3, 3)
    H = np.linalg.inv(N_uv) @ H_normed @ N_xy
    H /= H[2, 2]
    return H

def compute_normalization(pt2d):
    mu = np.mean(pt2d, axis=0).reshape(1,2) # [N, 2] - >[1, 2]

    trans = np.array([[1, 0, -mu[0, 0]],
                      [0, 1, -mu[0, 1]],
                      [0, 0, 1]], dtype=np.float32)
    pt2d_centered =  pt2d - mu
    magnitude = np.mean(np.linalg.norm(pt2d_centered, axis=1)) #[N,]
    factor = np.sqrt(2) / magnitude

    scale = np.array([[factor, 0, 0],
                      [0, factor, 0],
                      [0, 0, 1]], dtype=np.float32)

    N = np.matmul(scale, trans)
    return N

def refine_homographies(Hs_init, all_pt2ds, all_pt3ds):
    hom_jac = HomographyJacobian()

    Hs_refined = []
    for H_init, pt2ds, pt3ds in zip(Hs_init, all_pt2ds, all_pt3ds):
        H_refined = refine_homography(H_init, pt2ds, pt3ds, hom_jac)
        Hs_refined.append(H_refined)
    return Hs_refined

def refine_homography(H, pt2ds, pt3ds, hom_jac):
    # LM naive implementation
    H_curr = H
    H_flatten_curr = H_curr.ravel()
    max_iters = 20

    lambda_init = 1e-3
    lambda_min = 1e-10
    lambda_max = 1e+10
    error_min = 1e-12

    lm_lambda = lambda_init

    for iter in range(max_iters):
        J = hom_jac.compute(H_flatten_curr, pt3ds)
        JTJ = np.matmul(J.T, J)
        diagJTJ = np.diag(np.diagonal(JTJ))

        pt2ds_proj_curr = from_homogeneous(np.matmul(H_curr, to_homogeneous(pt3ds[:,:2]).T).T)

        r = pt2ds.reshape(-1,1) - pt2ds_proj_curr.reshape(-1,1)
        delta = (np.linalg.inv(JTJ + lm_lambda * diagJTJ) @ J.T @ r).ravel()

        error_curr = compute_error(pt2ds, pt2ds_proj_curr)

        H_flatten_next = H_flatten_curr + delta
        H_next = H_flatten_next.reshape(3, 3)
        pt2ds_proj_next = from_homogeneous(np.matmul(H_next, to_homogeneous(pt3ds[:,:2]).T).T)
        error_next = compute_error(pt2ds, pt2ds_proj_next)

        if error_next < error_curr : # get better
            H_flatten_curr += delta
            lm_lambda /= 10
        else: # get worse
            lm_lambda *= 10
        if not (lambda_min < lm_lambda < lambda_max) or error_curr < error_min:
            break

    H_refined = H_flatten_curr.reshape(3,3)
    H_refined /= H_refined[2,2]
    return H_refined

def compute_error(src, dst, reduction="sum"):
    error = np.linalg.norm(src - dst, axis=1)**2
    if reduction =="sum":
        error = np.sum(error)
    else:
        error = np.mean(error)
    return error

