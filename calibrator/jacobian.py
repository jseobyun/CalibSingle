import sympy
import numpy as np
from utils.math_utils import *

def create_jacobian_block(uv, all_symbols):
    """
    Input:
        uvExpression -- the sympy expression for projection
        derivativeSymbols -- the symbols with which to take the partial derivative
                of the projection expression (left-to-right along column dimension)

    Output:
        jacobianBlockExpr -- matrix containing the expressions for the partial
                derivative of the point projection expression wrt the corresponding
                derivative symbol
    """
    u, v = uv.ravel()
    us = []
    vs = []
    for i, symbol in enumerate(all_symbols):
        us.append(sympy.diff(u, symbol)) # not difference, partial derivative
        vs.append(sympy.diff(v, symbol))

    jacobian_block = sympy.Matrix([us, vs])
    return jacobian_block

def create_lambda_function(jacobian_block, all_symbols):
    f = sympy.lambdify(all_symbols, jacobian_block, "numpy")
    return f

def rearange_jacobian_results(results, N):
    block_values = np.zeros((2*N, results.shape[1]))
    for i in range(results.shape[1]):
        du = results[0,i]
        if isinstance(du, np.ndarray):
            du = du.ravel()
        dv = results[1,i]
        if isinstance(dv, np.ndarray):
            dv = dv.ravel()
        block_values[::2, i] = du
        block_values[1::2, i] = dv
    return block_values

def compute_jacobian(function, K, dist, T_kgs, pt3ds):
    epsilon = 1e-100
    P = list(K) + list(dist) + list(T_kgs)
    P = [p + epsilon for p in P]
    X = np.array(pt3ds[:,0]).reshape(-1,1) + epsilon
    Y = np.array(pt3ds[:,1]).reshape(-1,1) + epsilon
    Z = np.array(pt3ds[:,2]).reshape(-1,1) + epsilon
    N = pt3ds.shape[0]

    results = function(*P, X, Y, Z)
    block_values = rearange_jacobian_results(results, N)
    return block_values


class HomographyJacobian():
    def __init__(self):
        X, Y, _ = get_point_symbols()
        h = get_homography_symbols()
        H = np.array(h, dtype=object).reshape(3,3)

        xy_h = np.array([X, Y, 1], dtype=object).reshape(-1,1) # dtype=object needed
        uv = np.matmul(H, xy_h)
        uv = (uv / uv[2,:])[:2,:] # [2, 1]

        all_symbols = list(h) + [X, Y]
        jacobian_block = create_jacobian_block(uv, h)
        self._jacobian_block_function = create_lambda_function(jacobian_block, all_symbols)

    def compute(self, h, pt3ds):
        """
        Input:
            h -- vector containing the elements of the homography H, where
                    h = (H11, H12, H13, H21, H22, H23, H31, H32, H33)
            modelPoints -- (N, 3) model points which are projected into the camera

        Output:
            J -- the Jacobian matrix containing the partial derivatives of
                    the standard projection of all model points into the sensor
        """
        epsilon = 1e-100
        X = np.array(pt3ds[:,0], dtype=object).reshape(-1,1) + epsilon
        Y = np.array(pt3ds[:,1], dtype=object).reshape(-1,1) + epsilon

        results = self._jacobian_block_function(*h, X, Y)
        N = pt3ds.shape[0]
        block_values = rearange_jacobian_results(results, N)
        return block_values

class ProjectionJacobian():
    def __init__(self, distortion_model):
        self.distortion_model = distortion_model
        uv = self.distortion_model.get_projection_expression()
        self.intrinsic_symbols = get_intrinsic_symbols()
        self.extrinsic_symbols = get_extrinsic_symbols()
        self.distortion_symbols = self.distortion_model.get_distortion_symbols()
        self.point_symbols = get_point_symbols()

        intrinsic_jacobian_block = create_jacobian_block(uv, (self.intrinsic_symbols + self.distortion_symbols))
        self._intrinsic_jacobian_block_function = create_lambda_function(intrinsic_jacobian_block, (self.intrinsic_symbols + self.distortion_symbols + self.extrinsic_symbols + self.point_symbols))

        extrinsic_jacobian_block = create_jacobian_block(uv, self.extrinsic_symbols)
        self._extrinsic_jacobian_block_function = create_lambda_function(extrinsic_jacobian_block, (self.intrinsic_symbols + self.distortion_symbols + self.extrinsic_symbols + self.point_symbols))


    def compute(self, all_params, all_pt3ds):
        if isinstance(all_params, np.ndarray):
            all_params = all_params.ravel()

        num_imgs = len(all_pt3ds)
        intrinsic = all_params[:len(self.intrinsic_symbols)]
        dist = all_params[len(self.intrinsic_symbols):len(self.intrinsic_symbols)+len(self.distortion_symbols)]

        num_total_points = sum([np.shape(pt3ds)[0] for pt3ds in all_pt3ds])
        num_total_params = len(self.intrinsic_symbols) + len(self.distortion_symbols) + num_imgs * len(self.extrinsic_symbols)
        J = np.zeros([2*num_total_points, num_total_params])

        ### helper variable
        # intr + dist + num_imgs * extrinsic
        intr_end = len(intrinsic)
        dist_end = intr_end + len(dist)


        helper_idx = 0
        for i in range(num_imgs):
            pt3ds = all_pt3ds[i]
            extr_start = dist_end + i*len(self.extrinsic_symbols)
            extr_end = extr_start + len(self.extrinsic_symbols)

            extrinsic_i =  all_params[extr_start:extr_end]
            num_points = np.shape(pt3ds)[0]


            jac_intrinsic = self._compute_intrinsic_jacobian(intrinsic, dist, extrinsic_i, pt3ds)
            jac_extrinsic = self._compute_extrinsic_jacobian(intrinsic, dist, extrinsic_i, pt3ds)

            start = helper_idx
            end = helper_idx + 2*num_points
            J[start:end, :dist_end] = jac_intrinsic
            J[start:end, extr_start:extr_end] = jac_extrinsic
            helper_idx += 2*num_points
        return J


    def _compute_intrinsic_jacobian(self, K, dist, T_kg, pt3ds):
        jac_intrinsic = compute_jacobian(self._intrinsic_jacobian_block_function, K, dist, T_kg, pt3ds)
        return jac_intrinsic

    def _compute_extrinsic_jacobian(self, K, dist, T_kg, pt3ds):
        jac_extrinsic = compute_jacobian(self._extrinsic_jacobian_block_function, K, dist, T_kg, pt3ds)
        return jac_extrinsic






