import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from calibrator.homography import estimate_homographies, refine_homographies
from calibrator.intrinsic import initialize_intrinsic
from calibrator.extrinsic import initialize_extrinsics
from calibrator.distortion import RadialTangentialDistortion
from calibrator.jacobian import ProjectionJacobian
from utils.math_utils import matrix2euler, euler2matrix, Rt2T, transform, T2vecs

class Calibrator():
    def __init__(self, args):
        self.args = args
        self.calibration_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        self.result = {}
        self.is_wide_angle = args.is_wide_angle
        if not self.is_wide_angle:
            self.distortion_model = RadialTangentialDistortion()
        else:
            self.distortion_model = None
            NotImplementedError("Fisheye calibration is not implemented yet.")

        ### optimization params for non-opencv calibration
        self.proj_jac = ProjectionJacobian(self.distortion_model)



    def register(self, img_paths, all_pt2ds, all_pt3ds):
        self.img_paths = img_paths
        self.all_pt2ds = all_pt2ds
        self.all_pt3ds = all_pt3ds

        self.sample_img = cv2.imread(self.img_paths[0])
        self.img_h, self.img_w = np.shape(self.sample_img)[:2]


    def calibrate(self, use_opencv=True):
        if use_opencv:
            mtx = None
            dist = None
            rvecs = None
            tvecs = None
            flags = cv2.CALIB_RATIONAL_MODEL if self.is_wide_angle else None

            RMSE, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.all_pt3ds,
                                                                self.all_pt2ds,
                                                                (self.img_w, self.img_h),
                                                                mtx, dist, rvecs, tvecs,
                                                                flags,
                                                                self.calibration_criteria)

            new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (self.img_w, self.img_h), 1)
            sample_ud_img = cv2.undistort(self.sample_img, mtx, dist, None, new_mtx)
            self.result["RMSE"] = RMSE
            self.result["K"] = mtx
            self.result["dist"] = dist
            self.result["sample_ud_img"] = sample_ud_img
            self.result["rvecs"] = rvecs
            self.result["tvecs"] = tvecs

            return self.result
        else:
            ### initialization
            K_init, T_kgs_init, dist_init = self._initialize_params()
            RMSE, K, T_kgs, dist = self._refine_params(K_init, T_kgs_init, dist_init)

            new_mtx, roi = cv2.getOptimalNewCameraMatrix(K, dist, (self.img_w, self.img_h), 1)
            sample_ud_img = cv2.undistort(self.sample_img, K, dist, None, new_mtx)

            rvecs = []
            tvecs = []
            for T_kg in T_kgs:
                rvec, tvec =T2vecs(T_kg)
                rvecs.append(rvec)
                tvecs.append(tvec)

            self.result["RMSE"] = RMSE
            self.result["K"] = K
            self.result["dist"] = dist
            self.result["sample_ud_img"] = sample_ud_img
            self.result["rvecs"] = rvecs
            self.result["tvecs"] = tvecs
            return self.result

    def _initialize_params(self, mode="cholesky"):
        Hs_init = estimate_homographies(self.all_pt2ds, self.all_pt3ds)
        Hs_refined = refine_homographies(Hs_init, self.all_pt2ds, self.all_pt3ds) # update not that much...
        K_init = initialize_intrinsic(Hs_refined, mode)
        T_kgs_init = initialize_extrinsics(Hs_refined, K_init)

        dist_init = self.distortion_model.initialize_distortion(K_init, T_kgs_init, self.all_pt2ds, self.all_pt3ds)

        return K_init, T_kgs_init, dist_init

    def _refine_params(self, K_init, T_kgs_init, dist_init):
        all_params = self._compose_params(K_init, T_kgs_init, dist_init)

        max_iters = 100
        lambda_init = 1e-3
        lambda_min = 1e-10
        lambda_max = 1e+10
        error_min = 1e-12

        lm_lambda = lambda_init

        all_pt2ds = np.concatenate(self.all_pt2ds, axis=0)
        for iter in range(max_iters):
            J = self.proj_jac.compute(all_params, self.all_pt3ds)
            JTJ = J.T @ J
            diagJTJ = np.diag(np.diagonal(JTJ))

            all_pt2ds_proj_curr = self._project_all(all_params, self.all_pt3ds)
            r = all_pt2ds.reshape(-1,1) - all_pt2ds_proj_curr.reshape(-1, 1)
            delta = np.linalg.inv(JTJ + lm_lambda*diagJTJ) @ J.T @ r

            error_curr = self._compute_reproj_error(all_params, self.all_pt2ds, self.all_pt3ds)
            error_next = self._compute_reproj_error(all_params + delta, self.all_pt2ds, self.all_pt3ds)

            if error_next < error_curr : # get better:
                all_params += delta
                lm_lambda /= 10
            else:
                lm_lambda *= 10

            if not (lambda_min < lm_lambda < lambda_max) or error_curr < error_min:
                break

        K_final, T_kgs_final, dist_final = self._decompose_params(all_params)
        return error_curr, K_final, T_kgs_final, dist_final

    def _project_all(self, all_params, all_pt3ds):
        K, T_kgs, dist = self._decompose_params(all_params)
        all_pt2ds_reproj = np.empty((0, 2))
        for pt3ds, T_kg in zip(all_pt3ds, T_kgs):
            pt3ds_cam = transform(T_kg, pt3ds)
            pt2ds_d = self.distortion_model.project(K, dist, pt3ds_cam)
            all_pt2ds_reproj = np.vstack([all_pt2ds_reproj, pt2ds_d])
        return all_pt2ds_reproj

    def _compute_reproj_error(self, all_params, all_pt2ds, all_pt3ds, reduction="mean"):
        all_pt2ds_detected = np.concatenate(all_pt2ds, axis=0)
        all_pt2ds_reproj = self._project_all(all_params, all_pt3ds)

        error = np.linalg.norm(all_pt2ds_detected-all_pt2ds_reproj, axis=1)
        if reduction == "sum":
            error = np.sum(error)
        else:
            error = np.mean(error)
        return error

    def _compose_params(self, K_init, T_kgs_init, dist_init):
        alpha, beta, gamma = K_init[0,0], K_init[1,1], K_init[0,1]
        cx, cy = K_init[0,2], K_init[1,2]

        all_params = np.array([alpha, beta, gamma, cx, cy] + list(dist_init)).reshape(-1, 1)
        for T_kg in T_kgs_init:
            R_kg = T_kg[:3, :3]
            t_kg = T_kg[:3, -1]
            rx, ry, rz = matrix2euler(R_kg)
            tx, ty, tz = t_kg
            extrinsic_params = np.array([rx, ry, rz, tx, ty, tz]).reshape(-1,1)
            all_params = np.concatenate([all_params, extrinsic_params], axis=0)
        return all_params

    def _decompose_params(self, all_params):
        num_intrinsic = 5
        num_extrinsic = 6
        if isinstance(all_params, np.ndarray):
            all_params = all_params.ravel()
        num_dist = len(self.distortion_model.get_distortion_symbols())
        extrinsic_start = num_intrinsic + num_dist

        alpha,beta,gamma,cx,cy = all_params[:num_intrinsic]
        K = np.array([
            [alpha, gamma, cx],
            [0, beta, cy],
            [0, 0, 1]])

        T_kgs = []
        for i in range(extrinsic_start, len(all_params), num_extrinsic):
            rx, ry, rz, tx, ty, tz = all_params[i:i + num_extrinsic]
            R = euler2matrix((rx, ry, rz))
            t = (tx, ty, tz)
            T_kgs.append(Rt2T(R, t))

        dist = all_params[num_intrinsic:num_intrinsic + num_dist]
        return K, T_kgs, dist

    def print_results(self):
        print("This print format should be updated later.")
        print("K : ", self.result["K"], "  dist : ", self.result["dist"], "  RMSE : ", self.result["RMSE"])

    def visualize(self):
        ### undistorted img
        before = self.sample_img
        after = self.result["sample_ud_img"]

        canvas = np.concatenate([before, after], axis=1)
        canvas = cv2.resize(canvas, dsize=(0,0), fx=0.5, fy=0.5)
        cv2.imwrite(os.path.join(self.args.vis_dir, "sample_undistorted_img.jpg"), canvas)

        ### error distribution img
        self._display_errors()

    def _get_reproj_error(self, obj_point, img_point, rvec, tvec):
        mtx = self.result["K"]
        dist = self.result["dist"]
        reproj_point, _ = cv2.projectPoints(obj_point, rvec, tvec, mtx, dist)
        reproj_point = reproj_point.reshape(-1, 2)

        error = img_point - reproj_point
        rmse = np.mean(np.sqrt(np.sum(error ** 2, axis=1)))
        return error, rmse

    def _display_errors(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        limit_circle = plt.Circle((0, 0), 1.0, color='r', linestyle='--', fill=False, alpha=0.5)
        ax.add_artist(limit_circle)

        for i, (obj_point, img_point, rvec, tvec) in enumerate(zip(self.all_pt3ds, self.all_pt2ds, self.result["rvecs"], self.result["tvecs"])):
            error, rmse = self._get_reproj_error(obj_point, img_point, rvec, tvec)
            ax.scatter(error[:, 0], error[:, 1], marker='+')


        ax.set_xlabel('x (px)')
        ax.set_ylabel('y (px)')
        ax.set_aspect('equal')

        fig.suptitle("reprojection errors")
        plt.savefig(os.path.join(self.args.vis_dir, "error_distribution.jpg"))