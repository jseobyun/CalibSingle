import numpy as np

def initialize_intrinsic(Hs_refined, mode="cholesky"):
    num_hom = len(Hs_refined)
    M = np.zeros([2 * num_hom, 6], dtype=np.float32)
    for i in range(num_hom):
        H = Hs_refined[i]
        M[2 * i, :] = vectorize_homography(H, 0, 1)
        M[2 * i + 1, :] = vectorize_homography(H, 0, 0) - vectorize_homography(H, 1, 1)
    U, S, VT = np.linalg.svd(M)
    b = tuple(VT[-1])

    if mode.lower()=="cholesky":
        A = compute_intrinsic_from_cholesky(b)
    elif mode.lower() == "zhang":
        A = compute_intrinsic_from_zhang(b)
    elif mode.lower() =="burger":
        A = compute_intrinsic_from_burger(b)

    if np.sum(np.isnan(A)) > 0:
        raise ValueError(f"Computed intrinsic matirx contains NaN: \n {A}")
    return A


def vectorize_homography(H, p, q):
    vectors = np.array([[H[0, p] * H[0, q],
                         H[0, p] * H[1, q] + H[1, p] * H[0, q],
                         H[1, p] * H[1, q],
                         H[2, p] * H[0, q] + H[0, p] * H[2, q],
                         H[2, p] * H[1, q] + H[1, p] * H[2, q],
                         H[2, p] * H[2, q]]]).astype(np.float32)
    vectors = vectors.reshape(1, 6)
    return vectors


def compute_intrinsic_from_cholesky(b):
    B0, B1, B2, B3, B4, B5 = b
    # ensure B is positive semi-definite
    sign = +1
    if B0 < 0 or B2 < 0 or B5 < 0:
        sign = -1
    B = sign * np.array([
        [B0, B1, B3],
        [B1, B2, B4],
        [B3, B4, B5],
    ])
    L = np.linalg.cholesky(B)
    A = np.linalg.inv(L.T)
    A /= A[2, 2]
    return A

def compute_intrinsic_from_burger(b):
    """
    Computes the intrinsic matrix from the vector b using the closed
    form solution given in Burger, equations 99 - 104.

    Input:
        b -- vector made up of (B0, B1, B2, B3, B4, B5)^T

    Output:
        A -- intrinsic matrix
    """
    B0, B1, B2, B3, B4, B5 = b

    # eqs 104, 105
    w = B0*B2*B5 - B1**2*B5 - B0*B4**2 + 2*B1*B3*B4 - B2*B3**2
    d = B0*B2 - B1**2

    alpha = np.sqrt(w / (d * B0))          # eq 99
    beta = np.sqrt(w / d**2 * B0)         # eq 100
    gamma = np.sqrt(w / (d**2 * B0)) * B1  # eq 101
    uc = (B1*B4 - B2*B3) / d           # eq 102
    vc = (B1*B3 - B0*B4) / d           # eq 103

    A = np.array([
        [alpha, gamma, uc],
        [0, beta, vc],
        [0, 0,  1],
    ])
    return A

def compute_intrinsic_from_zhang(b):
    """
    Computes the intrinsic matrix from the vector b using the closed
    form solution given in Burger, equations 99 - 104.

    Input:
        b -- vector made up of (B0, B1, B2, B3, B4, B5)^T

    Output:
        A -- intrinsic matrix
    """
    B = vec2symmat(b)
    B11 = B[0,0]
    B12 = B[1,0]
    B13 = B[2,0]
    B22 = B[1,1]
    B23 = B[1,2]
    B33 = B[2,2]

    v0 = (B12 * B13 - B11 * B23) / (B11 * B22 - B12**2)
    lamb = B33 - (B13**2 + v0 * (B12 * B13 - B11 * B23)) / B11
    alpha = np.sqrt(lamb / B11)
    beta = np.sqrt((lamb * B11) / (B11 * B22 - B12**2))
    gamma = -B12 * alpha**2 * beta / lamb
    u0 = gamma * v0 / beta - B13 * alpha**2 / lamb

    A = np.array([
        [alpha, gamma, u0],
        [0, beta, v0],
        [0, 0,  1],
    ])
    return A


def vec2symmat(b):
    B0, B1, B2, B3, B4, B5 = b
    B = np.array([
        [B0, B1, B3],
        [B1, B2, B4],
        [B3, B4, B5],
    ])
    return B