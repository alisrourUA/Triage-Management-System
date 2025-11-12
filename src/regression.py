# regression.py
# Manual Multiple Linear Regression (no numpy/sklearn)
# ----------------------------------------------------
# Implements training and prediction of recovery time
# using the Normal Equation: (X^T X)β = X^T y
# Solved via Gaussian elimination (Gauss–Jordan).
# ----------------------------------------------------

# =====================================================
# 1) Matrix utilities
# =====================================================

def matmul(A, B):
    """Multiply matrices A (m×k) and B (k×n) -> (m×n)."""
    m, k = len(A), len(A[0])
    kb, n = len(B), len(B[0])
    if k != kb:
        raise ValueError("Incompatible shapes for matmul")
    C = [[0.0 for _ in range(n)] for _ in range(m)]
    for i in range(m):
        for j in range(n):
            s = 0.0
            for t in range(k):
                s += A[i][t] * B[t][j]
            C[i][j] = s
    return C


def transpose(M):
    """Transpose matrix M (m×n) -> (n×m)."""
    m, n = len(M), len(M[0])
    T = [[0.0 for _ in range(m)] for _ in range(n)]
    for i in range(m):
        for j in range(n):
            T[j][i] = M[i][j]
    return T


def gauss_jordan_solve(A, b):
    """
    Solve A x = b using Gauss-Jordan elimination.
    A is nxn, b is length n → returns list x of length n.
    """
    n = len(A)
    aug = [row[:] + [b[i]] for i, row in enumerate(A)]

    for col in range(n):
        # Find pivot
        pivot = col
        for r in range(col + 1, n):
            if abs(aug[r][col]) > abs(aug[pivot][col]):
                pivot = r
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]

        # Singular check
        if abs(aug[col][col]) < 1e-12:
            raise ValueError("Singular matrix in normal equations")

        # Normalize pivot row
        pivot_val = aug[col][col]
        for j in range(col, n + 1):
            aug[col][j] /= pivot_val

        # Eliminate other rows
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            for j in range(col, n + 1):
                aug[r][j] -= factor * aug[col][j]

    return [aug[i][n] for i in range(n)]



def build_design_matrix(patients, feature_names):
    """
    Build X (with intercept) and y from patient dicts.
    X row = [1, feature1, feature2, ...]
    y = Recovery_Time
    """
    X, y = [], []
    for p in patients:
        y_val = p.get("Recovery_Time", 0)
        row = [1.0]  # this is the intercept
        for f in feature_names:
            try:
                row.append(float(p.get(f, 0)))
            except Exception:
                row.append(0.0)
        X.append(row)
        y.append(float(y_val))
    return X, y


def fit_linear_regression(patients, feature_names):
    """
    Fit regression coefficients for y = b0 + b1*x1 + ... + bk*xk.
    """
    X, y = build_design_matrix(patients, feature_names)
    y_col = [[val] for val in y]

    XT = transpose(X)
    XTX = matmul(XT, X)
    XTy = matmul(XT, y_col)

    rhs = [row[0] for row in XTy]
    beta = gauss_jordan_solve(XTX, rhs)

    return {"features": feature_names[:], "beta": beta}


def predict_one(model, patient):
    """Predict recovery time for a single patient dict."""
    beta = model["beta"]
    feats = model["features"]
    s = beta[0]  # also, intercept
    for i, f in enumerate(feats, start=1):
        try:
            s += beta[i] * float(patient.get(f, 0))
        except Exception:
            continue
    return s


def predict_many(model, patients):
    """Predict for a list of patient dicts."""
    return [predict_one(model, p) for p in patients]


def mae(y_true, y_pred):
    n = len(y_true)
    return sum(abs(y_true[i] - y_pred[i]) for i in range(n)) / n if n else 0.0


def mse(y_true, y_pred):
    n = len(y_true)
    return sum((y_true[i] - y_pred[i]) ** 2 for i in range(n)) / n if n else 0.0


def r2_score(y_true, y_pred):
    n = len(y_true)
    if n == 0:
        return 0.0
    mean_y = sum(y_true) / n
    ss_tot = sum((y_true[i] - mean_y) ** 2 for i in range(n))
    ss_res = sum((y_true[i] - y_pred[i]) ** 2 for i in range(n))
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0


_cached_model = None  # store model in memory to avoid retraining repeatedly

def train_recovery_model(patients):
    global _cached_model
    features = ["Age", "Heart_Rate", "Blood_Pressure", "Oxygen_Level"]
    _cached_model = fit_linear_regression(patients, features)
    return _cached_model


def predict_recovery_time(patient):
    global _cached_model
    if _cached_model is None:
        raise RuntimeError(
            "Recovery model not trained. "
            "Call train_recovery_model(load_dataset().to_dict('records')) first."
        )
    return predict_one(_cached_model, patient)
