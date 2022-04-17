import sympy as sp
import numpy as np

def NCV(q_val, r_val):
    F = np.zeros((4, 4))
    F[0, 2] = 1
    F[1, 3] = 1
    T = sp.symbols('T')
    F = sp.Matrix(F)
    Fi = sp.exp(F * T)
    A = np.array(Fi.subs(T, 1), dtype=np.float32)

    L = np.zeros((4, 2))
    L[2, 0] = 1
    L[3, 1] = 1
    q = sp.symbols('q')
    L = sp.Matrix(L)
    Q = sp.integrate((Fi * L) * q * (Fi * L).T, (T, 0, T))
    Q = np.array(Q.subs([(q, q_val), (T, 1)]), dtype=np.float32)

    C = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]], dtype=np.float32)

    R = r_val * np.eye(2, dtype=np.float32)

    return A, C, Q, R

def RW(q_val, r_val):
    F = np.zeros((4, 4))
    T = sp.symbols('T')
    F = sp.Matrix(F)
    Fi = sp.exp(F * T)
    A = np.array(Fi.subs(T, 1), dtype=np.float32)

    L = np.zeros((4, 2))
    L[0, 0] = 1
    L[1, 1] = 1
    q = sp.symbols('q')
    L = sp.Matrix(L)
    Q = sp.integrate((Fi * L) * q * (Fi * L).T, (T, 0, T))
    Q = np.array(Q.subs([(q, q_val), (T, 1)]), dtype=np.float32)

    C = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]], dtype=np.float32)

    R = r_val * np.eye(2, dtype=np.float32)

    return A, C, Q, R

def NCA(q_val, r_val):
    F = np.vstack([np.hstack([np.zeros((4, 2)), np.eye(4)]), np.zeros((2, 6))]).astype(np.float32)
    T = sp.symbols('T')
    F = sp.Matrix(F)
    Fi = sp.exp(F * T)
    A = np.array(Fi.subs(T, 1), dtype=np.float32)

    L = np.zeros((6, 2))
    L[4, 0] = 1
    L[5, 1] = 1
    q = sp.symbols('q')
    L = sp.Matrix(L)
    Q = sp.integrate((Fi * L) * q * (Fi * L).T, (T, 0, T))
    Q = np.array(Q.subs([(q, q_val), (T, 1)]), dtype=np.float32)

    C = np.array([[1, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0]], dtype=np.float32)

    R = r_val * np.eye(2, dtype=np.float32)

    return A, C, Q, R
