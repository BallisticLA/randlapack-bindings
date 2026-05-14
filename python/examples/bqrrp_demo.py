"""Five-line demo of randlapack.bqrrp."""
import numpy as np

import randlapack as rl

A = np.random.default_rng(0).standard_normal((2000, 200))
Q, R, J = rl.bqrrp(A)
err_fact = np.linalg.norm(A[:, J] - Q @ R, "fro") / np.linalg.norm(A, "fro")
err_orth = np.linalg.norm(Q.T @ Q - np.eye(Q.shape[1]), "fro")
print(f"||A[:, J] - Q@R||_F / ||A||_F = {err_fact:.2e}")
print(f"||Q^T Q - I||_F             = {err_orth:.2e}")
