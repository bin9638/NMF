import numpy as np

def nmf_multiplicative_update(V, r, max_iter=1000, tol=1e-4):
    """
    Non-negative Matrix Factorization using Multiplicative Update Rules.

    Parameters:
        V (numpy.ndarray): Input matrix (m x n) to factorize, with non-negative values.
        r (int): Rank (number of latent components).
        max_iter (int): Maximum number of iterations.
        tol (float): Convergence tolerance.

    Returns:
        W (numpy.ndarray): Basis matrix (m x r).
        H (numpy.ndarray): Coefficient matrix (r x n).
    """
    m, n = V.shape

    # Initialize W and H with random non-negative values
    np.random.seed(42)  # For reproducibility
    W = np.random.rand(m, r)
    H = np.random.rand(r, n)

    # Iterative updates
    for i in range(max_iter):
        # Update H
        H_numerator = np.dot(W.T, V)
        H_denominator = np.dot(W.T, np.dot(W, H)) + 1e-9  # Avoid division by zero
        H *= H_numerator / H_denominator

        # Update W
        W_numerator = np.dot(V, H.T)
        W_denominator = np.dot(W, np.dot(H, H.T)) + 1e-9  # Avoid division by zero
        W *= W_numerator / W_denominator

        # Compute reconstruction error
        reconstruction = np.dot(W, H)
        error = np.linalg.norm(V - reconstruction, 'fro')

        if error < tol:
            print(f"Converged at iteration {i+1} with error {error:.4f}")
            break

    return W, H

# Example usage
if __name__ == "__main__":
    # Input matrix (example)
    V = np.array([
        [5, 3, 0, 1],
        [4, 0, 0, 1],
        [1, 1, 0, 5],
        [1, 0, 0, 4],
        [0, 1, 5, 4]
    ], dtype=float)

    # Factorize V into W and H
    rank = 2  # Number of latent components
    W, H = nmf_multiplicative_update(V, rank)

    print("Original Matrix V:")
    print(V)
    print("\nBasis Matrix W:")
    print(W)
    print("\nCoefficient Matrix H:")
    print(H)
    print("\nReconstructed Matrix (W @ H):")
    print(np.dot(W, H))
