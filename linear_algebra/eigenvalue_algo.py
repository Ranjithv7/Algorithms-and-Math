import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.linalg import eigh, eigvals
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

class EigenvalueSolver:
    """
    Collection of eigenvalue algorithms with deep mathematical explanations.
    Each method reveals different aspects of spectral analysis.
    """
    
    @staticmethod
    def power_method(A: np.ndarray, max_iter: int = 1000, tol: float = 1e-10) -> Tuple[float, np.ndarray, List[float]]:
        """
        Power method for finding the dominant eigenvalue and eigenvector.
        
        Mathematical Process:
        1. Start with random vector v₀
        2. Iterate: v_{k+1} = Av_k / ||Av_k||
        3. Eigenvalue estimate: λ ≈ v_k^T A v_k (Rayleigh quotient)
        
        Why this works:
        Any vector can be written as v₀ = Σ cᵢvᵢ where vᵢ are eigenvectors.
        After k iterations: A^k v₀ = Σ cᵢλᵢ^k vᵢ ≈ c₁λ₁^k v₁ (if |λ₁| > |λᵢ|)
        The dominant term overwhelms others as k increases.
        """
        n = A.shape[0]
        
        # Initialize with random vector
        # Mathematical insight: We need a vector with non-zero component 
        # in the dominant eigenspace for convergence
        v = np.random.randn(n)
        v = v / np.linalg.norm(v)  # Normalize to unit length
        
        eigenvalue_history = []
        
        for iteration in range(max_iter):
            # Apply matrix transformation: w = Av
            # Mathematical meaning: This applies the linear transformation A
            # to our current approximation of the dominant eigenvector
            w = A @ v
            
            # Compute eigenvalue estimate using Rayleigh quotient
            # Mathematical formula: R(v) = v^T A v / v^T v
            # For unit vector: R(v) = v^T A v
            # Theorem: R(v) is minimized at eigenvectors and gives eigenvalue
            eigenvalue = v.T @ w
            eigenvalue_history.append(eigenvalue)
            
            # Normalize the result to prevent overflow/underflow
            # Mathematical insight: We care about the direction (eigenvector),
            # not the magnitude. Normalization keeps ||v|| = 1 throughout.
            norm_w = np.linalg.norm(w)
            
            if norm_w < tol:
                raise ValueError("Matrix has eigenvalue zero or near-zero")
            
            v_new = w / norm_w
            
            # Check convergence: measure angle between successive vectors
            # Mathematical convergence criterion: cos(θ) = v_new^T v ≈ 1
            # This is more robust than checking eigenvalue changes
            if abs(v_new.T @ v) > 1 - tol:
                print(f"Power method converged in {iteration + 1} iterations")
                break
                
            v = v_new
        
        return eigenvalue, v, eigenvalue_history
    
    @staticmethod
    def inverse_power_method(A: np.ndarray, shift: float = 0.0, max_iter: int = 1000, 
                           tol: float = 1e-10) -> Tuple[float, np.ndarray]:
        """
        Inverse power method for finding eigenvalue closest to shift.
        
        Mathematical Theory:
        If λ is eigenvalue of A, then 1/(λ-σ) is eigenvalue of (A-σI)^(-1).
        The eigenvalue closest to σ becomes dominant in the inverse.
        
        Implementation: Instead of computing (A-σI)^(-1), we solve
        (A-σI)v_{k+1} = v_k at each iteration.
        """
        n = A.shape[0]
        
        # Form shifted matrix (A - σI)
        # Mathematical purpose: Transform so target eigenvalue becomes dominant
        A_shifted = A - shift * np.eye(n)
        
        # LU factorization for efficient repeated solving
        # Mathematical insight: We'll solve (A-σI)x = b many times
        # LU decomposition allows O(n²) solves instead of O(n³)
        try:
            from scipy.linalg import lu_factor, lu_solve
            lu, piv = lu_factor(A_shifted)
        except:
            raise ValueError(f"Matrix (A - {shift}I) is singular or near-singular")
        
        # Initialize random vector
        v = np.random.randn(n)
        v = v / np.linalg.norm(v)
        
        for iteration in range(max_iter):
            # Solve (A - σI)v_new = v for v_new
            # Mathematical operation: v_new = (A - σI)^(-1) v
            # This applies one step of power method to (A - σI)^(-1)
            v_new = lu_solve((lu, piv), v)
            
            # Normalize
            v_new = v_new / np.linalg.norm(v_new)
            
            # Check convergence
            if abs(v_new.T @ v) > 1 - tol:
                break
                
            v = v_new
        
        # Compute eigenvalue of original matrix
        # Mathematical recovery: If μ is eigenvalue of (A-σI)^(-1),
        # then λ = 1/μ + σ is eigenvalue of A
        shifted_eigenvalue = v.T @ (A_shifted @ v)
        eigenvalue = 1.0 / shifted_eigenvalue + shift if abs(shifted_eigenvalue) > tol else shift
        
        return eigenvalue, v
    
    @staticmethod
    def qr_algorithm(A: np.ndarray, max_iter: int = 1000, tol: float = 1e-12) -> Tuple[np.ndarray, np.ndarray]:
        """
        QR algorithm for finding all eigenvalues and eigenvectors.
        
        Mathematical Foundation:
        1. A₀ = A
        2. Aₖ = QₖRₖ (QR decomposition)
        3. Aₖ₊₁ = RₖQₖ (similarity transformation)
        
        Convergence: Aₖ → upper triangular with eigenvalues on diagonal.
        """
        n = A.shape[0]
        Ak = A.copy().astype(float)
        Q_total = np.eye(n)  # Accumulate eigenvectors
        
        for iteration in range(max_iter):
            # QR decomposition of current matrix
            # Mathematical step: Factor Aₖ = QₖRₖ where Qₖ orthogonal, Rₖ upper triangular
            Q, R = np.linalg.qr(Ak)
            
            # Form next iterate: Aₖ₊₁ = RₖQₖ
            # Mathematical insight: This is similar to Aₖ, so same eigenvalues
            # but better separated on diagonal
            Ak_new = R @ Q
            
            # Accumulate eigenvector transformations
            # Mathematical meaning: Total transformation Q_total = Q₁Q₂...Qₖ
            # gives eigenvectors of original matrix
            Q_total = Q_total @ Q
            
            # Check convergence: off-diagonal entries should approach zero
            # Mathematical criterion: ||Aₖ₊₁ - Aₖ||_F < tolerance
            if np.linalg.norm(Ak_new - Ak, 'fro') < tol:
                print(f"QR algorithm converged in {iteration + 1} iterations")
                break
                
            Ak = Ak_new
        
        # Extract eigenvalues from diagonal
        eigenvalues = np.diag(Ak)
        
        return eigenvalues, Q_total
    
    @staticmethod
    def lanczos_method(A: np.ndarray, num_eigenvalues: int = None, max_iter: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Lanczos method for symmetric matrices.
        
        Mathematical Process:
        Build orthogonal basis for Krylov subspace {v, Av, A²v, ...}
        while maintaining A's action in tridiagonal form.
        
        Three-term recurrence:
        βⱼ₊₁qⱼ₊₁ = Aqⱼ - αⱼqⱼ - βⱼqⱼ₋₁
        
        where αⱼ = qⱼᵀAqⱼ, βⱼ₊₁ = ||Aqⱼ - αⱼqⱼ - βⱼqⱼ₋₁||
        """
        n = A.shape[0]
        
        if num_eigenvalues is None:
            num_eigenvalues = min(n, 10)
        if max_iter is None:
            max_iter = min(n, 100)
        
        # Check symmetry
        if not np.allclose(A, A.T, rtol=1e-10):
            print("Warning: Matrix is not symmetric. Lanczos may not converge properly.")
        
        # Initialize first Lanczos vector
        # Mathematical choice: Random starting vector for Krylov subspace
        q = np.random.randn(n)
        q = q / np.linalg.norm(q)
        
        Q = np.zeros((n, max_iter))  # Store Lanczos vectors
        Q[:, 0] = q
        
        alpha = np.zeros(max_iter)  # Diagonal of tridiagonal matrix
        beta = np.zeros(max_iter)   # Off-diagonal of tridiagonal matrix
        
        # Previous Lanczos vector (initially zero)
        q_prev = np.zeros(n)
        
        for k in range(max_iter - 1):
            # Apply matrix: r = Aq_k
            # Mathematical meaning: Expand Krylov subspace by one dimension
            r = A @ q
            
            # Compute diagonal element: α_k = q_k^T A q_k
            # Mathematical interpretation: Diagonal entry of tridiagonalized A
            alpha[k] = q.T @ r
            
            # Orthogonalize against current and previous vectors
            # Mathematical step: r = r - α_k q_k - β_k q_{k-1}
            # This maintains orthogonality of Lanczos vectors
            r = r - alpha[k] * q - beta[k] * q_prev
            
            # Compute off-diagonal element: β_{k+1} = ||r||
            # Mathematical meaning: Measure of residual after orthogonalization
            beta[k + 1] = np.linalg.norm(r)
            
            # Check for breakdown: if β_{k+1} = 0, we've found invariant subspace
            if beta[k + 1] < 1e-14:
                print(f"Lanczos breakdown at iteration {k+1} (found invariant subspace)")
                max_iter = k + 1
                break
            
            # Update vectors for next iteration
            q_prev = q.copy()  # Store current as previous
            q = r / beta[k + 1]  # Normalize residual as next Lanczos vector
            Q[:, k + 1] = q
        
        # Build tridiagonal matrix T from computed coefficients
        # Mathematical construction: T represents action of A on Krylov subspace
        T = np.diag(alpha[:max_iter]) + np.diag(beta[1:max_iter], 1) + np.diag(beta[1:max_iter], -1)
        
        # Find eigenvalues and eigenvectors of tridiagonal matrix
        # Mathematical efficiency: Tridiagonal eigenvalue problem is O(n²) vs O(n³)
        eigenvalues, eigenvectors_T = eigh(T)
        
        # Transform eigenvectors back to original space
        # Mathematical transformation: If Ty = λy, then A(Qy) = λ(Qy)
        # where Q contains Lanczos vectors
        Q_truncated = Q[:, :max_iter]
        eigenvectors = Q_truncated @ eigenvectors_T
        
        # Return requested number of eigenvalues (typically extreme ones)
        return eigenvalues[:num_eigenvalues], eigenvectors[:, :num_eigenvalues]

class PageRankEngine:
    """
    PageRank implementation using power method.
    
    Mathematical Model:
    PageRank models web surfing as a Markov chain where:
    - Random surfer follows links with probability d
    - Teleports to random page with probability (1-d)
    
    The PageRank vector is the stationary distribution of this Markov chain.
    """
    
    def __init__(self, damping_factor: float = 0.85):
        """
        Initialize PageRank engine.
        
        Parameters:
        damping_factor: Probability of following links vs. random teleportation
        Mathematical range: 0 < d < 1, typically d = 0.85
        """
        self.damping_factor = damping_factor
        
    def build_google_matrix(self, adjacency_matrix: np.ndarray) -> np.ndarray:
        """
        Construct Google matrix from web graph adjacency matrix.
        
        Mathematical Formula:
        G = (1-d)/n * J + d * M
        
        where:
        - J is all-ones matrix (uniform teleportation)
        - M is column-stochastic link matrix
        - d is damping factor
        - n is number of pages
        """
        n = adjacency_matrix.shape[0]
        
        # Convert adjacency matrix to column-stochastic matrix M
        # Mathematical operation: M[i,j] = A[i,j] / out_degree[j]
        # Each column sums to 1 (probability distribution)
        column_sums = adjacency_matrix.sum(axis=0)
        
        # Handle dangling nodes (pages with no outlinks)
        # Mathematical fix: Replace zero columns with uniform distribution
        M = adjacency_matrix.copy().astype(float)
        for j in range(n):
            if column_sums[j] == 0:
                M[:, j] = 1.0 / n  # Uniform distribution for dangling nodes
            else:
                M[:, j] = M[:, j] / column_sums[j]  # Normalize column
        
        # Construct Google matrix
        # Mathematical interpretation: 
        # - (1-d)/n: Probability of random teleportation to any page
        # - d*M: Probability of following links according to link structure
        uniform_matrix = np.ones((n, n)) / n
        google_matrix = (1 - self.damping_factor) * uniform_matrix + self.damping_factor * M
        
        return google_matrix
    
    def compute_pagerank(self, adjacency_matrix: np.ndarray, max_iter: int = 1000, 
                        tol: float = 1e-10) -> Tuple[np.ndarray, List[float]]:
        """
        Compute PageRank using power method.
        
        Mathematical Process:
        1. Build Google matrix G
        2. Find dominant eigenvector of G (corresponds to λ = 1)
        3. This eigenvector is the PageRank distribution
        """
        G = self.build_google_matrix(adjacency_matrix)
        n = G.shape[0]
        
        # Initialize with uniform distribution
        # Mathematical choice: Equal probability for all pages initially
        pagerank = np.ones(n) / n
        
        pagerank_history = []
        
        for iteration in range(max_iter):
            # Power method iteration: v_{k+1} = G * v_k
            # Mathematical meaning: Apply one step of Markov chain evolution
            pagerank_new = G @ pagerank
            
            # Record L1 norm (total probability should remain 1)
            pagerank_history.append(np.linalg.norm(pagerank_new, 1))
            
            # Check convergence using L1 norm of difference
            # Mathematical criterion: ||v_{k+1} - v_k||_1 < tolerance
            if np.linalg.norm(pagerank_new - pagerank, 1) < tol:
                print(f"PageRank converged in {iteration + 1} iterations")
                break
                
            pagerank = pagerank_new
        
        return pagerank, pagerank_history

class SpectralGraphNeuralNetwork(nn.Module):
    """
    Graph Neural Network using spectral (eigenvalue-based) convolutions.
    
    Mathematical Foundation:
    Traditional GNNs use spatial convolutions: h' = σ(Ah)
    Spectral GNNs use eigenvalue decomposition: A = QΛQ^T
    Spectral convolution: h' = σ(Q g(Λ) Q^T h)
    
    where g(Λ) is a learnable function of eigenvalues.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 num_layers: int = 2, max_eigenvalues: int = 50):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.max_eigenvalues = max_eigenvalues
        
        # Learnable spectral filters
        # Mathematical meaning: g(λ) = Σᵢ θᵢ λⁱ (polynomial in eigenvalues)
        self.spectral_weights = nn.ModuleList()
        
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        for i in range(num_layers):
            # Each layer learns coefficients for polynomial spectral filter
            self.spectral_weights.append(
                nn.Parameter(torch.randn(dims[i], dims[i+1], max_eigenvalues))
            )
        
        self.activation = nn.ReLU()
        
        # Cache for graph eigendecomposition
        self.eigenvalues = None
        self.eigenvectors = None
        
    def precompute_spectrum(self, adjacency_matrix: torch.Tensor):
        """
        Precompute eigendecomposition of graph Laplacian.
        
        Mathematical Process:
        1. Compute normalized Laplacian: L = I - D^(-1/2) A D^(-1/2)
        2. Eigendecomposition: L = Q Λ Q^T
        3. Cache eigenvalues and eigenvectors for spectral convolutions
        """
        A = adjacency_matrix.detach().cpu().numpy()
        n = A.shape[0]
        
        # Compute degree matrix
        # Mathematical definition: D[i,i] = Σⱼ A[i,j] (node degree)
        degrees = np.array(A.sum(axis=1)).flatten()
        
        # Handle isolated nodes (degree = 0)
        degrees[degrees == 0] = 1
        
        # Normalized Laplacian: L = I - D^(-1/2) A D^(-1/2)
        # Mathematical properties:
        # - Symmetric and positive semi-definite
        # - Eigenvalues in [0, 2]
        # - Smallest eigenvalue is 0 (with eigenvector proportional to degrees)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))
        L = np.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt
        
        # Compute eigendecomposition using Lanczos method
        # Mathematical benefit: Sparse matrices, only need subset of eigenvalues
        try:
            num_eigs = min(self.max_eigenvalues, n - 1)
            eigenvalues, eigenvectors = EigenvalueSolver.lanczos_method(L, num_eigs)
            
            # Sort by eigenvalue magnitude (smallest first)
            # Mathematical insight: Smaller eigenvalues correspond to 
            # global graph structure, larger ones to local structure
            idx = np.argsort(eigenvalues)
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
        except Exception as e:
            print(f"Lanczos failed, using scipy.linalg.eigh: {e}")
            eigenvalues, eigenvectors = eigh(L)
            eigenvalues = eigenvalues[:self.max_eigenvalues]
            eigenvectors = eigenvectors[:, :self.max_eigenvalues]
        
        # Convert to PyTorch tensors
        self.eigenvalues = torch.FloatTensor(eigenvalues)
        self.eigenvectors = torch.FloatTensor(eigenvectors)
        
        if torch.cuda.is_available():
            self.eigenvalues = self.eigenvalues.cuda()
            self.eigenvectors = self.eigenvectors.cuda()
    
    def spectral_convolution(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        Perform spectral convolution using learned eigenvalue function.
        
        Mathematical Operation:
        1. Transform to spectral domain: x̂ = Q^T x
        2. Apply spectral filter: ŷ = g(Λ) x̂ where g is learned polynomial
        3. Transform back: y = Q ŷ
        
        Overall: y = Q g(Λ) Q^T x
        """
        if self.eigenvalues is None or self.eigenvectors is None:
            raise ValueError("Must call precompute_spectrum() before forward pass")
        
        batch_size, num_nodes, input_dim = x.shape
        output_dim = self.spectral_weights[layer_idx].shape[1]
        
        # Transform to spectral domain: x̂ = Q^T x
        # Mathematical meaning: Decompose node features into eigenmodes
        x_spectral = torch.einsum('nm,bmd->bnd', self.eigenvectors.T, x)
        
        # Apply learnable spectral filter g(λ)
        # Mathematical formula: g(λᵢ) = Σₖ θₖ λᵢᵏ (polynomial of degree K-1)
        num_eigenvalues = len(self.eigenvalues)
        spectral_filter = torch.zeros(input_dim, output_dim, num_eigenvalues, device=x.device)
        
        # Compute polynomial: g(λᵢ) = θ₀ + θ₁λᵢ + θ₂λᵢ² + ...
        for k in range(min(self.max_eigenvalues, num_eigenvalues)):
            if k < self.spectral_weights[layer_idx].shape[2]:
                # Polynomial term: θₖ * λᵢᵏ
                eigenvalue_powers = self.eigenvalues ** k
                spectral_filter[:, :, :num_eigenvalues] += (
                    self.spectral_weights[layer_idx][:, :, k:k+1] * 
                    eigenvalue_powers.unsqueeze(0).unsqueeze(0)
                )
        
        # Apply spectral filter: ŷ = g(Λ) x̂
        # Mathematical operation: Element-wise multiplication in spectral domain
        y_spectral = torch.einsum('bnd,don->bno', x_spectral, spectral_filter)
        
        # Transform back to spatial domain: y = Q ŷ
        # Mathematical meaning: Reconstruct filtered features from eigenmodes
        y = torch.einsum('nm,bmo->bno', self.eigenvectors, y_spectral)
        
        return y
    
    def forward(self, x: torch.Tensor, adjacency_matrix: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through spectral graph neural network.
        
        Mathematical Flow:
        x → Spectral Conv → ReLU → ... → Spectral Conv → output
        
        Each spectral convolution captures different frequency components
        of the graph signal.
        """
        # Precompute spectrum if not already done
        if self.eigenvalues is None:
            self.precompute_spectrum(adjacency_matrix)
        
        # Apply spectral convolutions with activations
        for i in range(self.num_layers):
            x = self.spectral_convolution(x, i)
            if i < self.num_layers - 1:  # No activation after last layer
                x = self.activation(x)
        
        return x

def create_synthetic_graph(num_nodes: int = 100, connection_prob: float = 0.1) -> np.ndarray:
    """
    Create synthetic graph for testing eigenvalue algorithms.
    
    Mathematical Model: Erdős-Rényi random graph
    - Each pair of nodes connected with probability p
    - Expected degree: (n-1)p
    - Giant component emerges when p > 1/(n-1)
    """
    # Generate random adjacency matrix
    # Mathematical process: A[i,j] = 1 with probability p, 0 otherwise
    adjacency = np.random.rand(num_nodes, num_nodes) < connection_prob
    
    # Make symmetric (undirected graph)
    # Mathematical operation: A = (A + A^T) > 0 (logical OR)
    adjacency = adjacency | adjacency.T
    
    # Remove self-loops
    # Mathematical constraint: A[i,i] = 0 for all i
    np.fill_diagonal(adjacency, 0)
    
    return adjacency.astype(float)

def analyze_graph_spectrum(adjacency_matrix: np.ndarray) -> dict:
    """
    Comprehensive spectral analysis of graph.
    
    Mathematical Properties Computed:
    1. Laplacian eigenvalues (connectivity, clustering)
    2. Adjacency eigenvalues (random walk properties)
    3. Spectral gap (expansion properties)
    4. Fiedler vector (graph partitioning)
    """
    n = adjacency_matrix.shape[0]
    
    # Compute degree matrix and Laplacian
    degrees = np.array(adjacency_matrix.sum(axis=1)).flatten()
    D = np.diag(degrees)
    L = D - adjacency_matrix  # Combinatorial Laplacian
    
    # Normalized Laplacian
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(degrees, 1)))
    L_norm = np.eye(n) - D_inv_sqrt @ adjacency_matrix @ D_inv_sqrt
    
    # Compute spectra using different methods
    print("Computing graph spectrum using different eigenvalue algorithms...")
    
    # 1. Full eigendecomposition (for comparison)
    L_eigenvalues = eigvals(L_norm)
    L_eigenvalues = np.sort(L_eigenvalues)
    
    # 2. Lanczos method (for large graphs)
    try:
        lanczos_eigenvalues, lanczos_eigenvectors = EigenvalueSolver.lanczos_method(
            L_norm, num_eigenvalues=min(20, n-1)
        )
    except:
        lanczos_eigenvalues = L_eigenvalues[:20]
        lanczos_eigenvectors = np.eye(n)[:, :20]
    
    # 3. Power method for largest eigenvalue
    try:
        max_eigenvalue, max_eigenvector, _ = EigenvalueSolver.power_method(L_norm)
    except:
        max_eigenvalue = L_eigenvalues[-1]
        max_eigenvector = np.random.randn(n)
    
    # Spectral properties
    spectral_gap = L_eigenvalues[1] - L_eigenvalues[0]  # Connectivity measure
    fiedler_value = L_eigenvalues[1]  # Second smallest eigenvalue
    
    return {
        'full_spectrum': L_eigenvalues,
        'lanczos_spectrum': lanczos_eigenvalues,
        'spectral_gap': spectral_gap,
        'fiedler_value': fiedler_value,
        'max_eigenvalue': max_eigenvalue,
        'connectivity': 1.0 / fiedler_value if fiedler_value > 1e-10 else float('inf'),
        'lanczos_eigenvectors': lanczos_eigenvectors
    }

def pagerank_demonstration():
    """
    Demonstrate PageRank algorithm on synthetic web graph.
    """
    print("=" * 60)
    print("PageRank Demonstration")
    print("=" * 60)
    
    # Create synthetic web graph
    np.random.seed(42)  # For reproducibility
    num_pages = 20
    
    # Create realistic web structure: some pages are hubs
    adjacency = np.zeros((num_pages, num_pages))
    
    # Add random links
    for i in range(num_pages):
        num_outlinks = np.random.poisson(3)  # Average 3 outlinks per page
        targets = np.random.choice(num_pages, size=min(num_outlinks, num_pages-1), replace=False)
        targets = targets[targets != i]  # No self-links
        adjacency[i, targets] = 1
    
    # Create a few authoritative pages (many incoming links)
    authority_pages = [0, 5, 10]
    for auth in authority_pages:
        # Many pages link to authority pages
        linkers = np.random.choice(num_pages, size=num_pages//2, replace=False)
        linkers = linkers[linkers != auth]
        adjacency[linkers, auth] = 1
    
    print(f"Created web graph with {num_pages} pages")
    print(f"Total links: {int(adjacency.sum())}")
    print(f"Average out-degree: {adjacency.sum(axis=1).mean():.2f}")
    
    # Compute PageRank
    pagerank_engine = PageRankEngine(damping_factor=0.85)
    pagerank_scores, history = pagerank_engine.compute_pagerank(adjacency)
    
    # Display results
    print("\nTop 10 pages by PageRank score:")
    ranked_pages = np.argsort(pagerank_scores)[::-1]
    for rank, page in enumerate(ranked_pages[:10]):
        in_degree = adjacency[:, page].sum()
        out_degree = adjacency[page, :].sum()
        print(f"  {rank+1:2d}. Page {page:2d}: PR = {pagerank_scores[page]:.4f} "
              f"(in: {int(in_degree)}, out: {int(out_degree)})")
    
    return adjacency, pagerank_scores

def spectral_gnn_demonstration():
    """
    Demonstrate spectral graph neural network on node classification.
    """
    print("\n" + "=" * 60)
    print("Spectral Graph Neural Network Demonstration")
    print("=" * 60)
    
    # Create synthetic graph and features
    np.random.seed(42)
    torch.manual_seed(42)
    
    num_nodes = 100
    input_dim = 10
    hidden_dim = 16
    output_dim = 3
    
    # Generate graph
    adjacency = create_synthetic_graph(num_nodes, connection_prob=0.1)
    
    # Generate node features (correlated with graph structure)
    eigenvals, eigenvecs = eigh(adjacency)
    
    # Use dominant eigenvectors as feature basis
    # Mathematical insight: Features correlated with graph structure
    # should be easier to classify using spectral methods
    features = np.random.randn(num_nodes, input_dim)
    for i in range(min(3, eigenvecs.shape[1])):
        features[:, i] = eigenvecs[:, -(i+1)]  # Use largest eigenvectors
    
    # Generate labels based on spectral clustering
    # Mathematical approach: Use Fiedler vector for binary partitioning
    fiedler_vector = eigenvecs[:, 1]  # Second smallest eigenvector
    labels = np.zeros(num_nodes, dtype=int)
    labels[fiedler_vector > np.percentile(fiedler_vector, 66.67)] = 2
    labels[(fiedler_vector > np.percentile(fiedler_vector, 33.33)) & 
           (fiedler_vector <= np.percentile(fiedler_vector, 66.67))] = 1
    
    # Convert to PyTorch tensors
    features_tensor = torch.FloatTensor(features).unsqueeze(0)  # Batch dimension
    adjacency_tensor = torch.FloatTensor(adjacency)
    labels_tensor = torch.LongTensor(labels)
    
    # Create and train model
    model = SpectralGraphNeuralNetwork(input_dim, hidden_dim, output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Training spectral GNN on {num_nodes} nodes...")
    print(f"Graph has {int(adjacency.sum()/2)} edges")
    
    # Training loop
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        
        # Forward pass
        output = model(features_tensor, adjacency_tensor)
        output = output.squeeze(0)  # Remove batch dimension
        
        # Compute loss
        loss = criterion(output, labels_tensor)
        loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0:
            with torch.no_grad():
                predictions = output.argmax(dim=1)
                accuracy = (predictions == labels_tensor).float().mean()
                print(f"  Epoch {epoch:3d}: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}")
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        output = model(features_tensor, adjacency_tensor).squeeze(0)
        predictions = output.argmax(dim=1)
        accuracy = (predictions == labels_tensor).float().mean()
        print(f"\nFinal accuracy: {accuracy:.4f}")
    
    return model, adjacency, features, labels

def comprehensive_eigenvalue_demo():
    """
    Comprehensive demonstration of all eigenvalue algorithms.
    """
    print("Comprehensive Eigenvalue Algorithm Demonstration")
    print("=" * 60)
    
    # Test matrix: symmetric with known eigenvalues
    np.random.seed(42)
    n = 5
    A_base = np.random.randn(n, n)
    A = A_base @ A_base.T  # Positive definite symmetric matrix
    
    print("Test Matrix A:")
    print(A)
    print()
    
    # True eigenvalues for comparison
    true_eigenvalues = eigvals(A)
    true_eigenvalues = np.sort(true_eigenvalues)[::-1]  # Descending order
    print("True eigenvalues (descending):")
    print([f"{val:.6f}" for val in true_eigenvalues])
    print()
    
    # 1. Power Method
    print("1. Power Method (largest eigenvalue):")
    max_eigenval, max_eigenvec, history = EigenvalueSolver.power_method(A)
    print(f"   Computed: {max_eigenval:.6f}")
    print(f"   True:     {true_eigenvalues[0]:.6f}")
    print(f"   Error:    {abs(max_eigenval - true_eigenvalues[0]):.2e}")
    print()
    
    # 2. Inverse Power Method
    print("2. Inverse Power Method (smallest eigenvalue):")
    min_eigenval, min_eigenvec = EigenvalueSolver.inverse_power_method(A)
    print(f"   Computed: {min_eigenval:.6f}")
    print(f"   True:     {true_eigenvalues[-1]:.6f}")
    print(f"   Error:    {abs(min_eigenval - true_eigenvalues[-1]):.2e}")
    print()
    
    # 3. QR Algorithm
    print("3. QR Algorithm (all eigenvalues):")
    qr_eigenvals, qr_eigenvecs = EigenvalueSolver.qr_algorithm(A)
    qr_eigenvals = np.sort(qr_eigenvals)[::-1]  # Descending order
    print("   Computed:", [f"{val:.6f}" for val in qr_eigenvals])
    print("   True:    ", [f"{val:.6f}" for val in true_eigenvalues])
    print("   Max error:", f"{np.max(np.abs(qr_eigenvals - true_eigenvalues)):.2e}")
    print()
    
    # 4. Lanczos Method
    print("4. Lanczos Method (subset of eigenvalues):")
    try:
        lanczos_eigenvals, lanczos_eigenvecs = EigenvalueSolver.lanczos_method(A, num_eigenvalues=3)
        lanczos_eigenvals = np.sort(lanczos_eigenvals)[::-1]  # Descending order
        print("   Computed:", [f"{val:.6f}" for val in lanczos_eigenvals])
        print("   True:    ", [f"{val:.6f}" for val in true_eigenvalues[:3]])
        print("   Max error:", f"{np.max(np.abs(lanczos_eigenvals - true_eigenvalues[:3])):.2e}")
    except Exception as e:
        print(f"   Lanczos method failed: {e}")
    print()

if __name__ == "__main__":
    # Run comprehensive demonstrations
    
    # 1. Basic eigenvalue algorithms
    comprehensive_eigenvalue_demo()
    
    # 2. PageRank application
    adjacency, pagerank_scores = pagerank_demonstration()
    
    # 3. Spectral analysis
    print("\n" + "=" * 60)
    print("Graph Spectral Analysis")
    print("=" * 60)
    spectrum_analysis = analyze_graph_spectrum(adjacency)
    print(f"Spectral gap (connectivity): {spectrum_analysis['spectral_gap']:.6f}")
    print(f"Fiedler value: {spectrum_analysis['fiedler_value']:.6f}")
    print(f"Graph connectivity measure: {spectrum_analysis['connectivity']:.2f}")
    print("First 10 Laplacian eigenvalues:")
    print([f"{val:.6f}" for val in spectrum_analysis['full_spectrum'][:10]])
    
    # 4. Spectral GNN
    try:
        model, graph_adj, graph_features, graph_labels = spectral_gnn_demonstration()
        print("\nSpectral GNN training completed successfully!")
    except Exception as e:
        print(f"Spectral GNN demonstration failed: {e}")
    
    print("\n" + "=" * 60)
    print("All eigenvalue algorithm demonstrations completed!")
    print("=" * 60)