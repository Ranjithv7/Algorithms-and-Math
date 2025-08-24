import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap

def visualize_eigenvector_concept():
    """
    Visual demonstration of what eigenvectors actually mean.
    Shows how most vectors get rotated, but eigenvectors only get scaled.
    """
    
    # Create a transformation matrix (stretch more in one direction)
    # This represents any linear transformation: rotation + scaling
    A = np.array([[2.0, 0.5],  # Stretches and slightly rotates
                  [0.5, 1.0]])
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(A)
    print("Matrix A:")
    print(A)
    print(f"\nEigenvalues: {eigenvalues}")
    print(f"Eigenvector 1: {eigenvectors[:, 0]}")
    print(f"Eigenvector 2: {eigenvectors[:, 1]}")
    
    # Create a grid of input vectors
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Show many random vectors and their transformations
    ax1.set_title("Most Vectors: Direction Changes", fontsize=14, fontweight='bold')
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(-3, 3)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linewidth=0.5)
    ax1.axvline(x=0, color='k', linewidth=0.5)
    
    # Random vectors to transform
    np.random.seed(42)
    random_vectors = np.random.randn(2, 8) * 1.5
    
    for i in range(random_vectors.shape[1]):
        v = random_vectors[:, i]
        Av = A @ v
        
        # Original vector (blue)
        ax1.arrow(0, 0, v[0], v[1], head_width=0.1, head_length=0.1, 
                 fc='blue', ec='blue', alpha=0.7, label='Original' if i == 0 else "")
        
        # Transformed vector (red)
        ax1.arrow(0, 0, Av[0], Av[1], head_width=0.1, head_length=0.1, 
                 fc='red', ec='red', alpha=0.7, label='Transformed' if i == 0 else "")
        
        # Show the transformation with a dashed line
        ax1.plot([v[0], Av[0]], [v[1], Av[1]], 'k--', alpha=0.3)
    
    ax1.legend()
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    
    # Plot 2: Show eigenvectors (special directions)
    ax2.set_title("Eigenvectors: Only Length Changes", fontsize=14, fontweight='bold')
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(-3, 3)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linewidth=0.5)
    ax2.axvline(x=0, color='k', linewidth=0.5)
    
    for i in range(2):
        v = eigenvectors[:, i]
        lambda_i = eigenvalues[i]
        Av = A @ v
        
        # Original eigenvector
        ax2.arrow(0, 0, v[0], v[1], head_width=0.1, head_length=0.1, 
                 fc='green', ec='green', linewidth=3, 
                 label=f'Eigenvector {i+1}' if i < 2 else "")
        
        # Transformed eigenvector (should be λv)
        ax2.arrow(0, 0, Av[0], Av[1], head_width=0.1, head_length=0.1, 
                 fc='orange', ec='orange', linewidth=3, linestyle='--',
                 label=f'λ{i+1} × Eigenvector {i+1} = {lambda_i:.2f} × v{i+1}' if i < 2 else "")
        
        # Verify: Av should equal λv
        theoretical_Av = lambda_i * v
        print(f"Eigenvector {i+1}: Av = {Av}, λv = {theoretical_Av}")
        print(f"Difference: {np.linalg.norm(Av - theoretical_Av):.10f}")
    
    ax2.legend()
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    
    # Plot 3: Show the "stretching sheet" analogy
    ax3.set_title("The Stretching Sheet Analogy", fontsize=14, fontweight='bold')
    
    # Create a grid of points (represents a sheet with pattern)
    x = np.linspace(-2, 2, 20)
    y = np.linspace(-2, 2, 20)
    X, Y = np.meshgrid(x, y)
    points = np.vstack([X.ravel(), Y.ravel()])
    
    # Transform all points
    transformed_points = A @ points
    
    # Original grid (blue dots)
    ax3.scatter(points[0, ::4], points[1, ::4], c='blue', alpha=0.6, s=20, label='Original Grid')
    
    # Transformed grid (red dots)
    ax3.scatter(transformed_points[0, ::4], transformed_points[1, ::4], 
               c='red', alpha=0.6, s=20, label='Transformed Grid')
    
    # Show eigenvector directions as the "principal stretching directions"
    for i in range(2):
        v = eigenvectors[:, i] * 2  # Scale for visibility
        lambda_i = eigenvalues[i]
        
        # Draw principal direction
        ax3.plot([-v[0], v[0]], [-v[1], v[1]], 'g-', linewidth=4, alpha=0.8,
                label=f'Principal Direction {i+1} (λ={lambda_i:.2f})')
    
    ax3.set_xlim(-4, 4)
    ax3.set_ylim(-4, 4)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    
    plt.tight_layout()
    plt.show()
    
    return A, eigenvalues, eigenvectors

def demonstrate_practical_meanings():
    """
    Show practical interpretations of eigenvalues in different contexts.
    """
    print("\n" + "="*60)
    print("PRACTICAL MEANINGS OF EIGENVALUES")
    print("="*60)
    
    # Example 1: Population Dynamics
    print("\n1. POPULATION DYNAMICS")
    print("-" * 30)
    
    # Predator-prey model
    # dp/dt = ap - bpq  (prey growth, predation loss)
    # dq/dt = cpq - dq  (predator growth from eating, natural death)
    # Linearized around equilibrium: [dp/dt, dq/dt] = A [p, q]
    
    A_pop = np.array([[0.1, -0.8],   # Prey growth vs predation
                      [0.2,  -0.3]])  # Predator benefit vs death
    
    eigenvals_pop, eigenvecs_pop = np.linalg.eig(A_pop)
    
    print(f"Population matrix A:")
    print(A_pop)
    print(f"Eigenvalues: {eigenvals_pop}")
    
    if np.max(np.real(eigenvals_pop)) > 0:
        print("→ UNSTABLE ECOSYSTEM: Populations will grow without bound")
    elif np.max(np.real(eigenvals_pop)) < 0:
        print("→ STABLE ECOSYSTEM: Populations return to equilibrium")
    else:
        print("→ MARGINALLY STABLE: Populations oscillate")
    
    print(f"Dominant eigenvector: {eigenvecs_pop[:, 0]}")
    print(f"→ Long-term ratio: {eigenvecs_pop[0, 0]/eigenvecs_pop[1, 0]:.2f} prey per predator")
    
    # Example 2: Market Analysis
    print("\n2. FINANCIAL MARKET ANALYSIS")
    print("-" * 30)
    
    # Correlation matrix between stocks
    np.random.seed(42)
    # Simulate correlated stock returns
    market_factor = np.random.randn(252)  # 1 year of daily returns
    stock_returns = np.zeros((5, 252))
    
    # Stock 1-3: Heavily influenced by market (tech stocks)
    stock_returns[0] = 0.8 * market_factor + 0.2 * np.random.randn(252)
    stock_returns[1] = 0.7 * market_factor + 0.3 * np.random.randn(252)
    stock_returns[2] = 0.9 * market_factor + 0.1 * np.random.randn(252)
    
    # Stock 4-5: Less market influence (defensive stocks)
    stock_returns[3] = 0.3 * market_factor + 0.7 * np.random.randn(252)
    stock_returns[4] = 0.2 * market_factor + 0.8 * np.random.randn(252)
    
    correlation_matrix = np.corrcoef(stock_returns)
    eigenvals_market, eigenvecs_market = np.linalg.eig(correlation_matrix)
    
    # Sort by eigenvalue magnitude
    idx = np.argsort(eigenvals_market)[::-1]
    eigenvals_market = eigenvals_market[idx]
    eigenvecs_market = eigenvecs_market[:, idx]
    
    print("Stock correlation matrix eigenvalues:")
    for i, val in enumerate(eigenvals_market):
        print(f"  λ{i+1} = {val:.3f} ({val/np.sum(eigenvals_market)*100:.1f}% of total variance)")
    
    print(f"\nDominant eigenvector (market factor): {eigenvecs_market[:, 0]}")
    print("→ Stocks 1-3 have high loadings (market-sensitive)")
    print("→ Stocks 4-5 have low loadings (defensive)")
    
    # Example 3: Social Network
    print("\n3. SOCIAL NETWORK ANALYSIS")
    print("-" * 30)
    
    # Create a small social network
    # 0: Influencer, 1-3: Friends, 4-6: Followers
    adjacency = np.array([
        [0, 1, 1, 1, 0, 0, 0],  # Influencer connects to friends
        [1, 0, 1, 0, 1, 0, 0],  # Friends connect to each other and followers
        [1, 1, 0, 1, 0, 1, 0],
        [1, 0, 1, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 0, 0],  # Followers only connect up
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0]
    ])
    
    # Compute centrality using dominant eigenvector
    eigenvals_social, eigenvecs_social = np.linalg.eig(adjacency.astype(float))
    dominant_idx = np.argmax(np.real(eigenvals_social))
    centrality = np.abs(np.real(eigenvecs_social[:, dominant_idx]))
    
    print("Social network adjacency matrix:")
    print(adjacency)
    print(f"\nDominant eigenvalue: {np.real(eigenvals_social[dominant_idx]):.3f}")
    print("Eigenvector centrality scores:")
    for i, score in enumerate(centrality):
        print(f"  Person {i}: {score:.3f}")
    
    most_central = np.argmax(centrality)
    print(f"→ Person {most_central} is most influential (highest eigenvector centrality)")

def demonstrate_krylov_intuition():
    """
    Show why Krylov subspaces work so well.
    """
    print("\n" + "="*60)
    print("KRYLOV SUBSPACE INTUITION")
    print("="*60)
    
    # Create a matrix with clear eigenstructure
    # Diagonal matrix with distinct eigenvalues
    eigenvals_true = np.array([5.0, 3.0, 1.0, 0.5, 0.1])
    n = len(eigenvals_true)
    
    # Random orthogonal matrix for eigenvectors
    np.random.seed(42)
    Q, _ = np.linalg.qr(np.random.randn(n, n))
    
    # Construct matrix: A = Q Λ Q^T
    A = Q @ np.diag(eigenvals_true) @ Q.T
    
    print("True eigenvalues:", eigenvals_true)
    
    # Start with random vector
    v0 = np.random.randn(n)
    v0 = v0 / np.linalg.norm(v0)
    
    print(f"Starting vector v0: {v0}")
    
    # Show how Krylov vectors capture eigenspace information
    print("\nKrylov subspace construction:")
    krylov_vectors = [v0]
    
    for k in range(1, 4):
        # Compute A^k v0
        vk = np.linalg.matrix_power(A, k) @ v0
        krylov_vectors.append(vk)
        
        # Project onto eigenspace to see which eigenvectors are captured
        projections = Q.T @ vk  # Coordinates in eigenvector basis
        
        print(f"\nA^{k} v0 projections onto eigenvectors:")
        for i, proj in enumerate(projections):
            print(f"  Eigenvector {i+1} (λ={eigenvals_true[i]}): {proj:.4f}")
        
        # Show that dominant eigenvalues grow fastest
        theoretical = eigenvals_true ** k * (Q.T @ v0)
        print(f"  Theoretical: {theoretical}")
        print(f"  Actual:      {projections}")
        print(f"  Match:       {np.allclose(projections, theoretical)}")
    
    # Demonstrate dimensionality reduction
    print(f"\nDIMENSIONALITY REDUCTION MAGIC:")
    print(f"Original problem: {n}×{n} matrix")
    
    # Build Krylov subspace with just 3 vectors
    K = np.column_stack(krylov_vectors[:3])
    K_orth, _ = np.linalg.qr(K)  # Orthogonalize
    
    # Project A onto Krylov subspace
    A_reduced = K_orth.T @ A @ K_orth
    
    print(f"Krylov subspace: {K_orth.shape[1]}×{K_orth.shape[1]} matrix")
    
    # Find eigenvalues in reduced space
    eigenvals_krylov = np.linalg.eigvals(A_reduced)
    eigenvals_krylov = np.sort(eigenvals_krylov)[::-1]
    
    print(f"True eigenvalues:    {eigenvals_true}")
    print(f"Krylov eigenvalues:  {eigenvals_krylov}")
    print(f"Error in largest 3:  {np.abs(eigenvals_krylov - eigenvals_true[:3])}")
    
    print("\n→ KEY INSIGHT: 3D Krylov subspace captures the 3 largest eigenvalues!")
    print("→ This is why Lanczos method works: extreme eigenvalues appear first")

if __name__ == "__main__":
    # Run all demonstrations
    print("EIGENVALUE INTUITION MASTERCLASS")
    print("="*60)
    
    # Visual demonstration
    A, eigenvalues, eigenvectors = visualize_eigenvector_concept()
    
    # Practical meanings
    demonstrate_practical_meanings()
    
    # Krylov subspace intuition
    demonstrate_krylov_intuition()
    
    print("\n" + "="*60)
    print("KEY TAKEAWAYS:")
    print("="*60)
    print("1. Eigenvectors = Natural directions of transformation")
    print("2. Eigenvalues = How much stretching in each direction")
    print("3. Dominant eigenvalue = Long-term system behavior")
    print("4. Krylov subspaces = Efficient way to find extreme eigenvalues")
    print("5. Applications everywhere: from Google to genetics to finance")
    print("="*60)