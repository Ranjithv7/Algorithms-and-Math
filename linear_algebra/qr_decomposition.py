import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.linalg import qr as scipy_qr
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')

class QRDecomposer:
    """
    Three methods of QR decomposition with detailed mathematical explanations.
    Each method demonstrates different aspects of the orthogonalization process.
    """
    
    @staticmethod
    def gram_schmidt_qr(A: np.ndarray, modified: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gram-Schmidt QR decomposition.
        
        Mathematical Process:
        For matrix A with columns [a1, a2, ..., an]:
        1. u1 = a1, q1 = u1/||u1||
        2. u2 = a2 - <a2,q1>q1, q2 = u2/||u2||
        3. uk = ak - Σ(j=1 to k-1) <ak,qj>qj, qk = uk/||uk||
        
        Why modified GS is better:
        Standard GS: uk = ak - Σ<ak,uj>uj/||uj||²  (uses non-orthogonal uj)
        Modified GS: uk = ak - Σ<ak,qj>qj         (uses orthogonal qj)
        """
        m, n = A.shape
        Q = np.zeros((m, n))
        R = np.zeros((n, n))
        
        for k in range(n):
            # Start with the k-th column of A
            # Mathematical meaning: ak is our target vector to orthogonalize
            uk = A[:, k].copy()
            
            for j in range(k):
                # Compute projection coefficient: <ak, qj>
                # This measures how much ak "overlaps" with the j-th orthogonal direction
                R[j, k] = np.dot(Q[:, j], uk if modified else A[:, k])
                
                # Remove the component of uk in the qj direction
                # Mathematical insight: We're subtracting the projection proj_qj(uk)
                # This ensures uk becomes orthogonal to all previous q vectors
                uk = uk - R[j, k] * Q[:, j]
            
            # Compute the norm of the orthogonalized vector
            # This becomes the diagonal entry of R
            R[k, k] = np.linalg.norm(uk)
            
            if R[k, k] < 1e-10:
                raise ValueError(f"Column {k} is linearly dependent on previous columns")
            
            # Normalize to get the orthonormal vector
            # Mathematical meaning: qk = uk/||uk|| ensures ||qk|| = 1
            Q[:, k] = uk / R[k, k]
        
        return Q, R
    
    @staticmethod
    def householder_qr(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Householder QR decomposition.
        
        Mathematical Process:
        For each column k, construct Householder reflector Hk that zeros 
        entries below the diagonal in that column.
        
        Householder matrix: H = I - 2vv^T where ||v|| = 1
        Choice of v: v = (x - α*e1)/||x - α*e1|| where α = -sign(x1)*||x||
        
        Why this choice of α:
        - Avoids catastrophic cancellation in floating point arithmetic
        - Ensures numerical stability when x1 ≈ ||x||
        """
        m, n = A.shape
        Q = np.eye(m)  # Accumulate Householder reflectors
        R = A.copy().astype(float)
        
        for k in range(min(m-1, n)):
            # Extract the subvector we want to transform
            # Mathematical meaning: x is the k-th column below diagonal
            x = R[k:, k].copy()
            
            if np.linalg.norm(x) < 1e-10:
                continue
            
            # Choose reflection direction to avoid cancellation
            # Mathematical insight: We reflect x onto α*e1 where α = ±||x||
            # The sign choice prevents subtraction of nearly equal numbers
            alpha = -np.sign(x[0]) * np.linalg.norm(x) if x[0] != 0 else -np.linalg.norm(x)
            
            # Construct the reflection vector
            # Mathematical meaning: v points in direction from x to α*e1
            e1 = np.zeros_like(x)
            e1[0] = 1
            v = x - alpha * e1
            v_norm = np.linalg.norm(v)
            
            if v_norm < 1e-10:
                continue
                
            v = v / v_norm  # Normalize: ||v|| = 1
            
            # Apply Householder reflection to remaining submatrix
            # Mathematical formula: H = I - 2vv^T
            # Effect: HR transforms R[k:, k:] so that column k becomes [α, 0, 0, ...]
            R[k:, k:] = R[k:, k:] - 2 * np.outer(v, v @ R[k:, k:])
            
            # Accumulate the reflection in Q
            # Mathematical insight: Q = H1^T * H2^T * ... * Hk^T
            # We build Q^T and transpose at the end
            Q_sub = np.eye(m-k) - 2 * np.outer(v, v)
            Q_full = np.eye(m)
            Q_full[k:, k:] = Q_sub
            Q = Q @ Q_full
        
        return Q, R
    
    @staticmethod
    def givens_qr(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Givens rotation QR decomposition.
        
        Mathematical Process:
        For each entry below diagonal, construct a 2x2 rotation matrix
        that zeros out that specific entry.
        
        Givens rotation: G = [c  s ] where c = a/r, s = b/r, r = sqrt(a² + b²)
                            [-s  c]
        
        Effect: G * [a] = [r]  (zeros out b)
                    [b]   [0]
        """
        m, n = A.shape
        Q = np.eye(m)
        R = A.copy().astype(float)
        
        # Process each column from left to right
        for col in range(min(m-1, n)):
            # Process each subdiagonal entry from bottom to top
            # Why bottom to top? So we don't disturb zeros we've already created
            for row in range(m-1, col, -1):
                # Extract the two elements we want to rotate
                a = R[row-1, col]  # Element above
                b = R[row, col]    # Element to be zeroed
                
                if abs(b) < 1e-10:
                    continue  # Already zero
                
                # Compute Givens rotation parameters
                # Mathematical derivation:
                # We want [c  s] [a] = [r] where r² = a² + b²
                #        [-s c] [b]   [0]
                # This gives: ca + sb = r and -sa + cb = 0
                # Solving: c = a/r, s = b/r
                r = np.sqrt(a**2 + b**2)
                c = a / r  # cosine
                s = b / r  # sine
                
                # Apply rotation to the entire row pair in R
                # Mathematical effect: Rotates rows (row-1) and (row) 
                # so that R[row, col] becomes zero
                R_old = R[[row-1, row], :].copy()
                R[row-1, :] = c * R_old[0, :] + s * R_old[1, :]
                R[row, :] = -s * R_old[0, :] + c * R_old[1, :]
                
                # Accumulate rotation in Q
                # Mathematical insight: Q = G1^T * G2^T * ... * Gk^T
                Q_old = Q[:, [row-1, row]].copy()
                Q[:, row-1] = c * Q_old[:, 0] + s * Q_old[:, 1]
                Q[:, row] = -s * Q_old[:, 0] + c * Q_old[:, 1]
        
        return Q, R

class OrthogonalLayer(nn.Module):
    """
    Neural network layer that maintains orthogonal weight matrices.
    
    Mathematical Foundation:
    Instead of learning arbitrary weights W, we maintain W^T W = I.
    This ensures:
    1. Perfect conditioning: κ(W) = 1
    2. Gradient preservation: ||∂L/∂h_prev|| = ||∂L/∂h_curr||
    3. Information conservation: no information loss through the layer
    """
    
    def __init__(self, input_dim: int, output_dim: int, method: str = 'qr'):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.method = method
        
        # Initialize with QR decomposition of random Gaussian matrix
        # Mathematical insight: This gives us a random orthogonal matrix
        # uniformly distributed on the Stiefel manifold
        init_matrix = torch.randn(output_dim, input_dim)
        Q, R = torch.qr(init_matrix)
        
        if method == 'qr':
            # Store as unconstrained parameter and project to orthogonal
            self.weight_raw = nn.Parameter(init_matrix)
            self.register_buffer('weight', Q)
        elif method == 'householder':
            # Parameterize using Householder reflectors
            # W = H_k * H_{k-1} * ... * H_1 where H_i = I - 2*v_i*v_i^T
            self.num_reflectors = min(input_dim, output_dim)
            self.reflectors = nn.Parameter(torch.randn(self.num_reflectors, input_dim))
        else:
            # Standard parameterization for comparison
            self.weight = nn.Parameter(Q)
    
    def get_orthogonal_weight(self) -> torch.Tensor:
        """
        Extract orthogonal weight matrix using the chosen method.
        """
        if self.method == 'qr':
            # Project raw weights to orthogonal manifold using QR
            # Mathematical operation: W = QR_decomp(W_raw)[0]
            Q, R = torch.qr(self.weight_raw)
            return Q
        
        elif self.method == 'householder':
            # Construct orthogonal matrix as product of Householder reflectors
            # Mathematical formula: W = ∏(I - 2*v_i*v_i^T)
            W = torch.eye(self.input_dim, device=self.reflectors.device)
            
            for i in range(self.num_reflectors):
                v = self.reflectors[i]
                v = v / torch.norm(v)  # Normalize: ||v|| = 1
                # Apply Householder reflection: H = I - 2*v*v^T
                H = torch.eye(self.input_dim, device=v.device) - 2 * torch.outer(v, v)
                W = H @ W
            
            return W[:self.output_dim, :]
        
        else:
            return self.weight
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through orthogonal layer.
        
        Mathematical operation: y = W * x where W^T W = I
        Property preserved: ||y|| = ||x|| (isometry)
        """
        W = self.get_orthogonal_weight()
        return F.linear(x, W)
    
    def orthogonality_error(self) -> float:
        """
        Measure how far the weight matrix is from being orthogonal.
        
        Mathematical measure: ||W^T W - I||_F
        Perfect orthogonality: error = 0
        """
        W = self.get_orthogonal_weight()
        should_be_identity = W @ W.T
        identity = torch.eye(should_be_identity.size(0), device=W.device)
        return torch.norm(should_be_identity - identity, 'fro').item()

class QRNeuralNetwork(nn.Module):
    """
    Deep neural network using QR-based orthogonal layers.
    
    Architecture Philosophy:
    - Use orthogonal transformations to preserve gradient magnitudes
    - Apply nonlinearities that don't destroy orthogonality benefits
    - Include standard layers where orthogonality isn't critical
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, 
                 orthogonal_method: str = 'qr'):
        super().__init__()
        
        self.layers = nn.ModuleList()
        
        # Build network with alternating orthogonal and standard layers
        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            # Use orthogonal layers for main transformations
            # Mathematical rationale: Preserve gradient flow through depth
            self.layers.append(OrthogonalLayer(prev_dim, hidden_dim, orthogonal_method))
            
            # Add batch normalization to center activations
            # Mathematical insight: Keeps activations in the linear regime of nonlinearities
            self.layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Use activation that preserves gradient magnitude
            # ReLU derivative: 1 if x > 0, 0 if x <= 0
            # Expected gradient magnitude ≈ 0.5 (assuming balanced positive/negative)
            self.layers.append(nn.ReLU())
            
            prev_dim = hidden_dim
        
        # Final layer: standard linear (not constrained to be orthogonal)
        # Rationale: Output layer needs flexibility to map to arbitrary target space
        self.output_layer = nn.Linear(prev_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through QR neural network.
        
        Mathematical flow:
        x → Orthogonal_1 → BN → ReLU → ... → Orthogonal_k → BN → ReLU → Linear → output
        
        Gradient flow properties:
        - Each orthogonal layer preserves gradient magnitude
        - Batch normalization prevents activation saturation
        - ReLU provides nonlinearity while maintaining reasonable gradients
        """
        for layer in self.layers:
            x = layer(x)
        return self.output_layer(x)
    
    def total_orthogonality_error(self) -> float:
        """
        Sum of orthogonality errors across all orthogonal layers.
        
        Mathematical interpretation: Measures how well we maintain
        the orthogonal constraint during training.
        """
        total_error = 0.0
        for layer in self.layers:
            if isinstance(layer, OrthogonalLayer):
                total_error += layer.orthogonality_error()
        return total_error
    
    def gradient_norms_by_layer(self, loss: torch.Tensor) -> List[float]:
        """
        Compute gradient norms at each layer to analyze gradient flow.
        
        Mathematical purpose: Detect vanishing/exploding gradients
        Healthy network: Gradient norms should be roughly constant across layers
        """
        loss.backward(retain_graph=True)
        
        norms = []
        for name, param in self.named_parameters():
            if param.grad is not None:
                norms.append(torch.norm(param.grad).item())
        
        return norms

class StandardNeuralNetwork(nn.Module):
    """
    Standard neural network for comparison with QR-based version.
    
    Uses conventional weight initialization and unconstrained optimization.
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
        
        # Xavier initialization (standard approach)
        # Mathematical basis: Keeps variance roughly constant through layers
        # Formula: W ~ N(0, 2/(fan_in + fan_out))
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

def generate_spiral_data(n_samples: int = 1000, noise: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate 2D spiral classification data.
    
    Mathematical description:
    Two interleaved spirals in polar coordinates:
    r = t, θ = t + class*π
    Converted to Cartesian: x = r*cos(θ), y = r*sin(θ)
    
    Why spirals? They require nonlinear decision boundaries that test
    the network's ability to learn complex transformations.
    """
    n_per_class = n_samples // 2
    t = np.linspace(0, 4*np.pi, n_per_class)
    
    # Class 0: spiral starting at angle 0
    x0 = t * np.cos(t) + noise * np.random.randn(n_per_class)
    y0 = t * np.sin(t) + noise * np.random.randn(n_per_class)
    
    # Class 1: spiral starting at angle π (opposite side)
    x1 = t * np.cos(t + np.pi) + noise * np.random.randn(n_per_class)
    y1 = t * np.sin(t + np.pi) + noise * np.random.randn(n_per_class)
    
    # Combine data
    X = np.vstack([np.column_stack([x0, y0]), np.column_stack([x1, y1])])
    y = np.hstack([np.zeros(n_per_class), np.ones(n_per_class)])
    
    # Normalize features for better training stability
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    
    return torch.FloatTensor(X), torch.LongTensor(y)

def train_and_compare():
    """
    Train both QR-based and standard neural networks on spiral classification.
    
    Comparison metrics:
    1. Training convergence speed
    2. Gradient norm stability
    3. Orthogonality preservation (QR network only)
    4. Final accuracy
    """
    
    # Generate training data
    X_train, y_train = generate_spiral_data(2000, noise=0.15)
    X_test, y_test = generate_spiral_data(500, noise=0.15)
    
    # Network architectures
    input_dim = 2
    hidden_dims = [64, 64, 64, 64]  # Deep network to test gradient flow
    output_dim = 2
    
    # Initialize networks
    qr_net = QRNeuralNetwork(input_dim, hidden_dims, output_dim, 'qr')
    standard_net = StandardNeuralNetwork(input_dim, hidden_dims, output_dim)
    
    # Optimizers
    qr_optimizer = torch.optim.Adam(qr_net.parameters(), lr=0.001)
    standard_optimizer = torch.optim.Adam(standard_net.parameters(), lr=0.001)
    
    # Training loop
    n_epochs = 1000
    qr_losses = []
    standard_losses = []
    qr_accuracies = []
    standard_accuracies = []
    orthogonality_errors = []
    
    print("Starting training comparison...")
    print("=" * 60)
    
    for epoch in range(n_epochs):
        # Train QR network
        qr_net.train()
        qr_optimizer.zero_grad()
        qr_outputs = qr_net(X_train)
        qr_loss = F.cross_entropy(qr_outputs, y_train)
        qr_loss.backward()
        qr_optimizer.step()
        
        # Train standard network
        standard_net.train()
        standard_optimizer.zero_grad()
        standard_outputs = standard_net(X_train)
        standard_loss = F.cross_entropy(standard_outputs, y_train)
        standard_loss.backward()
        standard_optimizer.step()
        
        # Record metrics
        qr_losses.append(qr_loss.item())
        standard_losses.append(standard_loss.item())
        
        # Test accuracy
        with torch.no_grad():
            qr_net.eval()
            standard_net.eval()
            
            qr_test_outputs = qr_net(X_test)
            standard_test_outputs = standard_net(X_test)
            
            qr_acc = (qr_test_outputs.argmax(1) == y_test).float().mean().item()
            standard_acc = (standard_test_outputs.argmax(1) == y_test).float().mean().item()
            
            qr_accuracies.append(qr_acc)
            standard_accuracies.append(standard_acc)
            
            # Orthogonality error for QR network
            ortho_error = qr_net.total_orthogonality_error()
            orthogonality_errors.append(ortho_error)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch:4d} | QR Loss: {qr_loss:.4f} | Standard Loss: {standard_loss:.4f}")
            print(f"           | QR Acc:  {qr_acc:.4f} | Standard Acc:  {standard_acc:.4f}")
            print(f"           | Orthogonality Error: {ortho_error:.6f}")
            print("-" * 60)
    
    return {
        'qr_losses': qr_losses,
        'standard_losses': standard_losses,
        'qr_accuracies': qr_accuracies,
        'standard_accuracies': standard_accuracies,
        'orthogonality_errors': orthogonality_errors,
        'qr_net': qr_net,
        'standard_net': standard_net,
        'test_data': (X_test, y_test)
    }

def analyze_gradient_flow(results):
    """
    Analyze gradient flow through the networks to demonstrate
    how QR decomposition prevents vanishing gradients.
    """
    qr_net = results['qr_net']
    standard_net = results['standard_net']
    X_test, y_test = results['test_data']
    
    print("\nGradient Flow Analysis")
    print("=" * 60)
    
    # Compute gradients for both networks
    qr_net.train()
    standard_net.train()
    
    qr_outputs = qr_net(X_test)
    standard_outputs = standard_net(X_test)
    
    qr_loss = F.cross_entropy(qr_outputs, y_test)
    standard_loss = F.cross_entropy(standard_outputs, y_test)
    
    qr_grad_norms = qr_net.gradient_norms_by_layer(qr_loss)
    standard_grad_norms = standard_net.gradient_norms_by_layer(standard_loss)
    
    print("Gradient norms by layer:")
    print("QR Network:      ", [f"{norm:.4f}" for norm in qr_grad_norms[:5]])
    print("Standard Network:", [f"{norm:.4f}" for norm in standard_grad_norms[:5]])
    
    # Measure gradient decay
    qr_decay = qr_grad_norms[0] / qr_grad_norms[-1] if qr_grad_norms[-1] > 0 else float('inf')
    standard_decay = standard_grad_norms[0] / standard_grad_norms[-1] if standard_grad_norms[-1] > 0 else float('inf')
    
    print(f"\nGradient decay ratio (first/last layer):")
    print(f"QR Network:       {qr_decay:.2f}")
    print(f"Standard Network: {standard_decay:.2f}")
    
    if qr_decay < standard_decay:
        print("✓ QR network shows better gradient preservation!")
    else:
        print("! Standard network performed better (unexpected)")

def visualize_results(results):
    """
    Create comprehensive visualizations of the training comparison.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    epochs = range(len(results['qr_losses']))
    
    # Loss comparison
    ax1.plot(epochs, results['qr_losses'], label='QR Network', alpha=0.7)
    ax1.plot(epochs, results['standard_losses'], label='Standard Network', alpha=0.7)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy comparison
    ax2.plot(epochs, results['qr_accuracies'], label='QR Network', alpha=0.7)
    ax2.plot(epochs, results['standard_accuracies'], label='Standard Network', alpha=0.7)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Test Accuracy')
    ax2.set_title('Test Accuracy Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Orthogonality error
    ax3.plot(epochs, results['orthogonality_errors'], color='red', alpha=0.7)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Orthogonality Error')
    ax3.set_title('QR Network: Orthogonality Preservation')
    ax3.grid(True, alpha=0.3)
    
    # Final decision boundaries
    X_test, y_test = results['test_data']
    
    # Create a grid for decision boundary visualization
    h = 0.1
    x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
    y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    grid_points = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
    
    with torch.no_grad():
        results['qr_net'].eval()
        qr_pred = results['qr_net'](grid_points).argmax(1).numpy()
    
    qr_pred = qr_pred.reshape(xx.shape)
    ax4.contourf(xx, yy, qr_pred, alpha=0.3, cmap='RdYlBu')
    scatter = ax4.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='RdYlBu', edgecolors='black')
    ax4.set_xlabel('Feature 1')
    ax4.set_ylabel('Feature 2')
    ax4.set_title('QR Network: Final Decision Boundary')
    
    plt.tight_layout()
    plt.show()
    
    # Print final summary
    print("\nFinal Results Summary")
    print("=" * 60)
    print(f"QR Network Final Accuracy:       {results['qr_accuracies'][-1]:.4f}")
    print(f"Standard Network Final Accuracy: {results['standard_accuracies'][-1]:.4f}")
    print(f"QR Network Final Loss:           {results['qr_losses'][-1]:.4f}")
    print(f"Standard Network Final Loss:     {results['standard_losses'][-1]:.4f}")
    print(f"Final Orthogonality Error:       {results['orthogonality_errors'][-1]:.6f}")

if __name__ == "__main__":
    # Demonstrate QR decomposition methods
    print("QR Decomposition Methods Demonstration")
    print("=" * 60)
    
    # Test matrix
    A = np.array([[1., 2., 3.],
                  [1., 3., 5.],
                  [1., 4., 7.]], dtype=float)
    
    print("Original matrix A:")
    print(A)
    print()
    
    # Test all three methods
    methods = [
        ('Gram-Schmidt (Modified)', QRDecomposer.gram_schmidt_qr),
        ('Householder', QRDecomposer.householder_qr),
        ('Givens', QRDecomposer.givens_qr)
    ]
    
    for name, method in methods:
        print(f"{name} QR Decomposition:")
        Q, R = method(A)
        
        print("Q (orthogonal matrix):")
        print(Q)
        print("R (upper triangular):")
        print(R)
        print("Verification A = QR:")
        print(Q @ R)
        print("Orthogonality check ||Q^T Q - I||:")
        print(np.linalg.norm(Q.T @ Q - np.eye(Q.shape[1])))
        print("-" * 40)
    
    # Run neural network comparison
    print("\nNeural Network Training Comparison")
    results = train_and_compare()
    
    # Analyze gradient flow
    analyze_gradient_flow(results)
    
    # Create visualizations
    visualize_results(results)