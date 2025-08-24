import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import logsumexp
import torch
import torch.nn.functional as F
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class GaussianMixtureEM:
    """
    Expectation-Maximization for Gaussian Mixture Models
    Implemented from mathematical first principles with full understanding
    """
    
    def __init__(self, n_components=3, max_iter=100, tol=1e-6, reg_covar=1e-6):
        """
        Parameters:
        -----------
        n_components : int
            Number of mixture components (clusters)
        max_iter : int  
            Maximum number of EM iterations
        tol : float
            Convergence tolerance for log-likelihood
        reg_covar : float
            Regularization for covariance matrices (numerical stability)
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        
        # These will store our learned parameters
        self.weights_ = None      # π_k (mixing coefficients)
        self.means_ = None        # μ_k (cluster means)  
        self.covariances_ = None  # Σ_k (cluster covariances)
        
        # Convergence tracking
        self.log_likelihood_history_ = []
        self.n_iter_ = 0
        self.converged_ = False
        
    def _initialize_parameters(self, X):
        """
        Initialize parameters using K-means++ strategy
        
        Why this approach?
        - Random initialization often leads to poor local optima
        - K-means gives sensible starting points for cluster centers
        - Small random covariances prevent singular matrices
        """
        n_samples, n_features = X.shape
        
        # Use K-means to initialize means intelligently
        kmeans = KMeans(n_clusters=self.n_components, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        
        # Initialize mixing coefficients (π_k)
        # Why uniform? No prior knowledge about cluster sizes
        self.weights_ = np.ones(self.n_components) / self.n_components
        
        # Initialize means from K-means centroids
        self.means_ = kmeans.cluster_centers_
        
        # Initialize covariances
        self.covariances_ = np.zeros((self.n_components, n_features, n_features))
        
        for k in range(self.n_components):
            # Find points assigned to cluster k
            cluster_points = X[cluster_labels == k]
            
            if len(cluster_points) > 1:
                # Use sample covariance of assigned points
                self.covariances_[k] = np.cov(cluster_points.T)
            else:
                # Fallback: identity matrix scaled by data variance
                self.covariances_[k] = np.eye(n_features) * np.var(X, axis=0).mean()
            
            # Add regularization for numerical stability
            # Why needed? Prevents singular matrices during optimization
            self.covariances_[k] += self.reg_covar * np.eye(n_features)
    
    def _multivariate_gaussian_log_pdf(self, X, mean, covariance):
        """
        Compute log PDF of multivariate Gaussian
        
        Mathematical formula:
        log p(x|μ,Σ) = -0.5 * [(x-μ)ᵀ Σ⁻¹ (x-μ) + log|Σ| + d*log(2π)]
        
        Why compute in log space?
        - Numerical stability: avoids underflow for small probabilities
        - Easier arithmetic: products become sums
        """
        n_samples, n_features = X.shape
        
        # Center the data: (x - μ)
        diff = X - mean
        
        # Compute precision matrix: Σ⁻¹ 
        # Why use solve instead of inv? More numerically stable
        try:
            precision = np.linalg.inv(covariance)
            log_det = np.linalg.slogdet(covariance)[1]
        except np.linalg.LinAlgError:
            # Fallback for singular matrices
            precision = np.linalg.pinv(covariance)
            sign, log_det = np.linalg.slogdet(covariance + 1e-6 * np.eye(n_features))
            
        # Quadratic form: (x-μ)ᵀ Σ⁻¹ (x-μ)
        quadratic_form = np.sum(diff @ precision * diff, axis=1)
        
        # Complete log probability
        log_prob = -0.5 * (quadratic_form + log_det + n_features * np.log(2 * np.pi))
        
        return log_prob
    
    def _e_step(self, X):
        """
        E-Step: Compute posterior probabilities γ_ik = P(z_i = k | x_i, θ)
        
        Mathematical insight:
        This is Bayes' rule applied to mixture components
        γ_ik ∝ π_k * p(x_i | μ_k, Σ_k)
        """
        n_samples = X.shape[0]
        
        # Log probabilities for numerical stability
        log_responsibilities = np.zeros((n_samples, self.n_components))
        
        for k in range(self.n_components):
            # Log of: π_k * p(x_i | μ_k, Σ_k)
            log_prob_k = self._multivariate_gaussian_log_pdf(
                X, self.means_[k], self.covariances_[k]
            )
            log_responsibilities[:, k] = np.log(self.weights_[k]) + log_prob_k
        
        # Normalize using log-sum-exp trick for numerical stability
        # Why log-sum-exp? Avoids overflow when exponentiating large log values
        log_norm = logsumexp(log_responsibilities, axis=1, keepdims=True)
        log_responsibilities -= log_norm
        
        # Convert back to probabilities
        responsibilities = np.exp(log_responsibilities)
        
        return responsibilities
    
    def _m_step(self, X, responsibilities):
        """
        M-Step: Update parameters to maximize Q(θ|θ^(t))
        
        Mathematical formulas derived from setting gradients to zero:
        π_k = (1/n) * Σ_i γ_ik
        μ_k = Σ_i γ_ik * x_i / Σ_i γ_ik  
        Σ_k = Σ_i γ_ik * (x_i - μ_k)(x_i - μ_k)ᵀ / Σ_i γ_ik
        """
        n_samples, n_features = X.shape
        
        # Effective number of points assigned to each component
        # This is Σ_i γ_ik for each k
        n_k = responsibilities.sum(axis=0)
        
        # Update mixing coefficients (π_k)
        # Why this formula? It's the average responsibility of component k
        self.weights_ = n_k / n_samples
        
        # Update means (μ_k)
        # Weighted average where weights are responsibilities
        for k in range(self.n_components):
            if n_k[k] > 1e-8:  # Avoid division by very small numbers
                self.means_[k] = np.sum(responsibilities[:, k:k+1] * X, axis=0) / n_k[k]
            # If n_k[k] ≈ 0, keep the previous mean (component is dying)
        
        # Update covariances (Σ_k)
        for k in range(self.n_components):
            if n_k[k] > 1e-8:
                # Centered data: (x_i - μ_k)
                diff = X - self.means_[k]
                
                # Weighted covariance matrix
                weighted_cov = np.zeros((n_features, n_features))
                for i in range(n_samples):
                    weighted_cov += responsibilities[i, k] * np.outer(diff[i], diff[i])
                
                self.covariances_[k] = weighted_cov / n_k[k]
                
                # Add regularization for numerical stability
                self.covariances_[k] += self.reg_covar * np.eye(n_features)
    
    def _compute_log_likelihood(self, X):
        """
        Compute the complete log-likelihood: Σ_i log p(x_i | θ)
        
        This is what EM is maximizing!
        """
        n_samples = X.shape[0]
        log_likelihood = 0
        
        for i in range(n_samples):
            # For each data point, compute log of mixture probability
            log_mixture_prob = -np.inf
            
            for k in range(self.n_components):
                log_component_prob = (
                    np.log(self.weights_[k]) + 
                    self._multivariate_gaussian_log_pdf(
                        X[i:i+1], self.means_[k], self.covariances_[k]
                    )[0]
                )
                
                # Log-sum-exp for numerical stability
                if log_mixture_prob == -np.inf:
                    log_mixture_prob = log_component_prob
                else:
                    log_mixture_prob = np.logaddexp(log_mixture_prob, log_component_prob)
            
            log_likelihood += log_mixture_prob
        
        return log_likelihood
    
    def fit(self, X, verbose=True):
        """
        Fit the Gaussian Mixture Model using EM algorithm
        
        Returns the history of parameters for visualization
        """
        self._initialize_parameters(X)
        
        # Store parameter history for visualization
        self.parameter_history_ = {
            'weights': [self.weights_.copy()],
            'means': [self.means_.copy()],
            'covariances': [self.covariances_.copy()]
        }
        
        prev_log_likelihood = -np.inf
        
        for iteration in range(self.max_iter):
            # E-Step: Compute responsibilities
            responsibilities = self._e_step(X)
            
            # M-Step: Update parameters
            self._m_step(X, responsibilities)
            
            # Store parameters
            self.parameter_history_['weights'].append(self.weights_.copy())
            self.parameter_history_['means'].append(self.means_.copy())
            self.parameter_history_['covariances'].append(self.covariances_.copy())
            
            # Check convergence
            current_log_likelihood = self._compute_log_likelihood(X)
            self.log_likelihood_history_.append(current_log_likelihood)
            
            if verbose and iteration % 10 == 0:
                print(f"Iteration {iteration}: Log-likelihood = {current_log_likelihood:.4f}")
            
            # Convergence check
            if abs(current_log_likelihood - prev_log_likelihood) < self.tol:
                if verbose:
                    print(f"Converged at iteration {iteration}")
                self.converged_ = True
                break
                
            prev_log_likelihood = current_log_likelihood
            
        self.n_iter_ = iteration + 1
        
        if not self.converged_ and verbose:
            print(f"Did not converge in {self.max_iter} iterations")
        
        return self
    
    def predict_proba(self, X):
        """Predict posterior probabilities for new data"""
        return self._e_step(X)
    
    def predict(self, X):
        """Predict cluster assignments (hard clustering)"""
        responsibilities = self.predict_proba(X)
        return np.argmax(responsibilities, axis=1)
    
    def sample(self, n_samples=100):
        """Generate samples from the learned mixture model"""
        # Choose component for each sample
        component_choices = np.random.choice(
            self.n_components, 
            size=n_samples, 
            p=self.weights_
        )
        
        samples = np.zeros((n_samples, self.means_.shape[1]))
        
        for k in range(self.n_components):
            mask = component_choices == k
            n_k = mask.sum()
            
            if n_k > 0:
                samples[mask] = np.random.multivariate_normal(
                    self.means_[k], 
                    self.covariances_[k], 
                    size=n_k
                )
        
        return samples


def create_challenging_dataset():
    """
    Create a challenging dataset that demonstrates EM capabilities
    - Multiple overlapping clusters
    - Different cluster sizes and shapes
    - Some noise points
    """
    np.random.seed(42)
    
    # Three main clusters with different characteristics
    cluster1 = np.random.multivariate_normal([2, 2], [[1, 0.5], [0.5, 1]], 100)
    cluster2 = np.random.multivariate_normal([6, 6], [[0.5, 0], [0, 2]], 150)  
    cluster3 = np.random.multivariate_normal([3, 7], [[2, -0.8], [-0.8, 1]], 80)
    
    # Add some noise points
    noise = np.random.uniform(-1, 9, (20, 2))
    
    # Combine all data
    X = np.vstack([cluster1, cluster2, cluster3, noise])
    
    # True labels for evaluation (unknown to EM)
    true_labels = np.hstack([
        np.zeros(100), 
        np.ones(150), 
        np.full(80, 2), 
        np.full(20, 3)  # noise cluster
    ])
    
    return X, true_labels


def visualize_em_process(gmm, X, true_labels):
    """
    Comprehensive visualization of the EM learning process
    """
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Data and true clusters
    plt.subplot(3, 4, 1)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=true_labels, alpha=0.6, cmap='tab10')
    plt.title('True Clusters')
    plt.colorbar(scatter)
    
    # 2. Initial vs Final clusters
    plt.subplot(3, 4, 2)
    final_labels = gmm.predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=final_labels, alpha=0.6, cmap='tab10')
    plt.scatter(gmm.means_[:, 0], gmm.means_[:, 1], 
               c='red', marker='x', s=200, linewidths=3)
    plt.title('EM Final Clustering')
    
    # 3. Log-likelihood convergence
    plt.subplot(3, 4, 3)
    plt.plot(gmm.log_likelihood_history_)
    plt.title('Log-Likelihood Convergence')
    plt.xlabel('Iteration')
    plt.ylabel('Log-Likelihood')
    plt.grid(True)
    
    # 4. Parameter evolution visualization
    plt.subplot(3, 4, 4)
    means_history = np.array(gmm.parameter_history_['means'])
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for k in range(gmm.n_components):
        plt.plot(means_history[:, k, 0], means_history[:, k, 1], 
                'o-', color=colors[k % len(colors)], alpha=0.7,
                label=f'Component {k}')
    plt.title('Mean Evolution During EM')
    plt.legend()
    plt.grid(True)
    
    # 5-8. Snapshots at different iterations
    snapshot_iterations = [0, len(gmm.parameter_history_['means'])//4, 
                          len(gmm.parameter_history_['means'])//2, -1]
    snapshot_titles = ['Initial', 'Quarter', 'Halfway', 'Final']
    
    for idx, (iter_idx, title) in enumerate(zip(snapshot_iterations, snapshot_titles)):
        plt.subplot(3, 4, 5 + idx)
        
        # Get parameters at this iteration
        means_iter = gmm.parameter_history_['means'][iter_idx]
        weights_iter = gmm.parameter_history_['weights'][iter_idx]
        covs_iter = gmm.parameter_history_['covariances'][iter_idx]
        
        # Plot data points
        plt.scatter(X[:, 0], X[:, 1], alpha=0.3, c='gray')
        
        # Plot Gaussian contours
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                            np.linspace(y_min, y_max, 50))
        
        for k in range(gmm.n_components):
            # Create temporary GMM for this component
            temp_gmm = GaussianMixtureEM(n_components=1)
            temp_gmm.weights_ = np.array([1.0])
            temp_gmm.means_ = means_iter[k:k+1]
            temp_gmm.covariances_ = covs_iter[k:k+1]
            
            # Compute probability density
            positions = np.dstack((xx, yy)).reshape(-1, 2)
            log_probs = temp_gmm._multivariate_gaussian_log_pdf(
                positions, means_iter[k], covs_iter[k]
            )
            probs = np.exp(log_probs).reshape(xx.shape)
            
            # Plot contour
            plt.contour(xx, yy, probs, levels=3, 
                       colors=[colors[k % len(colors)]], alpha=0.6)
            
            # Plot mean
            plt.scatter(means_iter[k, 0], means_iter[k, 1], 
                       c=colors[k % len(colors)], marker='x', s=100, linewidths=2)
        
        plt.title(f'{title} (Iter {iter_idx if iter_idx >= 0 else gmm.n_iter_-1})')
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
    
    # 9. Responsibility matrix heatmap
    plt.subplot(3, 4, 9)
    responsibilities = gmm.predict_proba(X)
    plt.imshow(responsibilities.T, aspect='auto', cmap='viridis')
    plt.title('Final Responsibilities γ_ik')
    plt.xlabel('Data Point Index')
    plt.ylabel('Component')
    plt.colorbar()
    
    # 10. Component weights evolution
    plt.subplot(3, 4, 10)
    weights_history = np.array(gmm.parameter_history_['weights'])
    for k in range(gmm.n_components):
        plt.plot(weights_history[:, k], 'o-', 
                color=colors[k % len(colors)], label=f'π_{k}')
    plt.title('Mixing Weights Evolution')
    plt.xlabel('Iteration')
    plt.ylabel('Weight')
    plt.legend()
    plt.grid(True)
    
    # 11. Determinant of covariances (measure of cluster spread)
    plt.subplot(3, 4, 11)
    covs_history = np.array(gmm.parameter_history_['covariances'])
    for k in range(gmm.n_components):
        dets = [np.linalg.det(cov) for cov in covs_history[:, k]]
        plt.plot(dets, 'o-', color=colors[k % len(colors)], 
                label=f'|Σ_{k}|')
    plt.title('Covariance Determinants')
    plt.xlabel('Iteration')
    plt.ylabel('Determinant')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    
    # 12. Generated samples from learned model
    plt.subplot(3, 4, 12)
    generated_samples = gmm.sample(n_samples=300)
    plt.scatter(generated_samples[:, 0], generated_samples[:, 1], 
               alpha=0.6, c='purple')
    plt.scatter(gmm.means_[:, 0], gmm.means_[:, 1], 
               c='red', marker='x', s=200, linewidths=3)
    plt.title('Generated Samples')
    
    plt.tight_layout()
    plt.show()


def model_selection_experiment(X):
    """
    Demonstrate model selection: finding optimal number of components
    """
    n_components_range = range(1, 8)
    aic_scores = []
    bic_scores = []
    log_likelihoods = []
    
    print("\\nModel Selection Experiment:")
    print("K\\tLog-Likelihood\\tAIC\\t\\tBIC")
    print("-" * 50)
    
    for k in n_components_range:
        gmm = GaussianMixtureEM(n_components=k, max_iter=100)
        gmm.fit(X, verbose=False)
        
        # Compute information criteria
        n_params = k * (1 + X.shape[1] + X.shape[1]*(X.shape[1]+1)/2) - 1
        
        log_likelihood = gmm.log_likelihood_history_[-1]
        aic = -2 * log_likelihood + 2 * n_params
        bic = -2 * log_likelihood + n_params * np.log(X.shape[0])
        
        log_likelihoods.append(log_likelihood)
        aic_scores.append(aic)
        bic_scores.append(bic)
        
        print(f"{k}\\t{log_likelihood:.2f}\\t\\t{aic:.2f}\\t\\t{bic:.2f}")
    
    # Plot model selection criteria
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(n_components_range, log_likelihoods, 'o-')
    plt.title('Log-Likelihood vs Components')
    plt.xlabel('Number of Components')
    plt.ylabel('Log-Likelihood')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(n_components_range, aic_scores, 'o-', color='red')
    plt.title('AIC vs Components (lower is better)')
    plt.xlabel('Number of Components')
    plt.ylabel('AIC')
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(n_components_range, bic_scores, 'o-', color='green')
    plt.title('BIC vs Components (lower is better)')
    plt.xlabel('Number of Components')
    plt.ylabel('BIC')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Find optimal
    optimal_k_aic = n_components_range[np.argmin(aic_scores)]
    optimal_k_bic = n_components_range[np.argmin(bic_scores)]
    
    print(f"\\nOptimal number of components:")
    print(f"AIC suggests: {optimal_k_aic}")
    print(f"BIC suggests: {optimal_k_bic}")
    
    return optimal_k_aic, optimal_k_bic


def demonstrate_edge_cases():
    """
    Demonstrate how EM handles challenging scenarios
    """
    print("\\n" + "="*60)
    print("EDGE CASE DEMONSTRATIONS")
    print("="*60)
    
    # Edge Case 1: Overlapping clusters
    print("\\n1. Heavily Overlapping Clusters:")
    np.random.seed(123)
    overlap_data = np.vstack([
        np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], 100),
        np.random.multivariate_normal([1, 1], [[1, 0], [0, 1]], 100)
    ])
    
    gmm_overlap = GaussianMixtureEM(n_components=2, max_iter=50)
    gmm_overlap.fit(overlap_data, verbose=False)
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.scatter(overlap_data[:, 0], overlap_data[:, 1], alpha=0.6)
    plt.title('Overlapping Clusters Data')
    
    plt.subplot(1, 3, 2)
    predicted_labels = gmm_overlap.predict(overlap_data)
    plt.scatter(overlap_data[:, 0], overlap_data[:, 1], 
               c=predicted_labels, alpha=0.6, cmap='tab10')
    plt.scatter(gmm_overlap.means_[:, 0], gmm_overlap.means_[:, 1], 
               c='red', marker='x', s=200, linewidths=3)
    plt.title('EM Result')
    
    plt.subplot(1, 3, 3)
    plt.plot(gmm_overlap.log_likelihood_history_)
    plt.title('Convergence')
    plt.xlabel('Iteration')
    plt.ylabel('Log-Likelihood')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Edge Case 2: Unequal cluster sizes
    print("\\n2. Unequal Cluster Sizes:")
    unequal_data = np.vstack([
        np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], 300),  # Large cluster
        np.random.multivariate_normal([5, 5], [[0.5, 0], [0, 0.5]], 50)  # Small cluster
    ])
    
    gmm_unequal = GaussianMixtureEM(n_components=2, max_iter=50)
    gmm_unequal.fit(unequal_data, verbose=False)
    
    print(f"Final mixing weights: {gmm_unequal.weights_}")
    print(f"Expected weights: [300/350, 50/350] = [{300/350:.3f}, {50/350:.3f}]")
    
    
if __name__ == "__main__":
    print("="*60)
    print("EXPECTATION-MAXIMIZATION ALGORITHM DEEP DIVE")
    print("="*60)
    
    # Create challenging dataset
    print("\\nCreating challenging multi-modal dataset...")
    X, true_labels = create_challenging_dataset()
    print(f"Dataset shape: {X.shape}")
    print(f"Number of true clusters: {len(np.unique(true_labels))}")
    
    # Fit EM algorithm
    print("\\nFitting Gaussian Mixture Model with EM...")
    gmm = GaussianMixtureEM(n_components=4, max_iter=100, tol=1e-6)
    gmm.fit(X, verbose=True)
    
    print(f"\\nFinal Results:")
    print(f"Converged: {gmm.converged_}")
    print(f"Iterations: {gmm.n_iter_}")
    print(f"Final log-likelihood: {gmm.log_likelihood_history_[-1]:.4f}")
    print(f"\\nLearned mixing weights π: {gmm.weights_}")
    print(f"\\nLearned means μ:")
    for k, mean in enumerate(gmm.means_):
        print(f"  Component {k}: [{mean[0]:.3f}, {mean[1]:.3f}]")
    
    # Comprehensive visualization
    print("\\nGenerating comprehensive visualization...")
    visualize_em_process(gmm, X, true_labels)
    
    # Model selection
    optimal_k_aic, optimal_k_bic = model_selection_experiment(X)
    
    # Edge cases
    demonstrate_edge_cases()
    
    print("\\n" + "="*60)
    print("KEY INSIGHTS FROM IMPLEMENTATION:")
    print("="*60)
    print("1. E-step computes soft cluster assignments using Bayes' rule")
    print("2. M-step updates parameters using weighted maximum likelihood")
    print("3. Log-likelihood monotonically increases (guaranteed by theory)")
    print("4. Initialization matters - K-means++ provides good starting points")
    print("5. Regularization prevents numerical instabilities")
    print("6. Model selection helps choose optimal number of components")
    print("7. EM finds local optima - multiple runs with different seeds recommended")
    print("="*60)