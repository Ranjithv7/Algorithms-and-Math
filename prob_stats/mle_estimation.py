import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, optimize
from scipy.special import logsumexp
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class MLEEstimator:
    """
    Maximum Likelihood Estimation from first principles
    
    This class implements MLE for various distributions and shows
    the beautiful mathematical theory in action!
    """
    
    def __init__(self, distribution='normal'):
        """
        Initialize MLE estimator for different distributions
        
        Args:
            distribution: 'normal', 'bernoulli', 'exponential', 'gamma'
        """
        self.distribution = distribution
        self.params = None
        self.log_likelihood_history = []
        self.data = None
        
    def fit(self, data, method='closed_form', initial_params=None, verbose=True):
        """
        Fit parameters using Maximum Likelihood Estimation
        
        Args:
            data: observed data points
            method: 'closed_form' or 'numerical' 
            initial_params: starting values for numerical optimization
        """
        self.data = np.array(data)
        n = len(self.data)
        
        if verbose:
            print(f"\\nFitting {self.distribution} distribution to {n} data points...")
            print(f"Data preview: {self.data[:5]}...")
        
        if self.distribution == 'normal':
            if method == 'closed_form':
                # Beautiful closed form solution!
                mu_hat = np.mean(self.data)
                sigma_hat = np.sqrt(np.mean((self.data - mu_hat)**2))
                self.params = {'mu': mu_hat, 'sigma': sigma_hat}
                
                if verbose:
                    print(f"\\nClosed form MLE solution:")
                    print(f"μ̂ = {mu_hat:.4f}")
                    print(f"σ̂ = {sigma_hat:.4f}")
                
            elif method == 'numerical':
                # Show numerical optimization in action
                if initial_params is None:
                    initial_params = [np.mean(self.data), np.std(self.data)]
                
                def negative_log_likelihood(params):
                    mu, sigma = params
                    if sigma <= 0:  # Ensure positive variance
                        return np.inf
                    
                    # Log-likelihood for normal distribution
                    ll = -0.5 * n * np.log(2 * np.pi * sigma**2) - np.sum((self.data - mu)**2) / (2 * sigma**2)
                    self.log_likelihood_history.append(-ll)  # Store for visualization
                    return -ll  # Minimize negative log-likelihood
                
                # Numerical optimization
                result = optimize.minimize(negative_log_likelihood, initial_params, 
                                         method='BFGS')
                
                if result.success:
                    mu_hat, sigma_hat = result.x
                    self.params = {'mu': mu_hat, 'sigma': sigma_hat}
                    
                    if verbose:
                        print(f"\\nNumerical optimization converged in {result.nit} iterations")
                        print(f"μ̂ = {mu_hat:.4f}")
                        print(f"σ̂ = {sigma_hat:.4f}")
                else:
                    print("Optimization failed!")
                    
        elif self.distribution == 'bernoulli':
            # Coin flipping example
            if not np.all(np.isin(self.data, [0, 1])):
                raise ValueError("Bernoulli data must be 0s and 1s")
            
            # Closed form: p̂ = (number of 1s) / total
            p_hat = np.mean(self.data)
            self.params = {'p': p_hat}
            
            if verbose:
                num_ones = np.sum(self.data)
                print(f"\\nBernoulli MLE:")
                print(f"Observed: {num_ones} successes out of {n} trials")
                print(f"p̂ = {num_ones}/{n} = {p_hat:.4f}")
                
        elif self.distribution == 'exponential':
            # Exponential distribution: f(x) = λe^(-λx)
            if np.any(self.data <= 0):
                raise ValueError("Exponential data must be positive")
            
            # Closed form: λ̂ = 1 / sample_mean
            lambda_hat = 1 / np.mean(self.data)
            self.params = {'lambda': lambda_hat}
            
            if verbose:
                print(f"\\nExponential MLE:")
                print(f"Sample mean = {np.mean(self.data):.4f}")
                print(f"λ̂ = 1/mean = {lambda_hat:.4f}")
        
        # Compute final log-likelihood
        final_ll = self.log_likelihood(self.params)
        
        if verbose:
            print(f"\\nFinal log-likelihood: {final_ll:.4f}")
            
        return self
    
    def log_likelihood(self, params=None):
        """
        Compute log-likelihood for given parameters
        This shows the objective function that MLE maximizes!
        """
        if params is None:
            params = self.params
            
        if self.data is None:
            raise ValueError("No data fitted yet")
        
        if self.distribution == 'normal':
            mu, sigma = params['mu'], params['sigma']
            ll = -0.5 * len(self.data) * np.log(2 * np.pi * sigma**2) - np.sum((self.data - mu)**2) / (2 * sigma**2)
            
        elif self.distribution == 'bernoulli':
            p = params['p']
            ll = np.sum(self.data * np.log(p) + (1 - self.data) * np.log(1 - p))
            
        elif self.distribution == 'exponential':
            lam = params['lambda']
            ll = len(self.data) * np.log(lam) - lam * np.sum(self.data)
            
        return ll
    
    def plot_likelihood_surface(self, param_ranges=None):
        """
        Visualize the likelihood surface - see the mountain that MLE climbs!
        """
        if self.distribution == 'normal':
            if param_ranges is None:
                mu_true = self.params['mu']
                sigma_true = self.params['sigma']
                mu_range = np.linspace(mu_true - 2*sigma_true, mu_true + 2*sigma_true, 50)
                sigma_range = np.linspace(sigma_true * 0.1, sigma_true * 2, 50)
            else:
                mu_range, sigma_range = param_ranges
            
            # Create grid
            MU, SIGMA = np.meshgrid(mu_range, sigma_range)
            log_likelihood_grid = np.zeros_like(MU)
            
            # Compute log-likelihood for each parameter combination
            for i in range(len(mu_range)):
                for j in range(len(sigma_range)):
                    params = {'mu': MU[j, i], 'sigma': SIGMA[j, i]}
                    log_likelihood_grid[j, i] = self.log_likelihood(params)
            
            # Plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # 3D surface
            from mpl_toolkits.mplot3d import Axes3D
            ax1 = fig.add_subplot(121, projection='3d')
            surf = ax1.plot_surface(MU, SIGMA, log_likelihood_grid, cmap='viridis', alpha=0.7)
            ax1.scatter([self.params['mu']], [self.params['sigma']], 
                       [self.log_likelihood()], color='red', s=100, label='MLE')
            ax1.set_xlabel('μ')
            ax1.set_ylabel('σ')
            ax1.set_zlabel('Log-Likelihood')
            ax1.set_title('Likelihood Surface (3D)')
            
            # Contour plot
            contour = ax2.contour(MU, SIGMA, log_likelihood_grid, levels=20)
            ax2.clabel(contour, inline=True, fontsize=8)
            ax2.scatter([self.params['mu']], [self.params['sigma']], 
                       color='red', s=100, marker='x', linewidths=3, label='MLE')
            ax2.set_xlabel('μ')
            ax2.set_ylabel('σ')
            ax2.set_title('Likelihood Contours (2D)')
            ax2.legend()
            
            plt.tight_layout()
            plt.show()
            
            print(f"\\nThe red point shows the MLE estimate at the peak of the likelihood!")
            
        elif self.distribution == 'bernoulli':
            # 1D likelihood curve for Bernoulli
            p_range = np.linspace(0.001, 0.999, 1000)
            log_likelihoods = []
            
            for p in p_range:
                ll = self.log_likelihood({'p': p})
                log_likelihoods.append(ll)
            
            plt.figure(figsize=(10, 6))
            plt.plot(p_range, log_likelihoods, 'b-', linewidth=2, label='Log-Likelihood')
            plt.axvline(self.params['p'], color='red', linestyle='--', linewidth=2, label=f'MLE: p̂={self.params["p"]:.3f}')
            plt.xlabel('Parameter p')
            plt.ylabel('Log-Likelihood')
            plt.title('Likelihood Function for Bernoulli Distribution')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()
            
            print(f"\\nThe red line shows the MLE estimate at the peak!")


def solve_coin_flipping_problem():
    """
    Problem 1: Classic Coin Flipping
    
    You flip a suspicious coin 100 times and get this sequence.
    Is the coin fair? What's the probability of heads?
    """
    print("="*60)
    print("PROBLEM 1: COIN FLIPPING ANALYSIS")
    print("="*60)
    
    # Generate realistic coin flip data
    np.random.seed(42)
    true_p = 0.65  # Coin is biased toward heads
    n_flips = 100
    
    # Simulate coin flips
    coin_flips = np.random.binomial(1, true_p, n_flips)
    
    print(f"\\nScenario: You flip a coin {n_flips} times")
    print(f"Results: {np.sum(coin_flips)} heads, {n_flips - np.sum(coin_flips)} tails")
    print(f"Sequence (first 20): {coin_flips[:20]}")
    
    # Fit using MLE
    estimator = MLEEstimator('bernoulli')
    estimator.fit(coin_flips)
    
    # Visualize the likelihood
    estimator.plot_likelihood_surface()
    
    # Statistical significance test
    p_hat = estimator.params['p']
    
    # Test if coin is fair (p = 0.5)
    print(f"\\nHypothesis Test: Is the coin fair?")
    print(f"H₀: p = 0.5 (fair coin)")
    print(f"H₁: p ≠ 0.5 (biased coin)")
    
    # Compute likelihood ratio
    ll_mle = estimator.log_likelihood()
    ll_fair = estimator.log_likelihood({'p': 0.5})
    
    print(f"\\nLog-likelihood at MLE (p̂={p_hat:.3f}): {ll_mle:.4f}")
    print(f"Log-likelihood at p=0.5: {ll_fair:.4f}")
    print(f"Likelihood ratio: {2*(ll_mle - ll_fair):.4f}")
    
    # Critical value for χ² test with 1 df at α=0.05 is 3.84
    if 2*(ll_mle - ll_fair) > 3.84:
        print("✓ Reject H₀: Coin is significantly biased!")
    else:
        print("✗ Fail to reject H₀: Cannot conclude coin is biased")
    
    print(f"\\nTrue probability (unknown in practice): {true_p}")
    print(f"MLE estimate: {p_hat:.4f}")
    print(f"Estimation error: {abs(p_hat - true_p):.4f}")


def solve_height_estimation_problem():
    """
    Problem 2: Population Height Estimation
    
    You sample heights from a population. What's the mean and variance?
    Compare closed-form vs numerical optimization.
    """
    print("\\n" + "="*60)
    print("PROBLEM 2: POPULATION HEIGHT ESTIMATION")
    print("="*60)
    
    # Generate realistic height data
    np.random.seed(123)
    true_mu = 170  # cm
    true_sigma = 8  # cm
    n_samples = 50
    
    heights = np.random.normal(true_mu, true_sigma, n_samples)
    
    print(f"\\nScenario: Measure heights of {n_samples} people")
    print(f"Sample: {heights[:10]} ...")
    print(f"Sample statistics:")
    print(f"  Sample mean: {np.mean(heights):.2f} cm")
    print(f"  Sample std: {np.std(heights):.2f} cm")
    
    # Method 1: Closed form MLE
    print(f"\\n" + "-"*40)
    print("METHOD 1: CLOSED FORM MLE")
    print("-"*40)
    
    estimator1 = MLEEstimator('normal')
    estimator1.fit(heights, method='closed_form')
    
    # Method 2: Numerical optimization
    print(f"\\n" + "-"*40)
    print("METHOD 2: NUMERICAL OPTIMIZATION")
    print("-"*40)
    
    estimator2 = MLEEstimator('normal')
    estimator2.fit(heights, method='numerical', initial_params=[160, 10])
    
    # Compare results
    print(f"\\n" + "-"*40)
    print("COMPARISON OF METHODS")
    print("-"*40)
    
    print(f"True parameters (unknown in practice):")
    print(f"  μ = {true_mu}, σ = {true_sigma}")
    
    print(f"\\nClosed form MLE:")
    print(f"  μ̂ = {estimator1.params['mu']:.4f}")
    print(f"  σ̂ = {estimator1.params['sigma']:.4f}")
    
    print(f"\\nNumerical MLE:")
    print(f"  μ̂ = {estimator2.params['mu']:.4f}")
    print(f"  σ̂ = {estimator2.params['sigma']:.4f}")
    
    print(f"\\nDifference between methods:")
    print(f"  Δμ = {abs(estimator1.params['mu'] - estimator2.params['mu']):.6f}")
    print(f"  Δσ = {abs(estimator1.params['sigma'] - estimator2.params['sigma']):.6f}")
    
    # Visualize likelihood surface
    estimator1.plot_likelihood_surface()
    
    # Show convergence of numerical optimization
    if estimator2.log_likelihood_history:
        plt.figure(figsize=(10, 6))
        plt.plot(estimator2.log_likelihood_history, 'b-', linewidth=2)
        plt.xlabel('Optimization Iteration')
        plt.ylabel('Negative Log-Likelihood')
        plt.title('Convergence of Numerical Optimization')
        plt.grid(True, alpha=0.3)
        plt.show()
        print("\\nNumerical optimization converged to the same solution!")


def solve_logistic_regression_problem():
    """
    Problem 3: Binary Classification with Logistic Regression
    
    Predict if a student passes based on study hours and sleep hours.
    This shows MLE when no closed form exists!
    """
    print("\\n" + "="*60)
    print("PROBLEM 3: LOGISTIC REGRESSION (NO CLOSED FORM)")
    print("="*60)
    
    # Generate synthetic student data
    np.random.seed(42)
    n_students = 200
    
    # Features: [study_hours, sleep_hours]
    study_hours = np.random.uniform(0, 12, n_students)
    sleep_hours = np.random.uniform(4, 10, n_students)
    X = np.column_stack([np.ones(n_students), study_hours, sleep_hours])  # Add bias term
    
    # True parameters (unknown in practice)
    true_beta = np.array([-5, 0.8, 0.3])  # [bias, study_coef, sleep_coef]
    
    # Generate binary outcomes
    logits = X @ true_beta
    probabilities = 1 / (1 + np.exp(-logits))  # Sigmoid function
    outcomes = np.random.binomial(1, probabilities)
    
    print(f"\\nScenario: Predict if {n_students} students pass exam")
    print(f"Features: study hours (0-12), sleep hours (4-10)")
    print(f"Outcome: pass (1) or fail (0)")
    print(f"\\nData preview:")
    for i in range(5):
        print(f"  Student {i+1}: {study_hours[i]:.1f}h study, {sleep_hours[i]:.1f}h sleep → {'Pass' if outcomes[i] else 'Fail'}")
    
    print(f"\\nOverall: {np.sum(outcomes)}/{n_students} students passed ({100*np.mean(outcomes):.1f}%)")
    
    # Implement logistic regression MLE from scratch
    def sigmoid(z):
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))  # Clip for numerical stability
    
    def negative_log_likelihood(beta):
        logits = X @ beta
        probs = sigmoid(logits)
        # Avoid log(0) by adding small epsilon
        eps = 1e-15
        probs = np.clip(probs, eps, 1 - eps)
        
        # Binary cross-entropy (negative log-likelihood)
        nll = -np.sum(outcomes * np.log(probs) + (1 - outcomes) * np.log(1 - probs))
        return nll
    
    # Gradient of negative log-likelihood
    def gradient(beta):
        logits = X @ beta
        probs = sigmoid(logits)
        return -X.T @ (outcomes - probs)
    
    # Numerical optimization
    print(f"\\n" + "-"*40)
    print("NUMERICAL MLE OPTIMIZATION")
    print("-"*40)
    
    # Starting values
    initial_beta = np.random.normal(0, 0.1, 3)
    print(f"Initial parameters: {initial_beta}")
    
    # Optimize using scipy
    result = optimize.minimize(negative_log_likelihood, initial_beta, 
                              jac=gradient, method='BFGS')
    
    if result.success:
        beta_mle = result.x
        print(f"\\nOptimization converged in {result.nit} iterations")
        print(f"Final parameters: {beta_mle}")
        print(f"Final negative log-likelihood: {result.fun:.4f}")
        
        # Compare with true parameters
        print(f"\\nTrue parameters (unknown in practice): {true_beta}")
        print(f"MLE estimates: {beta_mle}")
        print(f"Estimation errors: {np.abs(beta_mle - true_beta)}")
        
        # Make predictions
        test_logits = X @ beta_mle
        test_probs = sigmoid(test_logits)
        predictions = (test_probs > 0.5).astype(int)
        
        accuracy = np.mean(predictions == outcomes)
        print(f"\\nPrediction accuracy: {accuracy:.3f}")
        
        # Interpret coefficients
        print(f"\\nCoefficient interpretation:")
        print(f"  Bias: {beta_mle[0]:.3f}")
        print(f"  Study hours: {beta_mle[1]:.3f} (each hour increases log-odds by {beta_mle[1]:.3f})")
        print(f"  Sleep hours: {beta_mle[2]:.3f} (each hour increases log-odds by {beta_mle[2]:.3f})")
        
        # Visualize results
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Data scatter plot
        colors = ['red' if y == 0 else 'green' for y in outcomes]
        ax1.scatter(study_hours, sleep_hours, c=colors, alpha=0.6)
        ax1.set_xlabel('Study Hours')
        ax1.set_ylabel('Sleep Hours')
        ax1.set_title('Student Data (Red=Fail, Green=Pass)')
        
        # 2. Predicted probabilities
        ax2.scatter(study_hours, sleep_hours, c=test_probs, cmap='RdYlGn', alpha=0.6)
        ax2.set_xlabel('Study Hours')
        ax2.set_ylabel('Sleep Hours')
        ax2.set_title('Predicted Probabilities')
        plt.colorbar(ax2.collections[0], ax=ax2)
        
        # 3. Study hours vs pass probability
        study_sorted = np.argsort(study_hours)
        ax3.plot(study_hours[study_sorted], test_probs[study_sorted], 'b-', alpha=0.7)
        ax3.scatter(study_hours, outcomes, alpha=0.3, c='red')
        ax3.set_xlabel('Study Hours')
        ax3.set_ylabel('Pass Probability')
        ax3.set_title('Study Hours Effect')
        
        # 4. Sleep hours vs pass probability  
        sleep_sorted = np.argsort(sleep_hours)
        ax4.plot(sleep_hours[sleep_sorted], test_probs[sleep_sorted], 'g-', alpha=0.7)
        ax4.scatter(sleep_hours, outcomes, alpha=0.3, c='red')
        ax4.set_xlabel('Sleep Hours')
        ax4.set_ylabel('Pass Probability')
        ax4.set_title('Sleep Hours Effect')
        
        plt.tight_layout()
        plt.show()
        
    else:
        print("Optimization failed!")


def solve_mixture_model_problem():
    """
    Problem 4: When MLE Fails - Multiple Local Maxima
    
    Try to fit a mixture of normals with different initializations.
    Shows why EM algorithm was needed!
    """
    print("\\n" + "="*60)
    print("PROBLEM 4: WHEN MLE FAILS - MIXTURE MODELS")
    print("="*60)
    
    # Generate mixture data
    np.random.seed(42)
    n_samples = 300
    
    # True mixture: 70% from N(2, 1), 30% from N(7, 1.5)
    component = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    data = np.zeros(n_samples)
    
    for i in range(n_samples):
        if component[i] == 0:
            data[i] = np.random.normal(2, 1)
        else:
            data[i] = np.random.normal(7, 1.5)
    
    print(f"\\nGenerated mixture data:")
    print(f"  Component 1: 70% from N(2, 1)")
    print(f"  Component 2: 30% from N(7, 1.5)")
    print(f"  Total samples: {n_samples}")
    
    # Try to fit single normal (will fail to capture bimodality)
    print(f"\\n" + "-"*40)
    print("ATTEMPT 1: FIT SINGLE NORMAL (WRONG MODEL)")
    print("-"*40)
    
    single_normal = MLEEstimator('normal')
    single_normal.fit(data, verbose=False)
    
    print(f"Single normal MLE:")
    print(f"  μ̂ = {single_normal.params['mu']:.3f}")
    print(f"  σ̂ = {single_normal.params['sigma']:.3f}")
    print(f"  Log-likelihood: {single_normal.log_likelihood():.3f}")
    
    # Try to fit mixture model with direct MLE (will struggle with local maxima)
    print(f"\\n" + "-"*40)
    print("ATTEMPT 2: DIRECT MIXTURE MLE (MULTIPLE INITIALIZATIONS)")
    print("-"*40)
    
    def mixture_negative_log_likelihood(params):
        pi1, mu1, sigma1, mu2, sigma2 = params
        pi2 = 1 - pi1
        
        # Ensure valid parameters
        if not (0 <= pi1 <= 1 and sigma1 > 0 and sigma2 > 0):
            return np.inf
        
        # Compute mixture likelihood for each point
        ll = 0
        for x in data:
            prob = pi1 * stats.norm.pdf(x, mu1, sigma1) + pi2 * stats.norm.pdf(x, mu2, sigma2)
            if prob > 0:
                ll += np.log(prob)
            else:
                return np.inf
        
        return -ll
    
    # Try multiple random initializations
    best_ll = -np.inf
    best_params = None
    n_tries = 10
    
    results = []
    
    for trial in range(n_tries):
        # Random initialization
        init_params = [
            np.random.uniform(0.2, 0.8),  # pi1
            np.random.uniform(np.min(data), np.max(data)),  # mu1
            np.random.uniform(0.5, 2.0),  # sigma1
            np.random.uniform(np.min(data), np.max(data)),  # mu2
            np.random.uniform(0.5, 2.0),  # sigma2
        ]
        
        try:
            result = optimize.minimize(mixture_negative_log_likelihood, init_params, 
                                     method='L-BFGS-B',
                                     bounds=[(0.01, 0.99), (None, None), (0.1, None), 
                                            (None, None), (0.1, None)])
            
            if result.success:
                final_ll = -result.fun
                results.append((final_ll, result.x))
                
                if final_ll > best_ll:
                    best_ll = final_ll
                    best_params = result.x
                    
                print(f"Trial {trial+1}: LL={final_ll:.3f}, params={result.x}")
            else:
                print(f"Trial {trial+1}: Failed to converge")
                
        except:
            print(f"Trial {trial+1}: Numerical error")
    
    if best_params is not None:
        pi1, mu1, sigma1, mu2, sigma2 = best_params
        pi2 = 1 - pi1
        
        print(f"\\nBest result from {n_tries} trials:")
        print(f"  Component 1: π={pi1:.3f}, μ={mu1:.3f}, σ={sigma1:.3f}")
        print(f"  Component 2: π={pi2:.3f}, μ={mu2:.3f}, σ={sigma2:.3f}")
        print(f"  Best log-likelihood: {best_ll:.3f}")
        
        print(f"\\nTrue parameters:")
        print(f"  Component 1: π=0.700, μ=2.000, σ=1.000")
        print(f"  Component 2: π=0.300, μ=7.000, σ=1.500")
        
        # Show how different initializations led to different solutions
        if len(results) > 1:
            unique_solutions = []
            for ll, params in results:
                is_new = True
                for ull, uparams in unique_solutions:
                    if abs(ll - ull) < 1e-3:
                        is_new = False
                        break
                if is_new:
                    unique_solutions.append((ll, params))
            
            print(f"\\nFound {len(unique_solutions)} distinct local maxima!")
            for i, (ll, params) in enumerate(unique_solutions):
                print(f"  Solution {i+1}: LL={ll:.3f}")
        
        # Visualize the results
        plt.figure(figsize=(15, 5))
        
        # Plot 1: Data histogram with fitted models
        plt.subplot(1, 3, 1)
        plt.hist(data, bins=30, density=True, alpha=0.7, color='lightblue', edgecolor='black')
        
        x_range = np.linspace(np.min(data), np.max(data), 1000)
        
        # Single normal fit
        single_pdf = stats.norm.pdf(x_range, single_normal.params['mu'], single_normal.params['sigma'])
        plt.plot(x_range, single_pdf, 'r-', linewidth=2, label='Single Normal')
        
        # Mixture fit
        mixture_pdf = (pi1 * stats.norm.pdf(x_range, mu1, sigma1) + 
                      pi2 * stats.norm.pdf(x_range, mu2, sigma2))
        plt.plot(x_range, mixture_pdf, 'g-', linewidth=2, label='Mixture Model')
        
        # True mixture
        true_pdf = (0.7 * stats.norm.pdf(x_range, 2, 1) + 
                   0.3 * stats.norm.pdf(x_range, 7, 1.5))
        plt.plot(x_range, true_pdf, 'k--', linewidth=2, label='True Mixture')
        
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.title('Model Comparison')
        plt.legend()
        
        # Plot 2: Likelihood comparison
        plt.subplot(1, 3, 2)
        models = ['Single\\nNormal', 'Mixture\\nModel']
        likelihoods = [single_normal.log_likelihood(), best_ll]
        bars = plt.bar(models, likelihoods, color=['red', 'green'], alpha=0.7)
        plt.ylabel('Log-Likelihood')
        plt.title('Model Comparison')
        
        # Add values on bars
        for bar, ll in zip(bars, likelihoods):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{ll:.1f}', ha='center', va='bottom')
        
        # Plot 3: Optimization landscape (simplified 2D slice)
        plt.subplot(1, 3, 3)
        if len(results) > 1:
            ll_values = [r[0] for r in results]
            plt.scatter(range(len(results)), ll_values, c=ll_values, 
                       cmap='viridis', s=100, alpha=0.7)
            plt.xlabel('Initialization Trial')
            plt.ylabel('Final Log-Likelihood')
            plt.title('Multiple Local Maxima')
            plt.colorbar()
        
        plt.tight_layout()
        plt.show()
        
        print(f"\\n✓ This shows why EM algorithm was needed for mixture models!")
        print(f"✓ Direct MLE optimization struggles with multiple local maxima")
        print(f"✓ Different initializations find different solutions")
    else:
        print("All optimizations failed!")


def solve_overfitting_demonstration():
    """
    Problem 5: When MLE Overfits
    
    Show how MLE can overfit with polynomial regression.
    """
    print("\\n" + "="*60)
    print("PROBLEM 5: MLE OVERFITTING DEMONSTRATION")
    print("="*60)
    
    # Generate polynomial data with noise
    np.random.seed(42)
    n_points = 15
    
    # True function: quadratic
    x = np.linspace(0, 1, n_points)
    true_y = 2 + 3*x - 4*x**2
    noise = np.random.normal(0, 0.3, n_points)
    y = true_y + noise
    
    print(f"\\nGenerated {n_points} data points from quadratic function with noise")
    print(f"True function: y = 2 + 3x - 4x²")
    print(f"Noise level: σ = 0.3")
    
    # Fit polynomials of different degrees
    degrees = [1, 2, 3, 5, 8, 12]
    results = {}
    
    for degree in degrees:
        # Create polynomial features
        X = np.vander(x, degree + 1, increasing=True)
        
        # MLE for linear regression (normal distribution assumption)
        # This is equivalent to least squares!
        beta_mle = np.linalg.lstsq(X, y, rcond=None)[0]
        
        # Compute log-likelihood (assuming Gaussian errors)
        y_pred = X @ beta_mle
        residuals = y - y_pred
        sigma_mle = np.sqrt(np.mean(residuals**2))
        
        # Log-likelihood for normal distribution
        n = len(y)
        log_likelihood = -0.5 * n * np.log(2 * np.pi * sigma_mle**2) - np.sum(residuals**2) / (2 * sigma_mle**2)
        
        results[degree] = {
            'coefficients': beta_mle,
            'sigma': sigma_mle,
            'log_likelihood': log_likelihood,
            'predictions': y_pred
        }
        
        print(f"\\nDegree {degree}:")
        print(f"  Log-likelihood: {log_likelihood:.3f}")
        print(f"  RMSE: {np.sqrt(np.mean(residuals**2)):.4f}")
        if degree <= 3:
            print(f"  Coefficients: {beta_mle}")
    
    # Visualize results
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Data and fitted curves
    x_fine = np.linspace(0, 1, 200)
    true_y_fine = 2 + 3*x_fine - 4*x_fine**2
    
    ax1.scatter(x, y, c='red', s=50, alpha=0.7, label='Data', zorder=3)
    ax1.plot(x_fine, true_y_fine, 'k--', linewidth=2, label='True function', zorder=2)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(degrees)))
    for degree, color in zip(degrees, colors):
        X_fine = np.vander(x_fine, degree + 1, increasing=True)
        y_fine_pred = X_fine @ results[degree]['coefficients']
        ax1.plot(x_fine, y_fine_pred, color=color, linewidth=2, 
                label=f'Degree {degree}', alpha=0.8, zorder=1)
    
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Polynomial Fits (MLE)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Log-likelihood vs degree
    degrees_list = list(results.keys())
    log_likelihoods = [results[d]['log_likelihood'] for d in degrees_list]
    
    ax2.plot(degrees_list, log_likelihoods, 'bo-', linewidth=2, markersize=8)
    ax2.set_xlabel('Polynomial Degree')
    ax2.set_ylabel('Log-Likelihood')
    ax2.set_title('Log-Likelihood vs Model Complexity')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: RMSE on training data
    rmse_values = [np.sqrt(np.mean((y - results[d]['predictions'])**2)) for d in degrees_list]
    
    ax3.plot(degrees_list, rmse_values, 'ro-', linewidth=2, markersize=8)
    ax3.set_xlabel('Polynomial Degree')
    ax3.set_ylabel('Training RMSE')
    ax3.set_title('Training Error vs Model Complexity')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Focus on high-degree overfitting
    x_zoom = x[5:10]  # Focus on middle region
    y_zoom = y[5:10]
    
    ax4.scatter(x, y, c='red', s=50, alpha=0.7, label='All data')
    ax4.scatter(x_zoom, y_zoom, c='darkred', s=100, label='Zoom region')
    
    # Show highest degree fit
    highest_degree = max(degrees)
    X_fine = np.vander(x_fine, highest_degree + 1, increasing=True)
    y_fine_pred = X_fine @ results[highest_degree]['coefficients']
    ax4.plot(x_fine, y_fine_pred, 'purple', linewidth=3, 
            label=f'Degree {highest_degree} (Overfitted)')
    ax4.plot(x_fine, true_y_fine, 'k--', linewidth=2, label='True function')
    
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.set_title('Overfitting with High-Degree Polynomial')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(-2, 4)  # Limit y-axis to show oscillations
    
    plt.tight_layout()
    plt.show()
    
    # Analysis
    print(f"\\n" + "-"*40)
    print("OVERFITTING ANALYSIS")
    print("-"*40)
    
    print(f"\\nKey observations:")
    print(f"1. Log-likelihood keeps increasing with degree (MLE always prefers more complex models)")
    print(f"2. Training RMSE decreases with degree (perfect fit with enough parameters)")
    print(f"3. High-degree polynomials oscillate wildly between data points")
    print(f"4. MLE doesn't penalize complexity - it only cares about fitting observed data")
    
    print(f"\\nBest log-likelihood: Degree {max(degrees)} with LL = {max(log_likelihoods):.3f}")
    print(f"But this model is clearly overfitted!")
    
    print(f"\\n✓ This demonstrates why regularization (Ridge, Lasso) was invented")
    print(f"✓ Pure MLE doesn't consider generalization")
    print(f"✓ Need to balance fit vs complexity")


if __name__ == "__main__":
    print("="*60)
    print("MAXIMUM LIKELIHOOD ESTIMATION: COMPLETE PROBLEM SOLVING")
    print("="*60)
    
    # Problem 1: Classic coin flipping
    solve_coin_flipping_problem()
    
    # Problem 2: Normal distribution parameter estimation
    solve_height_estimation_problem()
    
    # Problem 3: Logistic regression (no closed form)
    solve_logistic_regression_problem()
    
    # Problem 4: Mixture models (multiple local maxima)
    solve_mixture_model_problem()
    
    # Problem 5: Overfitting demonstration
    solve_overfitting_demonstration()
    
    print("\\n" + "="*60)
    print("KEY INSIGHTS FROM PROBLEM SOLVING:")
    print("="*60)
    print("1. MLE gives intuitive answers for simple problems (coin, normal)")
    print("2. Numerical optimization needed when no closed form exists")
    print("3. Multiple local maxima make optimization challenging")
    print("4. MLE can overfit without regularization")
    print("5. Likelihood surface visualization reveals optimization landscape")
    print("6. Every loss function corresponds to a probabilistic assumption")
    print("7. Cross-entropy = MLE for categorical distributions")
    print("8. MSE = MLE for Gaussian distributions") 
    print("9. Understanding MLE reveals why modern AI methods work")
    print("="*60)