import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import logsumexp
import time

# Set random seed for reproducibility
np.random.seed(42)

class MixtureDistribution:
    """
    Target distribution: π(x) ∝ 0.4×N(x|-2, 1²) + 0.6×N(x|3, 1.5²)
    """
    def __init__(self):
        self.weights = [0.4, 0.6]
        self.means = [-2, 3]
        self.stds = [1, 1.5]
    
    def log_pdf(self, x):
        """Log probability density (unnormalized)"""
        log_probs = []
        for w, mu, sigma in zip(self.weights, self.means, self.stds):
            log_probs.append(np.log(w) - 0.5 * np.log(2 * np.pi * sigma**2) 
                           - 0.5 * (x - mu)**2 / sigma**2)
        return logsumexp(log_probs)
    
    def pdf(self, x):
        """Probability density (unnormalized)"""
        return np.exp(self.log_pdf(x))
    
    def true_samples(self, n_samples):
        """Generate true samples for comparison"""
        component_samples = np.random.choice(2, size=n_samples, p=self.weights)
        samples = np.zeros(n_samples)
        for i in range(n_samples):
            comp = component_samples[i]
            samples[i] = np.random.normal(self.means[comp], self.stds[comp])
        return samples

# Initialize target distribution
target = MixtureDistribution()

class MetropolisHastings:
    """
    Metropolis-Hastings with random walk proposal
    """
    def __init__(self, target_log_pdf, step_size=1.0):
        self.target_log_pdf = target_log_pdf
        self.step_size = step_size
        self.accepted = 0
        self.total = 0
    
    def sample(self, n_samples, x_init=0):
        """Generate MCMC samples"""
        samples = np.zeros(n_samples)
        x_current = x_init
        
        for i in range(n_samples):
            # Propose new state
            x_proposed = x_current + np.random.normal(0, self.step_size)
            
            # Calculate acceptance probability
            log_alpha = self.target_log_pdf(x_proposed) - self.target_log_pdf(x_current)
            alpha = min(1, np.exp(log_alpha))
            
            # Accept or reject
            if np.random.rand() < alpha:
                x_current = x_proposed
                self.accepted += 1
            
            samples[i] = x_current
            self.total += 1
        
        return samples
    
    def acceptance_rate(self):
        return self.accepted / self.total if self.total > 0 else 0

class GibbsSampler:
    """
    Gibbs sampler treating mixture as latent variable model
    """
    def __init__(self, weights, means, stds):
        self.weights = np.array(weights)
        self.means = np.array(means)
        self.stds = np.array(stds)
        self.n_components = len(weights)
    
    def sample(self, n_samples, x_init=0):
        """Generate Gibbs samples"""
        samples = np.zeros(n_samples)
        components = np.zeros(n_samples, dtype=int)
        
        # Initialize
        x_current = x_init
        z_current = 0  # component assignment
        
        for i in range(n_samples):
            # Update component assignment z given x
            log_probs = np.zeros(self.n_components)
            for k in range(self.n_components):
                log_probs[k] = (np.log(self.weights[k]) 
                              - 0.5 * (x_current - self.means[k])**2 / self.stds[k]**2)
            
            # Normalize and sample
            probs = np.exp(log_probs - logsumexp(log_probs))
            z_current = np.random.choice(self.n_components, p=probs)
            
            # Update x given component assignment z
            x_current = np.random.normal(self.means[z_current], self.stds[z_current])
            
            samples[i] = x_current
            components[i] = z_current
        
        return samples, components

class RejectionSampler:
    """
    Rejection sampler with Gaussian envelope
    """
    def __init__(self, target_pdf, proposal_mean=0, proposal_std=3):
        self.target_pdf = target_pdf
        self.proposal_mean = proposal_mean
        self.proposal_std = proposal_std
        self.accepted = 0
        self.total = 0
        
        # Find envelope constant M (approximately)
        x_test = np.linspace(-8, 8, 1000)
        target_vals = [self.target_pdf(x) for x in x_test]
        proposal_vals = stats.norm.pdf(x_test, proposal_mean, proposal_std)
        self.M = np.max(np.array(target_vals) / np.array(proposal_vals)) * 1.1  # Add safety margin
    
    def sample(self, n_samples):
        """Generate rejection samples"""
        samples = []
        
        while len(samples) < n_samples:
            # Sample from proposal
            x_proposed = np.random.normal(self.proposal_mean, self.proposal_std)
            
            # Calculate acceptance probability
            proposal_density = stats.norm.pdf(x_proposed, self.proposal_mean, self.proposal_std)
            target_density = self.target_pdf(x_proposed)
            acceptance_prob = target_density / (self.M * proposal_density)
            
            self.total += 1
            
            # Accept or reject
            if np.random.rand() < acceptance_prob:
                samples.append(x_proposed)
                self.accepted += 1
        
        return np.array(samples)
    
    def acceptance_rate(self):
        return self.accepted / self.total if self.total > 0 else 0

def compare_algorithms():
    """
    Compare all sampling algorithms on the same problem
    """
    n_samples = 5000
    
    print("🎯 SAMPLING ALGORITHMS COMPARISON")
    print("=" * 50)
    print(f"Target: Mixture of Gaussians")
    print(f"Component 1: 40% weight, N(-2, 1²)")
    print(f"Component 2: 60% weight, N(3, 1.5²)")
    print(f"Samples: {n_samples}")
    print()
    
    # Generate true samples for comparison
    true_samples = target.true_samples(n_samples)
    
    # 1. Metropolis-Hastings
    print("1️⃣ METROPOLIS-HASTINGS")
    print("-" * 25)
    
    mh_sampler = MetropolisHastings(target.log_pdf, step_size=1.5)
    start_time = time.time()
    mh_samples = mh_sampler.sample(n_samples, x_init=0)
    mh_time = time.time() - start_time
    
    print(f"⏱️  Time: {mh_time:.3f} seconds")
    print(f"✅ Acceptance Rate: {mh_sampler.acceptance_rate():.1%}")
    print(f"📊 Sample Mean: {np.mean(mh_samples):.3f}")
    print(f"📊 Sample Std: {np.std(mh_samples):.3f}")
    print()
    
    # 2. Gibbs Sampling
    print("2️⃣ GIBBS SAMPLING")
    print("-" * 18)
    
    gibbs_sampler = GibbsSampler(target.weights, target.means, target.stds)
    start_time = time.time()
    gibbs_samples, gibbs_components = gibbs_sampler.sample(n_samples, x_init=0)
    gibbs_time = time.time() - start_time
    
    print(f"⏱️  Time: {gibbs_time:.3f} seconds")
    print(f"✅ No Rejections: 100% efficiency")
    print(f"📊 Sample Mean: {np.mean(gibbs_samples):.3f}")
    print(f"📊 Sample Std: {np.std(gibbs_samples):.3f}")
    print(f"🎯 Component 1 Frequency: {np.mean(gibbs_components == 0):.1%}")
    print(f"🎯 Component 2 Frequency: {np.mean(gibbs_components == 1):.1%}")
    print()
    
    # 3. Rejection Sampling
    print("3️⃣ REJECTION SAMPLING")
    print("-" * 20)
    
    rejection_sampler = RejectionSampler(target.pdf, proposal_mean=0.5, proposal_std=3)
    start_time = time.time()
    rejection_samples = rejection_sampler.sample(n_samples)
    rejection_time = time.time() - start_time
    
    print(f"⏱️  Time: {rejection_time:.3f} seconds")
    print(f"❌ Acceptance Rate: {rejection_sampler.acceptance_rate():.1%}")
    print(f"📊 Sample Mean: {np.mean(rejection_samples):.3f}")
    print(f"📊 Sample Std: {np.std(rejection_samples):.3f}")
    print(f"💸 Efficiency: {rejection_sampler.acceptance_rate():.1%} (waste: {100-rejection_sampler.acceptance_rate()*100:.1f}%)")
    print()
    
    # 4. True Distribution
    print("4️⃣ TRUE DISTRIBUTION")
    print("-" * 19)
    print(f"📊 True Mean: {np.mean(true_samples):.3f}")
    print(f"📊 True Std: {np.std(true_samples):.3f}")
    print()
    
    # Performance Comparison
    print("⚡ PERFORMANCE COMPARISON")
    print("=" * 30)
    
    methods = ['True', 'Metropolis-Hastings', 'Gibbs', 'Rejection']
    times = [0, mh_time, gibbs_time, rejection_time]
    efficiencies = [100, mh_sampler.acceptance_rate()*100, 100, rejection_sampler.acceptance_rate()*100]
    
    print(f"{'Method':<20} {'Time (s)':<10} {'Efficiency':<12} {'Relative Speed'}")
    print("-" * 60)
    
    for i, (method, t, eff) in enumerate(zip(methods, times, efficiencies)):
        if i == 0:
            print(f"{method:<20} {'N/A':<10} {'100%':<12} {'Baseline'}")
        else:
            rel_speed = times[1] / t if t > 0 else float('inf')
            print(f"{method:<20} {t:<10.3f} {eff:<12.1f}% {rel_speed:<.1f}x")
    
    print()
    
    # Create visualization
    create_comparison_plot(true_samples, mh_samples, gibbs_samples, rejection_samples)
    
    return {
        'true': true_samples,
        'metropolis': mh_samples,
        'gibbs': gibbs_samples,
        'rejection': rejection_samples,
        'times': times[1:],
        'efficiencies': efficiencies[1:]
    }

def create_comparison_plot(true_samples, mh_samples, gibbs_samples, rejection_samples):
    """
    Create comparison visualization
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # True distribution
    x_range = np.linspace(-7, 8, 1000)
    true_density = [target.pdf(x) for x in x_range]
    
    # Plot 1: True samples vs True density
    ax1.hist(true_samples, bins=50, density=True, alpha=0.7, color='green', label='True Samples')
    ax1.plot(x_range, true_density / np.trapz(true_density, x_range), 'g-', linewidth=2, label='True Density')
    ax1.set_title('🎯 True Distribution (Ground Truth)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('x')
    ax1.set_ylabel('Density')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Metropolis-Hastings
    ax2.hist(mh_samples, bins=50, density=True, alpha=0.7, color='blue', label='MH Samples')
    ax2.plot(x_range, true_density / np.trapz(true_density, x_range), 'g--', linewidth=2, label='True Density')
    ax2.set_title('🚶 Metropolis-Hastings', fontsize=14, fontweight='bold')
    ax2.set_xlabel('x')
    ax2.set_ylabel('Density')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Gibbs Sampling
    ax3.hist(gibbs_samples, bins=50, density=True, alpha=0.7, color='red', label='Gibbs Samples')
    ax3.plot(x_range, true_density / np.trapz(true_density, x_range), 'g--', linewidth=2, label='True Density')
    ax3.set_title('🧩 Gibbs Sampling', fontsize=14, fontweight='bold')
    ax3.set_xlabel('x')
    ax3.set_ylabel('Density')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Rejection Sampling
    ax4.hist(rejection_samples, bins=50, density=True, alpha=0.7, color='orange', label='Rejection Samples')
    ax4.plot(x_range, true_density / np.trapz(true_density, x_range), 'g--', linewidth=2, label='True Density')
    ax4.set_title('🎯 Rejection Sampling', fontsize=14, fontweight='bold')
    ax4.set_xlabel('x')
    ax4.set_ylabel('Density')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def analyze_convergence():
    """
    Analyze convergence properties of MCMC methods
    """
    print("\n🔄 CONVERGENCE ANALYSIS")
    print("=" * 25)
    
    # Generate long chains
    n_samples = 10000
    
    # Metropolis-Hastings convergence
    mh_sampler = MetropolisHastings(target.log_pdf, step_size=1.5)
    mh_chain = mh_sampler.sample(n_samples)
    
    # Compute running averages
    running_mean = np.cumsum(mh_chain) / np.arange(1, n_samples + 1)
    true_mean = 0.4 * (-2) + 0.6 * 3  # Theoretical mean
    
    # Plot convergence
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(mh_chain[:1000])
    plt.title('MCMC Chain (first 1000 samples)')
    plt.xlabel('Iteration')
    plt.ylabel('x')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(running_mean)
    plt.axhline(y=true_mean, color='r', linestyle='--', label=f'True Mean = {true_mean:.2f}')
    plt.title('Running Average Convergence')
    plt.xlabel('Iteration')
    plt.ylabel('Running Mean')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"🎯 True Mean: {true_mean:.3f}")
    print(f"📊 Final Sample Mean: {np.mean(mh_chain):.3f}")
    print(f"📏 Absolute Error: {abs(np.mean(mh_chain) - true_mean):.3f}")

def practical_insights():
    """
    Provide practical insights about when to use each method
    """
    print("\n💡 PRACTICAL INSIGHTS")
    print("=" * 25)
    
    insights = {
        "Metropolis-Hastings": {
            "✅ Best for": ["Complex multimodal distributions", "When you only know π(x) up to constant", "High-dimensional problems"],
            "❌ Avoid when": ["Conditional distributions are available", "Need independent samples", "Acceptance rate too low"],
            "🔧 Tuning": ["Adjust step size for 20-50% acceptance", "Use adaptive proposals", "Multiple chains for convergence"]
        },
        
        "Gibbs Sampling": {
            "✅ Best for": ["Hierarchical Bayesian models", "When conditionals are conjugate", "Discrete + continuous variables"],
            "❌ Avoid when": ["High correlation between variables", "Non-conjugate conditionals", "Variables are independent"],
            "🔧 Tuning": ["Random vs systematic scan", "Block updates for correlated variables", "Collapsed sampling when possible"]
        },
        
        "Rejection Sampling": {
            "✅ Best for": ["1D or 2D problems", "Teaching/understanding", "When direct sampling impossible"],
            "❌ Avoid when": ["High dimensions (d > 3)", "Production systems", "Efficiency matters"],
            "🔧 Tuning": ["Find tight envelope M", "Good proposal distribution", "Monitor acceptance rate"]
        }
    }
    
    for method, info in insights.items():
        print(f"\n{method.upper()}")
        print("-" * len(method))
        
        for category, items in info.items():
            print(f"{category}")
            for item in items:
                print(f"  • {item}")

if __name__ == "__main__":
    # Run the complete comparison
    results = compare_algorithms()
    
    # Analyze convergence
    analyze_convergence()
    
    # Print practical insights
    practical_insights()
    
    print("\n🎉 SUMMARY")
    print("=" * 10)
    print("✅ All sampling algorithms implemented and compared!")
    print("📊 Gibbs Sampling was most efficient (no rejections)")
    print("🚶 Metropolis-Hastings was most robust (works everywhere)")
    print("🎯 Rejection Sampling was least efficient but conceptually simple")
    print("\n💭 Key Takeaway: Choose algorithm based on problem structure!")