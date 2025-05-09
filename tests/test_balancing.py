import numpy as np
import pytest
from cisd.balancing import Balancer, IPWBalancer

def test_balancer_base_class():
    """Test the base Balancer class."""
    balancer = Balancer()
    
    # Check initial state
    assert balancer._is_fitted == False
    
    # Test fit method
    X = np.random.randn(100, 5)
    D = np.random.randint(0, 2, 100)
    balancer.fit(X, D)
    assert balancer._is_fitted == True

def test_ipw_balancer():
    """Test the IPW (Inverse Probability Weighting) Balancer."""
    # Initialize balancer
    balancer = IPWBalancer(estimator='logistic')
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 100
    n_features = 5
    X = np.random.randn(n_samples, n_features)
    # Create biased treatment: more likely for units with positive first feature
    propensity = 1 / (1 + np.exp(-X[:, 0]))
    D = np.random.binomial(1, propensity)
    
    # Fit balancer
    balancer.fit(X, D)
    
    # Test if balancer is fitted
    assert balancer._is_fitted
    assert hasattr(balancer, 'propensity_model')
    
    # Test transform method
    weights = balancer.transform(X, D)
    
    # Check weights properties
    assert len(weights) == n_samples
    assert np.all(weights > 0)  # All weights should be positive
    
    # Higher weights should be assigned to treated units with low propensity
    # and control units with high propensity
    treated_idx = D == 1
    control_idx = D == 0
    
    # Propensities close to 0 should get high weights if treated
    low_prop_treated = np.logical_and(treated_idx, propensity < 0.3)
    if np.any(low_prop_treated):
        assert np.mean(weights[low_prop_treated]) > np.mean(weights)
    
    # Propensities close to 1 should get high weights if control
    high_prop_control = np.logical_and(control_idx, propensity > 0.7)
    if np.any(high_prop_control):
        assert np.mean(weights[high_prop_control]) > np.mean(weights)

def test_kernel_mmd_balancer():
    """Test the KernelMMDBalancer."""
    # This test would depend on the actual implementation of KernelMMDBalancer
    # We'll add a minimal test to ensure it runs without errors
    
    try:
        from cisd.balancing import KernelMMDBalancer
        
        # Initialize balancer
        balancer = KernelMMDBalancer(kernel='rbf')
        
        # Generate synthetic data
        np.random.seed(42)
        n_samples = 100
        n_features = 5
        X = np.random.randn(n_samples, n_features)
        D = np.random.randint(0, 2, n_samples)
        
        # Fit balancer
        balancer.fit(X, D)
        
        # Test if balancer is fitted
        assert balancer._is_fitted
        
        # Test transform method
        weights = balancer.transform(X, D)
        
        # Check weights properties
        assert len(weights) == n_samples
        assert np.all(weights >= 0)  # All weights should be non-negative
        
    except ImportError:
        # If the class doesn't exist yet, skip this test
        pytest.skip("KernelMMDBalancer not implemented yet")
