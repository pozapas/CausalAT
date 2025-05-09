import numpy as np
import pytest
from cisd.causal import CausalLearner, DoublyRobust

def test_causal_learner_base_class():
    """Test the base CausalLearner class."""
    learner = CausalLearner()
    
    # Check initial state
    assert learner._is_fitted == False
    
    # Test fit method
    Z = np.random.randn(100, 5)
    D = np.random.randint(0, 2, 100)
    Y = np.random.randn(100)
    learner.fit(Z, D, Y)
    assert learner._is_fitted == True

def test_doubly_robust():
    """Test the DoublyRobust causal learner."""
    # Initialize doubly robust estimator
    learner = DoublyRobust(
        outcome_model_class='linear',
        propensity_model_class='logistic'
    )
    
    # Generate synthetic data with known effect
    np.random.seed(42)
    n_samples = 1000
    n_features = 5
    
    # Features
    Z = np.random.randn(n_samples, n_features)
    
    # True propensity depends on first feature
    propensity = 1 / (1 + np.exp(-Z[:, 0]))
    D = np.random.binomial(1, propensity)
    
    # Outcome model: Y = 2*Z1 - 3*Z2 + 5*D + noise
    true_effect = 5.0
    Y = 2 * Z[:, 0] - 3 * Z[:, 1] + true_effect * D + np.random.randn(n_samples)
    
    # Fit the model
    learner.fit(Z, D, Y)
    
    # Test if model is fitted
    assert learner._is_fitted
    assert hasattr(learner, 'outcome_models')
    assert hasattr(learner, 'propensity_model')
    
    # Estimate effect
    estimated_effect = learner.estimate(Z)
    
    # Effect should be close to true_effect
    assert np.abs(estimated_effect - true_effect) < 1.0, f"Estimated effect {estimated_effect} should be close to true effect {true_effect}"

def test_influence_functions():
    """Test causal estimator with influence function corrections."""
    # This test would depend on the actual implementation of influence functions
    # We'll add a minimal test to ensure it runs without errors
    
    try:
        from cisd.causal import InfluenceFunctionEstimator
        
        # Initialize estimator
        learner = InfluenceFunctionEstimator()
        
        # Generate synthetic data
        np.random.seed(42)
        n_samples = 100
        n_features = 5
        Z = np.random.randn(n_samples, n_features)
        D = np.random.randint(0, 2, n_samples)
        Y = np.random.randn(n_samples)
        
        # Fit the model
        learner.fit(Z, D, Y)
        
        # Test if model is fitted
        assert learner._is_fitted
        
        # Estimate effect
        effect = learner.estimate(Z)
        
        # Check that effect is a scalar or array with reasonable values
        assert effect is not None
        
    except ImportError:
        # If the class doesn't exist yet, skip this test
        pytest.skip("InfluenceFunctionEstimator not implemented yet")
