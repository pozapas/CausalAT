import numpy as np
import pytest
from cisd.ai_pipeline import ThreeLayerArchitecture

# Define mock components for testing
class MockRepresentationLearner:
    def __init__(self):
        self._is_fitted = False
    
    def fit(self, X, y=None):
        self._is_fitted = True
        return self
    
    def transform(self, X):
        return np.hstack([X, X])  # Double the dimensions for testing

class MockBalancer:
    def __init__(self):
        self._is_fitted = False
    
    def fit(self, X, D):
        self._is_fitted = True
        return self
    
    def transform(self, X, D):
        # Return simple weights: 2 for treated, 1 for control
        return np.where(D == 1, 2, 1)

class MockCausalLearner:
    def __init__(self):
        self._is_fitted = False
        self.effect_estimate = 2.5  # Fixed effect for testing
    
    def fit(self, Z, D, Y, W=None):
        self._is_fitted = True
        return self
    
    def estimate(self, Z, D=None, Y=None, W=None):
        return self.effect_estimate

def test_three_layer_architecture_init():
    """Test initialization of the ThreeLayerArchitecture."""
    rep_learner = MockRepresentationLearner()
    balancer = MockBalancer()
    causal_learner = MockCausalLearner()
    
    pipeline = ThreeLayerArchitecture(
        representation_learner=rep_learner,
        balancer=balancer,
        causal_learner=causal_learner,
        objective_lambda=0.5
    )
    
    # Check attributes
    assert pipeline.representation_learner == rep_learner
    assert pipeline.balancer == balancer
    assert pipeline.causal_learner == causal_learner
    assert pipeline.objective_lambda == 0.5
    assert pipeline._is_fitted == False

def test_three_layer_architecture_fit():
    """Test fitting the ThreeLayerArchitecture."""
    rep_learner = MockRepresentationLearner()
    balancer = MockBalancer()
    causal_learner = MockCausalLearner()
    
    pipeline = ThreeLayerArchitecture(
        representation_learner=rep_learner,
        balancer=balancer,
        causal_learner=causal_learner
    )
    
    # Generate synthetic data
    n_samples = 100
    n_features = 5
    X = np.random.randn(n_samples, n_features)
    D = np.random.randint(0, 2, n_samples)
    Y = np.random.randn(n_samples)
    
    # Fit pipeline
    pipeline.fit(X, D, Y)
    
    # Check if all components are fitted
    assert pipeline._is_fitted
    assert rep_learner._is_fitted
    assert balancer._is_fitted
    assert causal_learner._is_fitted

def test_three_layer_architecture_estimate_effect():
    """Test estimating effects with the ThreeLayerArchitecture."""
    rep_learner = MockRepresentationLearner()
    balancer = MockBalancer()
    causal_learner = MockCausalLearner()
    
    pipeline = ThreeLayerArchitecture(
        representation_learner=rep_learner,
        balancer=balancer,
        causal_learner=causal_learner
    )
    
    # Generate synthetic data
    n_samples = 100
    n_features = 5
    X = np.random.randn(n_samples, n_features)
    D = np.random.randint(0, 2, n_samples)
    Y = np.random.randn(n_samples)
    
    # Fit pipeline
    pipeline.fit(X, D, Y)
    
    # Estimate effect
    effect = pipeline.estimate_effect(X)
    
    # Check effect equals the mock causal learner's fixed effect
    assert effect == causal_learner.effect_estimate
