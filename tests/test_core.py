import numpy as np
import pytest
from cisd.core import CISD

def test_cisd_init():
    """Test the initialization of the CISD class."""
    # Define simple mock models and selectors
    def scenario_selector(X):
        return X  # Identity function as a simple selector
    
    propensity_model = lambda: None  # Simple mock object
    propensity_model.predict_proba = lambda X: np.array([[0.3, 0.7]] * X.shape[0])
    propensity_model.fit = lambda X, y: None
    
    outcome_model = lambda: None
    outcome_model.predict = lambda X: np.zeros(X.shape[0])
    outcome_model.fit = lambda X, y: None
    
    # Initialize CISD
    cisd = CISD(
        scenario_selector=scenario_selector,
        propensity_model=propensity_model,
        outcome_model=outcome_model,
        random_state=42
    )
    
    # Check attributes
    assert cisd.scenario_selector == scenario_selector
    assert cisd.propensity_model == propensity_model
    assert cisd.outcome_model == outcome_model
    assert cisd.random_state == 42
    assert cisd._is_fitted == False

def test_cisd_fit():
    """Test fitting the CISD model components."""
    # Define simple mock models and selectors
    def scenario_selector(X):
        return X  # Identity function as a simple selector
    
    class MockPropensity:
        def fit(self, X, y):
            self.fitted = True
            return self
        def predict_proba(self, X):
            return np.array([[0.3, 0.7]] * X.shape[0])
    
    class MockOutcome:
        def fit(self, X, y):
            self.fitted = True
            return self
        def predict(self, X):
            return np.zeros(X.shape[0])
    
    propensity_model = MockPropensity()
    outcome_model = MockOutcome()
    
    # Initialize CISD
    cisd = CISD(
        scenario_selector=scenario_selector,
        propensity_model=propensity_model,
        outcome_model=outcome_model
    )
    
    # Generate synthetic data
    n_samples = 100
    n_features = 5
    X = np.random.randn(n_samples, n_features)
    D = np.random.randint(0, 2, n_samples)
    Y = np.random.randn(n_samples)
    
    # Fit CISD
    cisd.fit(X, D, Y)
    
    # Check if the model was fitted
    assert cisd._is_fitted
    assert hasattr(propensity_model, 'fitted')
    assert hasattr(outcome_model, 'fitted')

def test_cisd_estimate_effect():
    """Test estimating causal effect with CISD."""
    # This test would be more complex based on the actual implementation
    # For now, we're checking that the method exists and runs without errors
    
    def scenario_selector(X):
        return X  # Identity function as a simple selector
    
    class MockPropensity:
        def fit(self, X, y):
            return self
        def predict_proba(self, X):
            return np.array([[0.3, 0.7]] * X.shape[0])
    
    class MockOutcome:
        def fit(self, X, y):
            return self
        def predict(self, X):
            return np.zeros(X.shape[0])
    
    propensity_model = MockPropensity()
    outcome_model = MockOutcome()
    
    # Initialize CISD
    cisd = CISD(
        scenario_selector=scenario_selector,
        propensity_model=propensity_model,
        outcome_model=outcome_model
    )
    
    # Generate synthetic data
    n_samples = 100
    n_features = 5
    X = np.random.randn(n_samples, n_features)
    D = np.random.randint(0, 2, n_samples)
    Y = np.random.randn(n_samples)
    
    # Fit CISD
    cisd.fit(X, D, Y)
    
    # Estimate effect
    effect = cisd.estimate_effect(X)
    
    # Check that effect is a scalar or array with reasonable values
    assert effect is not None
