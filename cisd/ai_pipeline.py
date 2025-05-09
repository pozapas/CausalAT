"""
End-to-end AI pipelines for causal inference, combining representation learning,
balancing, and causal estimation components.
"""

import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Union
from sklearn.base import BaseEstimator, TransformerMixin

from .representation import RepresentationLearner, MultiModalEncoder
from .balancing import Balancer, IPWBalancer
from .causal import CausalLearner, DoublyRobust

class ThreeLayerArchitecture(BaseEstimator):
    """
    End-to-end three-layer architecture for causal inference with multimodal data.
    
    This implements the Φ → Ψ → Γ architecture described in Section 5.1, combining:
    1. Representation learning (Φ)
    2. Balancing (Ψ)
    3. Causal learning (Γ)
    
    Parameters
    ----------
    representation_learner : RepresentationLearner
        Component that embeds heterogeneous inputs into a latent feature vector.
    balancer : Balancer
        Component that outputs a stabilized weight to equate treated and control distributions.
    causal_learner : CausalLearner
        Component that produces orthogonal scores or influence function corrections.
    objective_lambda : float, default=1.0
        Weight for the balance term in the unified causal loss.
    fit_params : dict, optional
        Additional parameters to pass to the underlying components during fitting.
    """
    
    def __init__(
        self,
        representation_learner: RepresentationLearner,
        balancer: Balancer,
        causal_learner: CausalLearner,
        objective_lambda: float = 1.0,
        fit_params: Optional[Dict] = None
    ):
        self.representation_learner = representation_learner
        self.balancer = balancer
        self.causal_learner = causal_learner
        self.objective_lambda = objective_lambda
        self.fit_params = fit_params or {}
        self._is_fitted = False
    
    def fit(self, X, D, Y, M=None, S=None):
        """
        Fit the three-layer architecture.
        
        Parameters
        ----------
        X : array-like or dict of array-like
            Raw multimodal features (e.g., images, text, etc.).
        D : array-like of shape (n_samples,)
            Treatment indicator (0 or 1).
        Y : array-like of shape (n_samples,)
            Outcome variable.
        M : array-like, optional
            Mediator variables.
        S : array-like, optional
            Scenario variables.
            
        Returns
        -------
        self : object
            Returns self.
        """
        # Convert D and Y to numpy arrays if they're not already
        D = np.asarray(D)
        Y = np.asarray(Y)
        
        # 1. Learn representations (Φ)
        Z = self.representation_learner.fit_transform(X, Y)
        
        # 2. Learn balancing weights (Ψ)
        W = self.balancer.fit_transform(Z, D)
        
        # 3. Learn causal effects (Γ)
        if M is not None and S is not None:
            # If we have mediator and scenario variables, pass them to the causal learner
            self.causal_learner.fit(Z, D, Y, W, M, S)
        else:
            self.causal_learner.fit(Z, D, Y, W)
        
        self._is_fitted = True
        return self
    
    def estimate(self, X, D=None, Y=None, M=None, S=None):
        """
        Estimate the causal effect.
        
        Parameters
        ----------
        X : array-like or dict of array-like
            Raw multimodal features (e.g., images, text, etc.).
        D : array-like, optional
            Treatment indicator (0 or 1).
        Y : array-like, optional
            Outcome variable.
        M : array-like, optional
            Mediator variables.
        S : array-like, optional
            Scenario variables.
            
        Returns
        -------
        effect : dict
            Estimated causal effect and related statistics.
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # 1. Transform raw inputs to latent space
        Z = self.representation_learner.transform(X)
        
        # 2. Compute balancing weights (if D is provided)
        W = None
        if D is not None:
            W = self.balancer.transform(Z, D)
        
        # 3. Estimate causal effect
        if M is not None and S is not None:
            return self.causal_learner.estimate(Z, D, Y, W, M, S)
        else:
            return self.causal_learner.estimate(Z, D, Y, W)
    
    def influence_function(self, X, D, Y, M=None, S=None):
        """
        Compute the efficient influence function.
        
        Parameters
        ----------
        X : array-like or dict of array-like
            Raw multimodal features (e.g., images, text, etc.).
        D : array-like of shape (n_samples,)
            Treatment indicator (0 or 1).
        Y : array-like of shape (n_samples,)
            Outcome variable.
        M : array-like, optional
            Mediator variables.
        S : array-like, optional
            Scenario variables.
            
        Returns
        -------
        infl : array-like of shape (n_samples,)
            Influence function values.
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # 1. Transform raw inputs to latent space
        Z = self.representation_learner.transform(X)
        
        # 2. Compute balancing weights
        W = self.balancer.transform(Z, D)
        
        # 3. Compute influence function
        if M is not None and S is not None:
            return self.causal_learner.influence_function(Z, D, Y, W, M, S)
        else:
            return self.causal_learner.influence_function(Z, D, Y, W)


class ActiveBERTDML(ThreeLayerArchitecture):
    """
    Active-BERT-DML workflow for causal inference with text and image data.
    
    This implements the workflow described in Section 5.5, specialized for
    commute diaries with image and text data.
    
    Parameters
    ----------
    image_encoder : RepresentationLearner, optional
        Encoder for streetview images.
    text_encoder : RepresentationLearner, optional
        Encoder for textual data.
    balancer : Balancer, optional
        Component for balancing treated and control distributions.
    causal_learner : CausalLearner, optional
        Component for causal effect estimation.
    fusion_method : str, default='concatenate'
        Method for fusing image and text embeddings.
    latent_dim : int, default=128
        Dimension of the fused latent representation.
    """
    
    def __init__(
        self,
        image_encoder=None,
        text_encoder=None,
        balancer=None,
        causal_learner=None,
        fusion_method='concatenate',
        latent_dim=128
    ):
        # Import necessary components
        from .representation import StreetviewEncoder, TextEncoder, MultiModalEncoder
        from .balancing import KernelMMD
        from .causal import DoublyRobust
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        
        # Set default components if not provided
        if image_encoder is None:
            image_encoder = StreetviewEncoder(
                pretrained=True,
                embedding_dim=256,
                use_contrastive=True
            )
        
        if text_encoder is None:
            text_encoder = TextEncoder(
                embedding_dim=768,
                model_name='bert-base-uncased',
                use_causal_reg=True
            )
        
        # Create multimodal encoder
        representation_learner = MultiModalEncoder(
            encoders={'image': image_encoder, 'text': text_encoder},
            fusion_method=fusion_method,
            output_dim=latent_dim
        )
        
        # Set default balancer if not provided
        if balancer is None:
            balancer = KernelMMD(
                kernel='rbf',
                kernel_params={'gamma': 1.0/latent_dim},
                lambda_reg=0.01,
                n_iterations=500
            )
        
        # Set default causal learner if not provided
        if causal_learner is None:
            # Create outcome models (one for each treatment level)
            outcome_models = {
                '0': RandomForestRegressor(n_estimators=100, min_samples_leaf=5),
                '1': RandomForestRegressor(n_estimators=100, min_samples_leaf=5)
            }
            
            # Create propensity model
            propensity_model = RandomForestClassifier(n_estimators=100, min_samples_leaf=5)
            
            causal_learner = DoublyRobust(
                propensity_model=propensity_model,
                outcome_models=outcome_models,
                n_splits=5
            )
        
        super().__init__(
            representation_learner=representation_learner,
            balancer=balancer,
            causal_learner=causal_learner,
            objective_lambda=1.0
        )
    
    def fit(self, images, texts, D, Y, M=None, S=None):
        """
        Fit the Active-BERT-DML model with image and text data.
        
        Parameters
        ----------
        images : array-like of shape (n_samples, height, width, channels)
            Streetview images.
        texts : array-like of shape (n_samples,)
            Text data from commute diaries.
        D : array-like of shape (n_samples,)
            Treatment indicator (0 or 1).
        Y : array-like of shape (n_samples,)
            Outcome variable (e.g., eudaimonic scores).
        M : array-like, optional
            Mediator variables.
        S : array-like, optional
            Scenario variables.
            
        Returns
        -------
        self : object
            Returns self.
        """
        # Package the multimodal data into a dictionary
        X = {'image': images, 'text': texts}
        
        # Call the parent class's fit method
        super().fit(X, D, Y, M, S)
        
        return self
    
    def estimate(self, images, texts, D=None, Y=None, M=None, S=None):
        """
        Estimate the causal effect using image and text data.
        
        Parameters
        ----------
        images : array-like of shape (n_samples, height, width, channels)
            Streetview images.
        texts : array-like of shape (n_samples,)
            Text data from commute diaries.
        D : array-like, optional
            Treatment indicator (0 or 1).
        Y : array-like, optional
            Outcome variable (e.g., eudaimonic scores).
        M : array-like, optional
            Mediator variables.
        S : array-like, optional
            Scenario variables.
            
        Returns
        -------
        effect : dict
            Estimated causal effect and related statistics.
        """
        # Package the multimodal data into a dictionary
        X = {'image': images, 'text': texts}
        
        # Call the parent class's estimate method
        return super().estimate(X, D, Y, M, S)
