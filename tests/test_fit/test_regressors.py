"""Tests for regression models in the fit module."""

import numpy as np
import pytest


class TestRegressors:
    """Test regression model classes."""

    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic mixture weights and metrics."""
        np.random.seed(42)
        n_samples = 100
        n_sources = 3

        # Generate random mixture weights (sum to 1)
        X = np.random.dirichlet(np.ones(n_sources), n_samples)

        # Generate synthetic metric (linear combination + noise)
        true_coeffs = np.array([0.3, 0.5, 0.2])
        Y = X @ true_coeffs + np.random.normal(0, 0.01, n_samples)

        return X, Y.reshape(-1, 1)

    def test_lightgbm_regressor_fits(self, synthetic_data):
        """Test LightGBM regressor fits without error."""
        from olmix.fit.utils import LightGBMRegressor

        X, Y = synthetic_data
        reg = LightGBMRegressor()
        reg.fit(X, Y, idx=0)

        predictions = reg.predict(X)
        assert predictions.shape == (len(X),)

    def test_lightgbm_predictions_reasonable(self, synthetic_data):
        """Test LightGBM predictions are in reasonable range."""
        from olmix.fit.utils import LightGBMRegressor

        X, Y = synthetic_data
        reg = LightGBMRegressor()
        reg.fit(X, Y, idx=0)

        predictions = reg.predict(X)

        # Predictions should be close to actual values
        mse = np.mean((predictions - Y[:, 0]) ** 2)
        assert mse < 0.1, f"MSE {mse} too high"

    def test_linear_regressor_fits(self, synthetic_data):
        """Test Linear regressor fits without error."""
        from olmix.fit.utils import LinearRegressor

        X, Y = synthetic_data
        reg = LinearRegressor()
        reg.fit(X, Y, idx=0)

        predictions = reg.predict(X)
        assert predictions.shape == (len(X),)

    def test_regressor_base_class(self):
        """Test Regressor base class raises NotImplementedError."""
        from olmix.fit.utils import Regressor

        reg = Regressor()
        with pytest.raises(NotImplementedError):
            reg.fit(None, None, idx=0)

    def test_regressor_predict_without_model(self):
        """Test predict raises error without model."""
        from olmix.fit.utils import Regressor

        reg = Regressor()
        with pytest.raises(AttributeError):
            reg.predict(np.array([[0.5, 0.5]]))


class TestLogLinearRegressor:
    """Test LogLinear regressor specifically."""

    @pytest.fixture
    def positive_data(self):
        """Generate synthetic data with positive values."""
        np.random.seed(42)
        n_samples = 100
        n_sources = 3

        X = np.random.dirichlet(np.ones(n_sources), n_samples)
        # Ensure positive values for log-linear
        Y = np.abs(X @ np.array([0.3, 0.5, 0.2])) + 0.1 + np.abs(np.random.normal(0, 0.01, n_samples))

        return X, Y.reshape(-1, 1)

    def test_log_linear_regressor_fits(self, positive_data):
        """Test Log-linear regressor fits without error."""
        from olmix.fit.utils import LogLinearRegressor

        X, Y = positive_data

        reg = LogLinearRegressor()
        reg.fit(X, Y, idx=0)

        predictions = reg.predict(X)
        assert predictions.shape == (len(X),)
        # Log-linear should predict positive values
        assert all(p > 0 for p in predictions), "Log-linear should predict positive"


class TestModelSerialization:
    """Test model serialization."""

    @pytest.fixture
    def trained_model(self):
        """Return a trained LightGBM model."""
        from olmix.fit.utils import LightGBMRegressor

        np.random.seed(42)
        X = np.random.dirichlet(np.ones(3), 50)
        Y = (X @ np.array([0.3, 0.5, 0.2])).reshape(-1, 1)

        reg = LightGBMRegressor()
        reg.fit(X, Y, idx=0)
        return reg, X

    def test_regressor_serialization(self, trained_model):
        """Test regressor can be serialized and deserialized."""
        import pickle

        reg, X = trained_model

        # Get predictions before serialization
        pred_original = reg.predict(X[:5])

        # Serialize
        serialized = pickle.dumps(reg.model)

        # Deserialize
        loaded = pickle.loads(serialized)

        # Restore model
        reg.model = loaded
        pred_loaded = reg.predict(X[:5])

        np.testing.assert_array_almost_equal(pred_original, pred_loaded)
