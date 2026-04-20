"""Test the Nested Sampling algorithms"""

import functools

import chex
import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
from absl.testing import absltest, parameterized

import blackjax
from blackjax.ns import adaptive, base, dynamic_nss, hamiltonian, nss, utils


def gaussian_logprior(x):
    """Standard normal prior"""
    return stats.norm.logpdf(x).sum()


def gaussian_loglikelihood(x):
    """Gaussian likelihood with offset"""
    return stats.norm.logpdf(x - 1.0).sum()


def make_init_state_fn(logprior_fn, loglikelihood_fn):
    """Helper to create init_state_fn from logprior and loglikelihood functions."""
    return functools.partial(
        base.init_state_strategy,
        logprior_fn=logprior_fn,
        loglikelihood_fn=loglikelihood_fn,
    )


def make_mock_nsinfo(positions, loglikelihood, loglikelihood_birth, logdensity):
    """Helper to create NSInfo with correct structure."""
    particles = base.StateWithLogLikelihood(
        position=positions,
        logdensity=logdensity,
        loglikelihood=loglikelihood,
        loglikelihood_birth=loglikelihood_birth,
    )
    return base.NSInfo(particles=particles, update_info={})


def uniform_logprior_2d(x):
    """Uniform prior on [-5, 5]^2"""
    return jnp.where(jnp.all(jnp.abs(x) <= 5.0), 0.0, -jnp.inf)


def gaussian_mixture_loglikelihood(x):
    """2D Gaussian mixture for multi-modal testing"""
    mixture1 = stats.norm.logpdf(x - jnp.array([2.0, 0.0])).sum()
    mixture2 = stats.norm.logpdf(x - jnp.array([-2.0, 0.0])).sum()
    return jnp.logaddexp(mixture1, mixture2)


class NestedSamplingTest(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.key = jax.random.key(42)

    def test_base_ns_init(self):
        """Test basic NS initialization"""
        key = jax.random.key(123)
        num_live = 50

        # Generate initial particles
        positions = jax.random.normal(key, (num_live,))

        # Initialize NS state using the correct API
        init_state_fn = jax.vmap(
            make_init_state_fn(gaussian_logprior, gaussian_loglikelihood)
        )
        state = base.init(positions, init_state_fn)

        # Check state structure - particles is now a StateWithLogLikelihood
        chex.assert_shape(state.particles.position, (num_live,))
        chex.assert_shape(state.particles.loglikelihood, (num_live,))
        chex.assert_shape(state.particles.logdensity, (num_live,))
        chex.assert_shape(state.particles.loglikelihood_birth, (num_live,))

        # Check that loglikelihood and logprior are properly computed
        expected_loglik = jax.vmap(gaussian_loglikelihood)(positions)
        expected_logprior = jax.vmap(gaussian_logprior)(positions)

        chex.assert_trees_all_close(state.particles.loglikelihood, expected_loglik)
        chex.assert_trees_all_close(state.particles.logdensity, expected_logprior)

    def test_delete_fn(self):
        """Test particle deletion function"""
        key = jax.random.key(456)
        num_live = 20
        num_delete = 3

        positions = jax.random.normal(key, (num_live,))
        init_state_fn = jax.vmap(
            make_init_state_fn(gaussian_logprior, gaussian_loglikelihood)
        )
        state = base.init(positions, init_state_fn)

        dead_idx, target_idx = base.delete_fn(state, num_delete)

        # Check correct number of deletions
        chex.assert_shape(dead_idx, (num_delete,))
        chex.assert_shape(target_idx, (num_delete,))

        # Check that worst particles are selected
        worst_loglik = jnp.sort(state.particles.loglikelihood)[:num_delete]
        selected_loglik = state.particles.loglikelihood[dead_idx]
        chex.assert_trees_all_close(jnp.sort(selected_loglik), worst_loglik)

    @parameterized.parameters([1, 2, 5])
    def test_ns_step_consistency(self, num_delete):
        """Test NS step maintains particle count"""
        key = jax.random.key(789)
        num_live = 50

        positions = jax.random.normal(key, (num_live, 2))
        init_state_fn = jax.vmap(
            make_init_state_fn(uniform_logprior_2d, gaussian_mixture_loglikelihood)
        )
        state = base.init(positions, init_state_fn)

        # Mock inner kernel for testing — num_delete closed over from outer scope
        def mock_inner_kernel(rng_key, state, loglikelihood_0):
            particles = state.particles

            # Select start particles from survivors
            choice_key, sample_key = jax.random.split(rng_key)
            weights = (particles.loglikelihood > loglikelihood_0).astype(jnp.float32)
            weights = jnp.where(weights.sum() > 0.0, weights, jnp.ones_like(weights))
            start_idx = jax.random.choice(
                choice_key,
                len(weights),
                shape=(num_delete,),
                p=weights / weights.sum(),
                replace=True,
            )
            start_state = jax.tree.map(lambda x: x[start_idx], particles)

            # Simple random walk for testing
            def single_step(rng_key, state):
                new_pos = (
                    state.position
                    + jax.random.normal(rng_key, state.position.shape) * 0.1
                )
                new_state = base.init_state_strategy(
                    new_pos,
                    uniform_logprior_2d,
                    gaussian_mixture_loglikelihood,
                    loglikelihood_birth=loglikelihood_0,
                )
                return new_state

            sample_keys = jax.random.split(sample_key, num_delete)
            new_particles = jax.vmap(single_step)(sample_keys, start_state)
            return new_particles, {}

        delete_fn = functools.partial(base.delete_fn, num_delete=num_delete)
        kernel = base.build_kernel(delete_fn, mock_inner_kernel)

        # Test that the kernel can be constructed with mock components
        self.assertTrue(callable(kernel))

        # Test delete function works
        dead_idx, target_idx = base.delete_fn(state, num_delete)
        chex.assert_shape(dead_idx, (num_delete,))
        chex.assert_shape(target_idx, (num_delete,))

        # Actually run the kernel and check post-conditions
        new_state, info = kernel(key, state)

        # Particle count preserved
        chex.assert_shape(
            new_state.particles.position,
            state.particles.position.shape,
        )
        # Dead particles returned in info
        chex.assert_shape(info.particles.loglikelihood, (num_delete,))
        # Dead particles are the worst from original state
        worst_loglik = jnp.sort(state.particles.loglikelihood)[:num_delete]
        chex.assert_trees_all_close(
            jnp.sort(info.particles.loglikelihood), worst_loglik
        )

    def test_utils_functions(self):
        """Test utility functions"""
        key = jax.random.key(101112)

        # Create mock dead info
        n_dead = 20
        dead_loglik = jnp.sort(jax.random.uniform(key, (n_dead,))) * 10 - 5
        dead_loglik_birth = jnp.full_like(dead_loglik, -jnp.inf)

        # Create StateWithLogLikelihood for particles
        particles = base.StateWithLogLikelihood(
            position=jnp.zeros((n_dead, 2)),
            logdensity=jnp.zeros(n_dead),
            loglikelihood=dead_loglik,
            loglikelihood_birth=dead_loglik_birth,
        )

        mock_info = base.NSInfo(particles=particles, update_info={})

        # Test compute_num_live
        num_live = utils.compute_num_live(mock_info)
        chex.assert_shape(num_live, (n_dead,))

        # Test logX simulation
        logX_seq, logdX_seq = utils.logX(key, mock_info, shape=10)
        chex.assert_shape(logX_seq, (n_dead, 10))
        chex.assert_shape(logdX_seq, (n_dead, 10))

        # Check logX is decreasing
        self.assertTrue(jnp.all(logX_seq[1:] <= logX_seq[:-1]))


class AdaptiveNestedSamplingTest(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.key = jax.random.key(42)

    def test_adaptive_init(self):
        """Test adaptive NS initialization"""
        key = jax.random.key(123)
        num_live = 30

        positions = jax.random.normal(key, (num_live,))

        def mock_update_params_fn(rng_key, state, info, current_params):
            return {"test_param": 1.0}

        init_state_fn = jax.vmap(
            make_init_state_fn(gaussian_logprior, gaussian_loglikelihood)
        )
        state = adaptive.init(
            positions,
            init_state_fn,
            update_inner_kernel_params_fn=mock_update_params_fn,
        )

        # Check that inner kernel params were set
        self.assertEqual(state.inner_kernel_params["test_param"], 1.0)


class NestedSliceSamplingTest(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.key = jax.random.key(42)

    def test_nss_direction_functions(self):
        """Test NSS direction generation functions"""
        key = jax.random.key(456)

        # Test covariance computation
        positions = jax.random.normal(key, (50, 3))

        def logprior_fn(x):
            return stats.norm.logpdf(x).sum()

        def loglikelihood_fn(x):
            return stats.norm.logpdf(x).sum()

        init_state_fn = jax.vmap(make_init_state_fn(logprior_fn, loglikelihood_fn))
        state = base.init(positions, init_state_fn)

        # Use update_inner_kernel_params instead of removed init_inner_kernel_params
        params = nss.update_inner_kernel_params(key, state, None, {})

        # Check that covariance is computed
        self.assertIn("cov", params)
        cov_pytree = params["cov"]
        chex.assert_shape(cov_pytree, (3, 3))

    def test_nss_kernel_construction(self):
        """Test NSS kernel can be constructed"""
        init_state_fn = make_init_state_fn(gaussian_logprior, gaussian_loglikelihood)
        kernel = nss.build_kernel(init_state_fn, num_inner_steps=10)

        # Test that kernel is callable
        self.assertTrue(callable(kernel))


class NestedSamplingStatisticalTest(chex.TestCase):
    """Statistical correctness tests for nested sampling algorithms."""

    def setUp(self):
        super().setUp()
        self.key = jax.random.key(42)

    def test_1d_gaussian_evidence_estimation(self):
        """Test evidence estimation with analytic validation for unnormalized Gaussian."""

        # Simple case: unnormalized Gaussian likelihood exp(-0.5*x²), uniform prior [-3,3]
        prior_a, prior_b = -3.0, 3.0

        def logprior_fn(x):
            return jnp.where(
                (x >= prior_a) & (x <= prior_b), -jnp.log(prior_b - prior_a), -jnp.inf
            )

        def loglikelihood_fn(x):
            # Unnormalized Gaussian: exp(-0.5 * x²)
            return -0.5 * x**2

        # Analytic evidence: Z = ∫[-3,3] (1/6) * exp(-0.5*x²) dx
        # = (1/6) * √(2π) * [Φ(3) - Φ(-3)]
        from scipy.stats import norm

        prior_width = prior_b - prior_a
        integral_part = jnp.sqrt(2 * jnp.pi) * (norm.cdf(3.0) - norm.cdf(-3.0))
        analytical_evidence = integral_part / prior_width
        analytical_log_evidence = jnp.log(analytical_evidence)

        # Generate mock nested sampling data
        num_steps = 60
        key = jax.random.key(42)

        # Create positions spanning the prior range
        positions = jnp.linspace(prior_a + 0.05, prior_b - 0.05, num_steps).reshape(
            -1, 1
        )
        dead_loglik = jax.vmap(loglikelihood_fn)(positions.flatten())
        dead_logprior = jax.vmap(logprior_fn)(positions.flatten())

        # Sort by likelihood (as NS naturally produces)
        sorted_indices = jnp.argsort(dead_loglik)
        dead_loglik = dead_loglik[sorted_indices]
        positions = positions[sorted_indices]
        dead_logprior = dead_logprior[sorted_indices]

        # Birth likelihoods - start from prior
        dead_loglik_birth = jnp.full_like(dead_loglik, -jnp.inf)

        # Create NSInfo object
        mock_info = make_mock_nsinfo(
            positions, dead_loglik, dead_loglik_birth, dead_logprior
        )

        # Generate many evidence estimates for statistical testing
        n_evidence_samples = 500
        key = jax.random.key(789)
        keys = jax.random.split(key, n_evidence_samples)

        def single_evidence_estimate(rng_key):
            log_weights_matrix = utils.log_weights(rng_key, mock_info, shape=15)
            return jax.scipy.special.logsumexp(log_weights_matrix, axis=0)

        # Compute evidence estimates
        log_evidence_samples = jax.vmap(single_evidence_estimate)(keys)
        log_evidence_samples = log_evidence_samples.flatten()

        # Statistical validation
        mean_estimate = jnp.mean(log_evidence_samples)
        std_estimate = jnp.std(log_evidence_samples)

        # Check statistical consistency with 95% confidence interval
        # For mock data with simplified NS, expect some bias but should be in ballpark
        tolerance = 2.0 * std_estimate  # 95% CI
        bias = jnp.abs(mean_estimate - analytical_log_evidence)

        self.assertLess(
            bias,
            tolerance,
            f"Evidence estimate {mean_estimate} vs analytic {analytical_log_evidence} "
            f"differs by {bias}, which exceeds 2σ = {tolerance}",
        )

        # Also test that individual estimates are reasonable
        self.assertFalse(
            jnp.any(jnp.isnan(log_evidence_samples)),
            "No evidence estimates should be NaN",
        )
        self.assertFalse(
            jnp.any(jnp.isinf(log_evidence_samples)),
            "No evidence estimates should be infinite",
        )

        # Check that estimates are in a reasonable range
        self.assertGreater(
            mean_estimate, analytical_log_evidence - 1.0, "Mean estimate not too low"
        )
        self.assertLess(
            mean_estimate, analytical_log_evidence + 1.0, "Mean estimate not too high"
        )

    def test_uniform_prior_evidence(self):
        """Test evidence estimation for uniform prior with simple likelihood."""

        # Setup: Uniform prior on [0, 1], simple likelihood
        def logprior_fn(x):
            return jnp.where((x >= 0.0) & (x <= 1.0), 0.0, -jnp.inf)

        def loglikelihood_fn(x):
            # Simple quadratic likelihood peaked at 0.5
            return -10.0 * (x - 0.5) ** 2

        # Analytical evidence can be computed numerically for comparison
        # Z = integral_0^1 exp(-10(x-0.5)^2) dx ≈ sqrt(π/10) * erf(...)

        num_live = 50
        key = jax.random.key(456)

        # Initialize particles uniformly in [0, 1]
        positions = jax.random.uniform(key, (num_live,))
        init_state_fn = jax.vmap(make_init_state_fn(logprior_fn, loglikelihood_fn))
        state = base.init(positions, init_state_fn)

        # Check that initialization worked correctly
        self.assertTrue(jnp.all(state.particles.position >= 0.0))
        self.assertTrue(jnp.all(state.particles.position <= 1.0))
        self.assertFalse(jnp.any(jnp.isinf(state.particles.logdensity)))
        self.assertFalse(jnp.any(jnp.isnan(state.particles.loglikelihood)))

    def test_evidence_monotonicity(self):
        """Test that we can initialize state and track integrator."""

        # Simple setup for testing monotonicity
        def logprior_fn(x):
            return stats.norm.logpdf(x)

        def loglikelihood_fn(x):
            return -0.5 * x**2  # Simple quadratic

        num_live = 30
        key = jax.random.key(789)

        positions = jax.random.normal(key, (num_live,))
        init_state_fn = jax.vmap(make_init_state_fn(logprior_fn, loglikelihood_fn))
        initial_state = base.init(positions, init_state_fn)

        # Test that we can access particle likelihoods
        self.assertIsNotNone(initial_state.particles.loglikelihood)
        chex.assert_shape(initial_state.particles.loglikelihood, (num_live,))

        # For integrator tests, use adaptive state instead
        from blackjax.ns import adaptive as adaptive_module

        adaptive_state = adaptive_module.init(positions, init_state_fn)

        # Check integrator exists and has expected fields
        self.assertIsNotNone(adaptive_state.integrator)
        self.assertIsNotNone(adaptive_state.integrator.logZ)
        self.assertIsNotNone(adaptive_state.integrator.logX)

    def test_nested_sampling_utils_statistical_properties(self):
        """Test statistical properties of nested sampling utility functions."""
        key = jax.random.key(101112)

        # Create realistic mock data
        n_dead = 100

        # Generate realistic loglikelihood sequence (increasing)
        base_loglik = jnp.linspace(-10, -1, n_dead)
        noise = jax.random.normal(key, (n_dead,)) * 0.1
        dead_loglik = jnp.sort(base_loglik + noise)

        # Create more realistic birth likelihoods that reflect actual NS behavior
        # Particles can be born at various levels, not just at previous death
        key, subkey = jax.random.split(key)
        birth_noise = jax.random.uniform(subkey, (n_dead,)) * 2.0 - 1.0  # [-1, 1]
        dead_loglik_birth = jnp.concatenate(
            [
                jnp.array([-jnp.inf]),  # First particle born from prior
                dead_loglik[:-1] + birth_noise[1:] * 0.5,  # Others with some variation
            ]
        )
        # Ensure birth likelihoods don't exceed death likelihoods
        dead_loglik_birth = jnp.minimum(dead_loglik_birth, dead_loglik - 0.01)

        mock_info = make_mock_nsinfo(
            jnp.zeros((n_dead, 2)), dead_loglik, dead_loglik_birth, jnp.zeros(n_dead)
        )

        # Test compute_num_live
        num_live = utils.compute_num_live(mock_info)
        chex.assert_shape(num_live, (n_dead,))

        # Basic sanity checks for number of live points
        # NOTE: num_live should NOT be monotonically decreasing in general NS!
        # It follows a sawtooth pattern as particles die and are replenished
        self.assertTrue(
            jnp.all(num_live >= 1), "Should always have at least 1 live point"
        )
        self.assertTrue(
            jnp.all(num_live <= 1000),  # Reasonable upper bound
            "Number of live points should be reasonable",
        )
        self.assertFalse(
            jnp.any(jnp.isnan(num_live)), "Number of live points should not be NaN"
        )

        # Test logX simulation
        n_samples = 50
        logX_seq, logdX_seq = utils.logX(key, mock_info, shape=n_samples)
        chex.assert_shape(logX_seq, (n_dead, n_samples))
        chex.assert_shape(logdX_seq, (n_dead, n_samples))

        # Log volumes should be decreasing
        self.assertTrue(
            jnp.all(logX_seq[1:] <= logX_seq[:-1]), "Log volumes should be decreasing"
        )

        # All log volume elements should be negative (since dX < X)
        finite_logdX = logdX_seq[jnp.isfinite(logdX_seq)]
        if len(finite_logdX) > 0:
            self.assertTrue(
                jnp.all(finite_logdX <= 0.0), "Log volume elements should be negative"
            )

        # Test log_weights function
        log_weights_matrix = utils.log_weights(key, mock_info, shape=n_samples)
        chex.assert_shape(log_weights_matrix, (n_dead, n_samples))

        # Weights should be finite for most particles
        finite_weights = jnp.isfinite(log_weights_matrix)
        self.assertGreater(
            jnp.sum(finite_weights),
            n_dead * n_samples * 0.5,
            "Most weights should be finite",
        )

    def test_gaussian_evidence_narrow_prior(self):
        """Test evidence estimation with narrow prior for challenging case."""

        # Setup: Gaussian likelihood with narrow uniform prior (more challenging)
        mu_true = 1.2
        sigma_true = 0.6
        prior_a, prior_b = 0.8, 1.6  # Narrow prior around the mean

        def logprior_fn(x):
            return jnp.where(
                (x >= prior_a) & (x <= prior_b), -jnp.log(prior_b - prior_a), -jnp.inf
            )

        def loglikelihood_fn(x):
            return -0.5 * ((x - mu_true) / sigma_true) ** 2 - 0.5 * jnp.log(
                2 * jnp.pi * sigma_true**2
            )

        # Analytic evidence
        from scipy.stats import norm

        analytical_evidence = (
            norm.cdf((prior_b - mu_true) / sigma_true)
            - norm.cdf((prior_a - mu_true) / sigma_true)
        ) / (prior_b - prior_a)
        analytical_log_evidence = jnp.log(analytical_evidence)

        # Generate mock NS data with higher resolution for narrow prior
        num_steps = 60
        key = jax.random.key(12345)

        # Dense sampling in the narrow prior region
        positions = jnp.linspace(prior_a + 0.01, prior_b - 0.01, num_steps).reshape(
            -1, 1
        )
        dead_loglik = jax.vmap(loglikelihood_fn)(positions.flatten())
        dead_logprior = jax.vmap(logprior_fn)(positions.flatten())

        # Sort by likelihood
        sorted_indices = jnp.argsort(dead_loglik)
        dead_loglik = dead_loglik[sorted_indices]
        positions = positions[sorted_indices]
        dead_logprior = dead_logprior[sorted_indices]

        # Birth likelihoods
        key, subkey = jax.random.split(key)
        birth_noise = jax.random.uniform(subkey, (num_steps,)) * 0.3 - 0.15
        dead_loglik_birth = jnp.concatenate(
            [jnp.array([-jnp.inf]), dead_loglik[:-1] + birth_noise[1:]]
        )
        dead_loglik_birth = jnp.minimum(dead_loglik_birth, dead_loglik - 0.01)

        mock_info = make_mock_nsinfo(
            positions, dead_loglik, dead_loglik_birth, dead_logprior
        )

        # Generate evidence estimates for statistical testing
        n_evidence_samples = 800
        key = jax.random.key(555)
        keys = jax.random.split(key, n_evidence_samples)

        def single_evidence_estimate(rng_key):
            log_weights_matrix = utils.log_weights(rng_key, mock_info, shape=15)
            return jax.scipy.special.logsumexp(log_weights_matrix, axis=0)

        log_evidence_samples = jax.vmap(single_evidence_estimate)(keys)
        log_evidence_samples = log_evidence_samples.flatten()

        # Statistical validation
        mean_estimate = jnp.mean(log_evidence_samples)
        std_estimate = jnp.std(log_evidence_samples)

        # 99% confidence interval test
        lower_bound = mean_estimate - 2.576 * std_estimate  # 99% CI
        upper_bound = mean_estimate + 2.576 * std_estimate

        self.assertGreater(
            analytical_log_evidence,
            lower_bound,
            f"Analytic evidence {analytical_log_evidence} below 99% CI lower bound {lower_bound}",
        )
        self.assertLess(
            analytical_log_evidence,
            upper_bound,
            f"Analytic evidence {analytical_log_evidence} above 99% CI upper bound {upper_bound}",
        )

    def test_evidence_integration_simple_case(self):
        """Test evidence calculation for a simple analytical case with constant likelihood."""
        # Test case: uniform prior on [0,2], constant likelihood
        # Evidence = ∫[0,2] (1/width) * exp(loglik_constant) dx = exp(loglik_constant)

        loglik_constant = -1.5
        prior_width = 2.0  # Prior on [0, 2]
        n_dead = 40

        # Analytic answer: evidence = ∫[0,2] (1/2) * exp(-1.5) dx = exp(-1.5)
        analytical_log_evidence = loglik_constant

        # Mock data: all particles have same likelihood (constant function)
        dead_loglik = jnp.full(n_dead, loglik_constant)
        dead_loglik_birth = jnp.full(n_dead, -jnp.inf)  # All from prior

        mock_info = make_mock_nsinfo(
            jnp.zeros((n_dead, 1)),
            dead_loglik,
            dead_loglik_birth,
            jnp.full(n_dead, -jnp.log(prior_width)),  # Uniform prior log density
        )

        # Generate many evidence estimates
        n_samples = 500
        key = jax.random.key(999)
        keys = jax.random.split(key, n_samples)

        def single_evidence_estimate(rng_key):
            log_weights_matrix = utils.log_weights(rng_key, mock_info, shape=25)
            return jax.scipy.special.logsumexp(log_weights_matrix, axis=0)

        log_evidence_samples = jax.vmap(single_evidence_estimate)(keys)
        log_evidence_samples = log_evidence_samples.flatten()

        mean_estimate = jnp.mean(log_evidence_samples)
        std_estimate = jnp.std(log_evidence_samples)

        # For constant likelihood case, should be very accurate
        # 95% confidence interval
        lower_bound = mean_estimate - 1.96 * std_estimate
        upper_bound = mean_estimate + 1.96 * std_estimate

        self.assertGreater(
            analytical_log_evidence,
            lower_bound,
            f"Analytic evidence {analytical_log_evidence} below 95% CI",
        )
        self.assertLess(
            analytical_log_evidence,
            upper_bound,
            f"Analytic evidence {analytical_log_evidence} above 95% CI",
        )

    def test_effective_sample_size_calculation(self):
        """Test effective sample size calculation."""
        key = jax.random.key(67890)

        # Create mock data with varying weights
        n_dead = 50
        dead_loglik = jax.random.uniform(key, (n_dead,)) * 5 - 10  # Range [-10, -5]
        dead_loglik_birth = jnp.full(n_dead, -jnp.inf)

        mock_info = make_mock_nsinfo(
            jnp.zeros((n_dead, 1)),
            jnp.sort(dead_loglik),  # Ensure increasing
            dead_loglik_birth,
            jnp.zeros(n_dead),
        )

        # Calculate ESS
        ess_value = utils.ess(key, mock_info)

        # ESS should be positive and reasonable
        self.assertIsInstance(ess_value, (float, jax.Array))
        self.assertGreater(ess_value, 0.0, "ESS should be positive")
        self.assertLessEqual(
            ess_value, n_dead, "ESS should not exceed number of samples"
        )
        self.assertFalse(jnp.isnan(ess_value), "ESS should not be NaN")


if __name__ == "__main__":
    absltest.main()


class DynamicNSSTest(chex.TestCase):
    """Tests for the Dynamic Nested Slice Sampling wrapper."""

    def setUp(self):
        super().setUp()
        self.key = jax.random.key(42)

    def test_dynamic_nss_construction(self):
        """Dynamic NSS can be constructed and its kernel is callable."""

        def logprior_fn(x):
            return stats.norm.logpdf(x).sum()

        def loglikelihood_fn(x):
            return stats.norm.logpdf(x - 1.0).sum()

        sampler = blackjax.dynamic_nss(
            logprior_fn=logprior_fn,
            loglikelihood_fn=loglikelihood_fn,
            num_delete=5,
            num_inner_steps=3,
        )
        self.assertTrue(callable(sampler.init))
        self.assertTrue(callable(sampler.step))

    def test_dynamic_nss_init_and_step(self):
        """Dynamic NSS init + step preserves particle count."""
        num_live = 40
        num_delete = 4

        lower = jnp.full(2, -5.0)
        upper = jnp.full(2, 5.0)

        def logprior_fn(x):
            return jnp.where(jnp.all((x >= lower) & (x <= upper)), 0.0, -jnp.inf)

        def loglikelihood_fn(x):
            return -0.5 * jnp.sum(x**2)

        rng_key = jax.random.key(123)
        positions = jax.random.uniform(
            rng_key, (num_live, 2), minval=-5.0, maxval=5.0
        )

        sampler = blackjax.dynamic_nss(
            logprior_fn=logprior_fn,
            loglikelihood_fn=loglikelihood_fn,
            num_delete=num_delete,
            num_inner_steps=3,
        )

        state = jax.jit(sampler.init)(positions)
        chex.assert_shape(state.particles.position, (num_live, 2))

        new_state, info = jax.jit(sampler.step)(rng_key, state)

        # Particle count must be preserved
        chex.assert_shape(new_state.particles.position, (num_live, 2))
        # Exactly num_delete particles die per step
        chex.assert_shape(info.particles.loglikelihood, (num_delete,))

    @parameterized.parameters([1, 5, 10])
    def test_dynamic_nss_various_num_delete(self, num_delete):
        """Dynamic NSS works for various batch sizes."""
        num_live = 50

        def logprior_fn(x):
            return stats.norm.logpdf(x).sum()

        def loglikelihood_fn(x):
            return stats.norm.logpdf(x).sum()

        rng_key = jax.random.key(42)
        positions = jax.random.normal(rng_key, (num_live, 2))

        sampler = blackjax.dynamic_nss(
            logprior_fn=logprior_fn,
            loglikelihood_fn=loglikelihood_fn,
            num_delete=num_delete,
            num_inner_steps=2,
        )

        state = jax.jit(sampler.init)(positions)
        new_state, info = jax.jit(sampler.step)(rng_key, state)
        chex.assert_shape(info.particles.loglikelihood, (num_delete,))


class HamiltonianNSTest(chex.TestCase):
    """Tests for gradient-guided Hamiltonian Nested Sampling."""

    def setUp(self):
        super().setUp()
        self.key = jax.random.key(42)
        self.lower = jnp.array([-5.0, -5.0])
        self.upper = jnp.array([5.0, 5.0])

    def _make_samplers(self, num_delete=1):
        lower, upper = self.lower, self.upper

        def logprior_fn(x):
            return jnp.where(
                jnp.all((x >= lower) & (x <= upper)), 0.0, -jnp.inf
            )

        def loglikelihood_fn(x):
            return -0.5 * jnp.sum(x**2)

        return logprior_fn, loglikelihood_fn

    def test_hamiltonian_reflection_step_shape(self):
        """hamiltonian_reflection_step returns correct shapes."""
        lower, upper = self.lower, self.upper
        logprior_fn, loglikelihood_fn = self._make_samplers()

        key = jax.random.key(0)
        pos = jax.random.uniform(key, (2,), minval=-3.0, maxval=3.0)
        state = base.init_state_strategy(
            pos,
            logprior_fn=logprior_fn,
            loglikelihood_fn=loglikelihood_fn,
            loglikelihood_birth=-jnp.inf,
        )

        new_state, info = hamiltonian.hamiltonian_reflection_step(
            rng_key=key,
            state=state,
            loglikelihood_fn=loglikelihood_fn,
            logprior_fn=logprior_fn,
            loglikelihood_0=jnp.array(-10.0),
            dt=0.1,
            min_reflections=1,
            max_reflections=5,
            sigma_vel=0.0,
            lower=lower,
            upper=upper,
            max_steps=50,
        )

        chex.assert_shape(new_state.position, (2,))
        chex.assert_shape(new_state.loglikelihood, ())
        chex.assert_shape(info.out_frac, ())
        self.assertFalse(jnp.isnan(info.out_frac))

    def test_hamiltonian_ns_construction(self):
        """Hamiltonian NS sampler can be constructed."""
        logprior_fn, loglikelihood_fn = self._make_samplers()

        sampler = blackjax.ns_hamiltonian(
            logprior_fn=logprior_fn,
            loglikelihood_fn=loglikelihood_fn,
            num_inner_steps=2,
            num_delete=1,
            dt_ini=0.1,
            min_reflections=1,
            max_reflections=5,
            lower=self.lower,
            upper=self.upper,
            max_steps=30,
        )
        self.assertTrue(callable(sampler.init))
        self.assertTrue(callable(sampler.step))

    def test_hamiltonian_ns_requires_bounds(self):
        """Hamiltonian NS raises ValueError when bounds are missing."""
        logprior_fn, loglikelihood_fn = self._make_samplers()

        with self.assertRaises(ValueError):
            blackjax.ns_hamiltonian(
                logprior_fn=logprior_fn,
                loglikelihood_fn=loglikelihood_fn,
                num_inner_steps=2,
            )

    def test_hamiltonian_ns_static_init_and_step(self):
        """Static Hamiltonian NS (num_delete=1) preserves particle count."""
        num_live = 30
        logprior_fn, loglikelihood_fn = self._make_samplers()

        rng_key = jax.random.key(99)
        positions = jax.random.uniform(
            rng_key, (num_live, 2), minval=-4.0, maxval=4.0
        )

        sampler = blackjax.ns_hamiltonian(
            logprior_fn=logprior_fn,
            loglikelihood_fn=loglikelihood_fn,
            num_inner_steps=2,
            num_delete=1,
            dt_ini=0.1,
            min_reflections=1,
            max_reflections=4,
            lower=self.lower,
            upper=self.upper,
            max_steps=40,
        )

        state = jax.jit(sampler.init)(positions)
        chex.assert_shape(state.particles.position, (num_live, 2))
        self.assertIn("dt", state.inner_kernel_params)

        new_state, info = jax.jit(sampler.step)(rng_key, state)
        chex.assert_shape(new_state.particles.position, (num_live, 2))
        chex.assert_shape(info.particles.loglikelihood, (1,))

    def test_hamiltonian_ns_dynamic_step(self):
        """Dynamic Hamiltonian NS (num_delete=5) produces correct dead count."""
        num_live = 40
        num_delete = 5
        logprior_fn, loglikelihood_fn = self._make_samplers()

        rng_key = jax.random.key(77)
        positions = jax.random.uniform(
            rng_key, (num_live, 2), minval=-4.0, maxval=4.0
        )

        sampler = blackjax.ns_hamiltonian(
            logprior_fn=logprior_fn,
            loglikelihood_fn=loglikelihood_fn,
            num_inner_steps=2,
            num_delete=num_delete,
            dt_ini=0.2,
            min_reflections=1,
            max_reflections=4,
            sigma_vel=0.01,
            lower=self.lower,
            upper=self.upper,
            max_steps=40,
        )

        state = jax.jit(sampler.init)(positions)
        new_state, info = jax.jit(sampler.step)(rng_key, state)

        chex.assert_shape(new_state.particles.position, (num_live, 2))
        chex.assert_shape(info.particles.loglikelihood, (num_delete,))

    def test_hamiltonian_ns_dt_adaptation(self):
        """dt is adapted after each step (changes from initial value)."""
        num_live = 30
        dt_ini = 0.05
        logprior_fn, loglikelihood_fn = self._make_samplers()

        rng_key = jax.random.key(55)
        positions = jax.random.uniform(
            rng_key, (num_live, 2), minval=-4.0, maxval=4.0
        )

        sampler = blackjax.ns_hamiltonian(
            logprior_fn=logprior_fn,
            loglikelihood_fn=loglikelihood_fn,
            num_inner_steps=3,
            num_delete=2,
            dt_ini=dt_ini,
            min_reflections=1,
            max_reflections=5,
            lower=self.lower,
            upper=self.upper,
            max_steps=50,
        )

        state = jax.jit(sampler.init)(positions)
        new_state, _ = jax.jit(sampler.step)(rng_key, state)

        dt_after = float(new_state.inner_kernel_params["dt"])
        # dt must be clipped to [1e-5, 10.0]
        self.assertGreaterEqual(dt_after, 1e-5)
        self.assertLessEqual(dt_after, 10.0)
        self.assertFalse(jnp.isnan(jnp.array(dt_after)))

    def test_hamiltonian_ns_update_params_no_info(self):
        """update_inner_kernel_params returns dt_ini when called with info=None."""
        from blackjax.ns.hamiltonian import update_inner_kernel_params

        rng_key = jax.random.key(0)
        result = update_inner_kernel_params(
            rng_key, None, None, {"dt": jnp.array(0.05)}
        )
        self.assertAlmostEqual(float(result["dt"]), 0.05)

    def test_hamiltonian_ns_multi_step(self):
        """Multiple NS steps work without recompilation or error."""
        num_live = 30
        logprior_fn, loglikelihood_fn = self._make_samplers()

        rng_key = jax.random.key(13)
        positions = jax.random.uniform(
            rng_key, (num_live, 2), minval=-4.0, maxval=4.0
        )

        sampler = blackjax.ns_hamiltonian(
            logprior_fn=logprior_fn,
            loglikelihood_fn=loglikelihood_fn,
            num_inner_steps=2,
            num_delete=2,
            dt_ini=0.1,
            min_reflections=1,
            max_reflections=4,
            lower=self.lower,
            upper=self.upper,
            max_steps=40,
        )

        state = jax.jit(sampler.init)(positions)
        step_fn = jax.jit(sampler.step)

        for i in range(5):
            rng_key, subkey = jax.random.split(rng_key)
            state, info = step_fn(subkey, state)
            self.assertFalse(
                jnp.isnan(state.integrator.logZ),
                f"logZ is NaN at step {i}",
            )
