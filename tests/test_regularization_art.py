"""Numerical constraints for the continuous fitting and regularization study."""

import importlib.util
import unittest


@unittest.skipUnless(
    importlib.util.find_spec("zanim"), "Install requirements-zanim.txt"
)
class RegularizationTests(unittest.TestCase):
    def test_fitting_endpoints_and_capacity_difference(self):
        import numpy as np
        from zanim_scenes import regularization_suite as art
        from zanim_scenes.models import ep_04_4 as model

        low, high = art.fit_values(1)
        np.testing.assert_allclose(low, np.polyval(model.LOW_COEFFS, art.FIT_X))
        np.testing.assert_allclose(high, np.polyval(model.HIGH_COEFFS, art.FIT_X))
        np.testing.assert_allclose(
            np.polyval(model.HIGH_COEFFS, model.TRAIN_X), model.TRAIN_Y, atol=1e-10
        )
        self.assertLess(model.HIGH_TRAIN_LOSS, model.LOW_TRAIN_LOSS)
        self.assertGreater(model.HIGH_VAL_LOSS, model.LOW_VAL_LOSS)

    def test_ridge_stationarity_and_penalized_norm(self):
        import numpy as np
        from zanim_scenes import regularization_suite as art
        from zanim_scenes.models import ep_04_5 as model

        norms = []
        for q in np.linspace(0, 1, 9):
            weights, values = art.ridge_values(float(q))
            penalty = model.displayed_lambda(q)
            regularizer = np.r_[0, weights[1:]]
            gradient = (
                model.DESIGN.T @ (model.DESIGN @ weights - model.TARGETS)
                + penalty * regularizer
            )
            np.testing.assert_allclose(gradient, 0, atol=1e-8)
            np.testing.assert_allclose(
                values, model.polynomial_values(art.RIDGE_X, weights)
            )
            norms.append(model.weight_norm(weights))
        self.assertTrue(np.all(np.diff(norms) <= 1e-8))

    def test_dropout_plateaus_and_classic_inference_scaling(self):
        import numpy as np
        from zanim_scenes.regularization_suite import mask_values
        from zanim_scenes.models import ep_04_6 as model

        for row in range(4):
            for k in range(4):
                mask = mask_values(0.6 + k * 1.25, row)
                np.testing.assert_array_equal(mask, model.MASKS[(k + row) % 4])
                np.testing.assert_array_equal(
                    mask * model.HIDDEN.ravel(), model.H_TRAIN[(k + row) % 4]
                )
            np.testing.assert_allclose(
                mask_values(8.4, row) * model.HIDDEN.ravel(), model.H_INFER
            )
