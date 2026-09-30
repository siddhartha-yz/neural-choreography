"""Numerical constraints for the linear/softmax/MLP continuous movements."""

import importlib.util
import unittest


@unittest.skipUnless(
    importlib.util.find_spec("zanim"), "Install requirements-zanim.txt"
)
class FoundationTests(unittest.TestCase):
    def test_sgd_knots_follow_the_original_batch_gradients(self):
        import numpy as np
        from zanim_scenes.foundation_art import sgd_parameters
        from zanim_scenes.models import ep_03_1 as model

        for k, batch in enumerate(model.MINIBATCHES):
            w, b, _ = sgd_parameters(k / model.SGD_STEPS)
            xx, yy = model.FEATURES[batch].ravel(), model.TARGETS[batch].ravel()
            residual = w * xx + b - yy
            expected_w = w - model.LEARNING_RATE * 2 * np.mean(xx * residual)
            expected_b = b - model.LEARNING_RATE * 2 * np.mean(residual)
            next_w, next_b, _ = sgd_parameters((k + 1) / model.SGD_STEPS)
            self.assertAlmostEqual(next_w, expected_w)
            self.assertAlmostEqual(next_b, expected_b)
        final_w, final_b, _ = sgd_parameters(1)
        self.assertLess(
            model.mean_squared_loss(np.array([[final_w]]), np.array([[final_b]])),
            model.mean_squared_loss(model.INITIAL_W, model.INITIAL_B),
        )

    def test_softmax_circle_encodes_one_normalized_budget(self):
        import numpy as np
        from zanim_scenes.foundation_art import soft_state
        from zanim_scenes.models import ep_03_4 as model

        for t in np.linspace(0, 8.4, 31):
            x, logits, probabilities, angles = soft_state(float(t))
            np.testing.assert_allclose(logits, model.WEIGHTS @ x + model.BIAS.ravel())
            self.assertTrue(np.all(probabilities > 0))
            self.assertAlmostEqual(float(probabilities.sum()), 1)
            np.testing.assert_allclose(np.diff(angles) / (2 * np.pi), probabilities)
            np.testing.assert_allclose(
                model.softmax(logits + 10000), probabilities, atol=1e-12
            )
        for t in (0, 8.4):
            np.testing.assert_allclose(soft_state(t)[2], model.PROBABILITIES.ravel())
        self.assertGreater(np.linalg.norm(soft_state(3)[2] - soft_state(0)[2]), 0.05)

    def test_relu_lattice_keeps_the_original_sample_and_collapses_negative_regions(
        self,
    ):
        import numpy as np
        from zanim_scenes.foundation_art import INPUTS, GX, GY, lattice_values
        from zanim_scenes.models import ep_04_1 as model

        pre, hidden, outputs = lattice_values()
        j = (GY // 2) * GX + GX // 2
        np.testing.assert_allclose(INPUTS[j], model.INPUT_X.ravel())
        np.testing.assert_allclose(pre[j], model.PREACTIVATIONS.ravel())
        np.testing.assert_allclose(hidden[j], model.HIDDEN.ravel())
        np.testing.assert_allclose(outputs[j], model.OUTPUTS.ravel())
        self.assertTrue(np.all(hidden[pre < 0] == 0))
        np.testing.assert_allclose(hidden[pre > 0], pre[pre > 0])
        collapsed = np.all(pre < 0, axis=1)
        self.assertTrue(collapsed.any())
        np.testing.assert_allclose(
            outputs[collapsed],
            np.broadcast_to(model.BIAS_2.ravel(), outputs[collapsed].shape),
        )
