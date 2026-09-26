"""Numerical regressions against actual scene modules, without rendering."""

import importlib.util
import unittest

from scripts.render import ROOT


def load(episode):
    spec = importlib.util.spec_from_file_location("episode_" + episode.replace(".", "_"),
                                                 ROOT / "episodes" / episode / "scene.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(importlib.util.find_spec("manim"), "Install requirements.txt for scene checks")
class MathTests(unittest.TestCase):
    def test_all_scene_modules_import(self):
        for path in sorted((ROOT / "episodes").glob("*/scene.py")):
            with self.subTest(episode=path.parent.name):
                load(path.parent.name)

    def test_sgd_reduces_loss(self):
        scene = load("03.1")
        states = scene.mini_batch_sgd()
        self.assertLess(scene.mean_squared_loss(*states[-1]), scene.mean_squared_loss(*states[0]))

    def test_softmax_normalized_and_shift_invariant(self):
        import numpy as np
        scene = load("03.4")
        expected = scene.softmax(scene.LOGITS)
        self.assertTrue(np.all(expected >= 0))
        self.assertAlmostEqual(float(expected.sum()), 1)
        np.testing.assert_allclose(scene.softmax(scene.LOGITS + 10000), expected)

    def test_backprop_matches_finite_difference(self):
        import numpy as np
        scene = load("04.7")
        np.testing.assert_allclose(scene.GRAPH["dL_dx"], scene.finite_difference_dL_dx(), rtol=1e-6, atol=1e-8)

    def test_box_coordinates_roundtrip(self):
        import numpy as np
        scene = load("13.3")
        boxes = np.array([[-2., 1., 3., 8.], [0., 0., 0., 0.]])
        np.testing.assert_allclose(scene.box_center_to_corner(scene.box_corner_to_center(boxes)), boxes)

    def test_transposed_convolution_sums_overlap(self):
        import numpy as np
        scene = load("13.10")
        np.testing.assert_array_equal(scene.trans_conv(np.ones((2, 2)), np.ones((2, 2))),
                                      [[1, 2, 1], [2, 4, 2], [1, 2, 1]])


if __name__ == "__main__":
    unittest.main()
