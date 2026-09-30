"""Checks for the new mechanisms and the complete film's visual data."""

import importlib.util
import unittest


@unittest.skipUnless(
    importlib.util.find_spec("zanim"), "Install requirements-zanim.txt"
)
class CompleteFilmTests(unittest.TestCase):
    def test_render_window_keeps_future_opacity_peaks_and_identical_pixels(self):
        from zanim import Scene, Canvas, Circle
        from scripts.zanim_render_window import render_window, verify_window

        scene = Scene(canvas=Canvas(96, 96, 30), fps=20)
        departed = scene.add(Circle(0.3))
        departed.fade_out(duration=1)
        future = scene.add(Circle(0.5))
        future.opacity(to=0, duration=0)
        scene.wait(1)
        future.fade_in(duration=1)
        future.fade_out(duration=1)
        view = render_window(scene, 1.5, 3.5)
        self.assertEqual(len(view._registry), len(scene._registry) - 1)
        self.assertEqual(verify_window(scene, view, 1.5, 3.5), 6)
        self.assertEqual(len(scene._registry), 3)

    def test_common_softmax_division_preserves_ratios_and_total(self):
        import numpy as np
        from zanim_scenes.classification_art import normalization_values

        for time in np.linspace(0, 8.4, 25):
            _, masses, total, scale, probabilities, _ = normalization_values(
                float(time)
            )
            self.assertTrue(np.all(masses > 0))
            self.assertAlmostEqual(float(np.sum(masses)), total)
            np.testing.assert_allclose(
                (masses * scale) / (masses[0] * scale), masses / masses[0]
            )
            if time >= 4:
                np.testing.assert_allclose(masses * scale, probabilities, atol=1e-12)
                self.assertAlmostEqual(float(np.sum(masses * scale)), 1.0)
        self.assertGreater(normalization_values(0)[2], 1.4)

    def test_full_connection_weights_keep_original_sample_and_relu(self):
        import numpy as np
        from zanim_scenes.classification_art import (
            network_values,
            WEIGHT_1,
            WEIGHT_2,
            CALLS,
        )
        from zanim_scenes.models import ep_04_1 as original

        self.assertEqual(WEIGHT_1.shape, (4, 2))
        self.assertEqual(WEIGHT_2.shape, (2, 4))
        self.assertTrue(np.all(WEIGHT_1 != 0))
        self.assertTrue(np.all(WEIGHT_2 != 0))
        pre, h, out = network_values(original.INPUT_X.ravel())
        np.testing.assert_allclose(out, original.OUTPUTS.ravel())
        np.testing.assert_allclose(h[:2], original.HIDDEN.ravel())
        self.assertTrue(np.all(h[2:] == 0))
        for x in CALLS:
            pre, h, _ = network_values(x)
            self.assertTrue(np.all(h[pre < 0] == 0))
            np.testing.assert_allclose(h[pre >= 0], pre[pre >= 0])

    def test_remaining_movements_complete_the_catalog_once(self):
        from scripts.engines import ZANIM_EPISODES
        from zanim_scenes.finale_art import MOVEMENTS

        prefix = "03.1 03.4 04.1 04.4 04.5 04.6 04.7 04.8 05.1 06.2 06.3 06.4 06.5 06.6 07.2 07.3 07.4 07.5 07.6 07.7 08.4 08.7 09.1 09.2 09.3 09.4 09.6 09.8 10.1 10.2".split()
        codes = prefix + [movement[0] for movement in MOVEMENTS]
        self.assertEqual(len(codes), 49)
        self.assertEqual(len(set(codes)), 49)
        self.assertEqual(set(codes), set(ZANIM_EPISODES))

    def test_geometry_stays_finite_and_inside_the_stage(self):
        import numpy as np
        from zanim_scenes.finale_art import MOVEMENTS, picture

        for code, _, _, duration in MOVEMENTS:
            for t in np.linspace(0, duration, 9):
                dd, ll = picture(code, float(t), duration)
                self.assertTrue(all(np.isfinite(a).all() for a in dd + ll), code)
                pp = np.concatenate(
                    [a[:, :2] for a in dd] + [a[:, :4].reshape(-1, 2) for a in ll]
                )
                self.assertLess(float(np.abs(pp[:, 0]).max()), 8.8, code)
                self.assertLess(float(np.abs(pp[:, 1]).max()), 4.4, code)

    def test_transposed_contributions_sum_to_the_direct_scatter(self):
        import numpy as np
        from zanim_scenes.transposed_conv import X, K, contributions

        expected = np.zeros((3, 3))
        for r in range(2):
            for c in range(2):
                expected[r : r + 2, c : c + 2] += X[r][c] * np.asarray(K)
        np.testing.assert_array_equal(contributions()[-1][-1], expected)
        self.assertEqual(expected[1, 1], 4)
