"""Numerical regression checks for the native Zanim migration."""

import ast
import importlib.util
from pathlib import Path
import unittest

from scripts.engines import ZANIM_EPISODES, engine_for
from scripts.render import command

AVAILABLE = importlib.util.find_spec("zanim") is not None


class EngineTests(unittest.TestCase):
    def test_migrated_scenes_route_to_native_zanim(self):
        for episode in ZANIM_EPISODES:
            cmd = command(episode, "final", Path("media/test"))
            self.assertIn("scripts.render_zanim", cmd)
            self.assertNotIn("manim", cmd)
            self.assertTrue(cmd[-1].endswith(f"{episode}-final.mp4"))
        self.assertEqual(engine_for("03.1"), "zanim")


@unittest.skipUnless(AVAILABLE, "Install requirements-zanim.txt")
class NativeModelTests(unittest.TestCase):
    def test_transposed_convolution_accumulates_all_four_inputs(self):
        from zanim_scenes.transposed_conv import contributions

        layers = contributions()
        self.assertEqual(layers[-1][-1], [[0, 0, 1], [0, 4, 6], [4, 12, 9]])
        self.assertEqual(sum(map(sum, layers[-1][-1])), 36)

    def test_independent_output_kernels(self):
        from zanim_scenes.multi_channel import outputs

        result = outputs()
        self.assertEqual(result[0][0][0][0][0], 19)
        self.assertEqual(result[0][0][1][0][0], 37)
        self.assertEqual(result[0][1], [[56, 72], [104, 120]])
        self.assertEqual(result[1][1], [[76, 100], [148, 172]])

    def test_optimizer_paths_match_original_numerical_models(self):
        # Execute only the trusted repository's renderer-independent numerical
        # prefix, and compare every path against the migrated standalone module.
        import numpy as np
        from zanim_scenes.optimization import CONFIG, model

        root = Path(__file__).resolve().parents[1]
        for episode, (_, series) in CONFIG.items():
            with self.subTest(episode=episode):
                tree = ast.parse((root / "episodes" / episode / "scene.py").read_text())
                nodes = []
                for node in tree.body:
                    if isinstance(node, ast.ClassDef):
                        break
                    if isinstance(node, ast.ImportFrom) and node.module != "__future__":
                        continue
                    if isinstance(node, ast.Assign) and any(
                        (
                            isinstance(t, ast.Attribute)
                            or isinstance(t, ast.Name)
                            and t.id.endswith(("_COLORS", "_COLOR"))
                        )
                        for t in node.targets
                    ):
                        continue
                    nodes.append(node)
                original = {}
                exec(
                    compile(ast.Module(body=nodes, type_ignores=[]), episode, "exec"),
                    original,
                )
                for path_name, *_ in series:
                    actual = getattr(model(episode), path_name)
                    np.testing.assert_array_equal(actual, original[path_name])
                    self.assertTrue(np.isfinite(actual).all())

    def test_new_models_match_original_arrays(self):
        import numpy as np
        from zanim_scenes import (
            convolution,
            mechanisms,
            attention,
            recurrent,
            foundations,
            vision_language,
        )

        root = Path(__file__).resolve().parents[1]
        for module in (
            convolution,
            mechanisms,
            attention,
            recurrent,
            foundations,
            vision_language,
        ):
            for ep in module.EPISODE_IDS:
                nodes = []
                for node in ast.parse(
                    (root / "episodes" / ep / "scene.py").read_text()
                ).body:
                    if isinstance(node, ast.ClassDef):
                        break
                    if isinstance(node, ast.ImportFrom) and node.module != "__future__":
                        continue
                    if isinstance(node, ast.Assign) and any(
                        (
                            isinstance(t, ast.Attribute)
                            or isinstance(t, ast.Name)
                            and t.id.endswith(("_COLORS", "_COLOR"))
                        )
                        for t in node.targets
                    ):
                        continue
                    nodes.append(node)
                original = {}
                exec(
                    compile(ast.Module(body=nodes, type_ignores=[]), ep, "exec"),
                    original,
                )

                def compare(actual, expected):
                    if isinstance(expected, np.ndarray):
                        np.testing.assert_array_equal(actual, expected)
                    elif isinstance(expected, dict):
                        self.assertEqual(set(actual), set(expected))
                        for key in expected:
                            compare(actual[key], expected[key])
                    elif isinstance(expected, (list, tuple)):
                        self.assertEqual(len(actual), len(expected))
                        for a, b in zip(actual, expected):
                            compare(a, b)
                    else:
                        self.assertEqual(actual, expected)

                for name, value in original.items():
                    if name.isupper() and isinstance(
                        value, (np.ndarray, list, tuple, dict)
                    ):
                        with self.subTest(episode=ep, value=name):
                            compare(getattr(module.model(ep), name), value)

    def test_pooling_and_padding(self):
        import numpy as np
        from zanim_scenes.convolution import model

        p = model("06.5")
        np.testing.assert_array_equal(p.MAX_OUT, [[9, 8], [8, 7]])
        np.testing.assert_array_equal(p.AVG_OUT, [[3, 4], [3, 4]])
        c = model("06.3")
        self.assertEqual(c.Y_P0_S1.shape, (2, 2))
        self.assertEqual(c.Y_P1_S1.shape, (4, 4))
        self.assertEqual(c.Y_P1_S2.shape, (2, 2))
        np.testing.assert_array_equal(c.Y_P1_S2, c.Y_P1_S1[::2, ::2])

    def test_channel_concat_preserves_old_values(self):
        import numpy as np
        from zanim_scenes.convolution import model

        dense = model("07.7")
        np.testing.assert_array_equal(dense.STACK_2[:2], dense.INPUT_X)
        self.assertEqual(dense.STACK_2.shape, (4, 2, 2))
        nin = model("07.3")
        np.testing.assert_allclose(nin.LOGITS, [4, 5, 3])
        inception = model("07.4")
        for index, name in enumerate(("Y1", "Y3", "Y5", "YP")):
            np.testing.assert_array_equal(
                inception.CONCAT[:, :, index], getattr(inception, name)
            )

    def test_batch_norm_and_residual_invariants(self):
        import numpy as np
        from zanim_scenes.mechanisms import model

        norm = model("07.5")
        self.assertAlmostEqual(float(norm.X_HAT.mean()), 0)
        self.assertAlmostEqual(float(norm.X_HAT.var()), 1)
        np.testing.assert_allclose(norm.Y_OUT, 1.5 * norm.X_HAT + 0.4)
        residual = model("07.6")
        np.testing.assert_allclose(
            residual.OUTPUT_Y, np.maximum(0, residual.INPUT_X + residual.RESIDUAL_F)
        )
        self.assertEqual(residual.OUTPUT_Y.shape, residual.INPUT_X.shape)

    def test_convexity_counterexample(self):
        from zanim_scenes.mechanisms import model

        data = model("11.2")
        self.assertLess(data.F_Z, data.F_CHORD)
        self.assertGreater(data.G_Z, data.G_CHORD)
        self.assertAlmostEqual(data.Z, 0)

    def test_registry_matches_available_builders(self):
        from scripts.render_zanim import BUILDERS

        self.assertEqual(set(BUILDERS), set(ZANIM_EPISODES))
        from scripts.episode_info import EPISODES

        self.assertEqual(set(BUILDERS), set(EPISODES))

    def test_horizontal_trajectory_keeps_visible_width(self):
        from zanim_scenes.mesh import ribbon
        from zanim_scenes.style import TEAL

        strip = ribbon([(0, 0, 0), (1, 0, 0), (2, 0, 0)], TEAL, 0.05)
        zs = [v.z for v in strip.mesh.vertices]
        self.assertAlmostEqual(max(zs) - min(zs), 0.1)
        a, b, c = [strip.mesh.vertices[i] for i in strip.mesh.indices[:3]]
        area = abs((b.x - a.x) * (c.z - a.z) - (b.z - a.z) * (c.x - a.x))
        self.assertGreater(area, 0)

    def test_surface_coordinate_contract_for_pinned_release(self):
        from zanim import Surface3D
        from importlib.metadata import version

        self.assertEqual(version("zanim"), "0.7.0rc1")
        surface = Surface3D(lambda x, z: x * x + 2 * z * z, resolution=(3, 3))
        for vertex in surface.mesh.vertices:
            self.assertAlmostEqual(vertex.y, vertex.x**2 + 2 * vertex.z**2)

    def test_attention_normalization_and_weighted_outputs(self):
        import numpy as np
        from zanim_scenes.attention import EPISODE_IDS, stages

        for ep in EPISODE_IDS:
            for stage in stages(ep):
                with self.subTest(episode=ep, stage=stage.name):
                    self.assertTrue((stage.weights >= 0).all())
                    self.assertAlmostEqual(float(stage.weights.sum()), 1)
                    expected = np.exp(stage.scores - np.max(stage.scores))
                    np.testing.assert_allclose(stage.weights, expected / expected.sum())
                    np.testing.assert_allclose(
                        stage.output, np.atleast_1d(stage.weights @ stage.values)
                    )

    def test_gated_states(self):
        import numpy as np
        from zanim_scenes.recurrent import model

        d = model("09.1")
        np.testing.assert_allclose(d.H_NEXT, d.Z * d.H_PREV + (1 - d.Z) * d.H_CANDIDATE)
        d = model("09.2")
        previous = d.CELL_0
        for step in (d.STEP1, d.STEP2):
            for gate in ("F", "I", "O"):
                self.assertTrue(((step[gate] >= 0) & (step[gate] <= 1)).all())
            np.testing.assert_allclose(
                step["C"], step["F"] * previous + step["I"] * step["C_tilde"]
            )
            np.testing.assert_allclose(step["H"], step["O"] * np.tanh(step["C"]))
            previous = step["C"]

    def test_bptt_matches_finite_differences(self):
        import numpy as np
        from zanim_scenes.recurrent import model

        d = model("08.7")
        for t in range(1, len(d.HIDDEN)):
            losses = []
            for delta in (-1e-6, 1e-6):
                h = d.HIDDEN[t].copy() + delta
                for x in d.INPUTS[t:]:
                    h = np.tanh(d.W_XH @ x + d.W_HH @ h + d.BIAS)
                losses.append(float((0.5 * (h - d.TARGET) ** 2).item()))
            self.assertAlmostEqual(
                (losses[1] - losses[0]) / 2e-6, float(d.GRADIENT_H[t - 1, 0]), places=8
            )

    def test_beam_keeps_highest_available_joint_paths(self):
        from zanim_scenes.recurrent import model

        d = model("09.8")
        for prefixes, kept in (
            (d.K1_T1_KEEP, d.K1_T2_KEEP),
            (d.K2_T1_KEEP, d.K2_T2_KEEP),
        ):
            ranked = sorted(
                ((i, j) for i in prefixes for j in range(3)),
                key=lambda pair: -d.JOINT[pair],
            )
            self.assertEqual(kept, ranked[: len(kept)])
        self.assertGreater(d.JOINT[d.K2_T2_KEEP[0]], d.JOINT[d.K1_T2_KEEP[0]])

    def test_linear_training_and_regularization(self):
        import numpy as np
        from zanim_scenes.foundations import model

        d = model("03.1")
        self.assertLess(
            d.mean_squared_loss(*d.TRAINING_STATES[-1]),
            d.mean_squared_loss(*d.TRAINING_STATES[0]),
        )
        for (w, b), (new_w, new_b), idx in zip(
            d.TRAINING_STATES, d.TRAINING_STATES[1:], d.MINIBATCHES
        ):
            err = d.FEATURES[idx] @ w + b - d.TARGETS[idx]
            np.testing.assert_allclose(
                new_w, w - d.LEARNING_RATE * 2 / len(idx) * d.FEATURES[idx].T @ err
            )
            np.testing.assert_allclose(
                new_b, b - d.LEARNING_RATE * 2 * err.mean(axis=0, keepdims=True)
            )
        d = model("04.5")
        self.assertLess(
            d.weight_norm(d.WEIGHTS_DECAYED), d.weight_norm(d.WEIGHTS_UNPENALIZED)
        )
        gradient = d.DESIGN.T @ (d.DESIGN @ d.WEIGHTS_DECAYED - d.TARGETS)
        penalty = d.LAMBDA_FINAL * d.WEIGHTS_DECAYED.copy()
        penalty[0] = 0
        np.testing.assert_allclose(gradient + penalty, 0, atol=1e-8)

    def test_forward_backward_and_dropout(self):
        import numpy as np
        from zanim_scenes.foundations import model

        d = model("04.7")
        np.testing.assert_allclose(
            d.GRAPH["dL_dx"], d.finite_difference_dL_dx(), atol=1e-8
        )
        d = model("04.6")
        np.testing.assert_array_equal(d.H_TRAIN, d.MASKS * d.HIDDEN.ravel())
        np.testing.assert_array_equal(d.H_INFER, d.KEEP_P * d.HIDDEN.ravel())
        d = model("03.4")
        self.assertAlmostEqual(float(d.PROBABILITIES.sum()), 1)
        self.assertEqual(d.MAX_CLASS, int(np.argmax(d.LOGITS)))

    def test_box_roundtrip_and_iou(self):
        import numpy as np
        from zanim_scenes.vision_language import model

        d = model("13.3")
        np.testing.assert_allclose(d.ROUNDTRIP, d.CORNER)
        d = model("13.4")
        np.testing.assert_allclose(d.box_iou(d.ANCHORS, d.ANCHORS).diagonal(), 1)
        np.testing.assert_allclose(d.IOU, [0.5, 0.25])
        d = model("13.5")
        self.assertAlmostEqual(d.FINE_IOU[d.HIT_FINE], 0.64)
        self.assertAlmostEqual(d.COARSE_IOU[d.HIT_COARSE], 0.09)

    def test_embedding_update_improves_context_probability(self):
        import numpy as np
        from zanim_scenes.vision_language import model

        d = model("14.1")
        probabilities = [st["p"][d.CONTEXT] for st in d.STATES]
        self.assertTrue((np.diff(probabilities) > 0).all())
        for st in d.STATES:
            np.testing.assert_allclose(st["p"], d.softmax(st["U"] @ st["v"]))
        d = model("14.7")
        np.testing.assert_allclose(d.COMPOSED - d.KING, d.WOMAN - d.MAN)
        np.testing.assert_allclose(d.RESIDUAL, d.COMPOSED - d.QUEEN)

    def test_rnn_ensemble_sequences_are_independent(self):
        import numpy as np
        from zanim_scenes.rnn_motion import ensemble_states
        from zanim_scenes.models import ep_08_4 as d

        tokens, states = ensemble_states()
        self.assertEqual(states.shape, (48, 4, 2))
        np.testing.assert_allclose(states[0], np.asarray(d.HIDDEN_STATES).reshape(4, 2))
        for i in range(len(tokens)):
            for t in range(3):
                np.testing.assert_allclose(
                    states[i, t + 1],
                    d.rnn_step(tokens[i, t, :, None], states[i, t, :, None]).ravel(),
                )
        self.assertEqual(len(np.unique(states[:, -1], axis=0)), 48)
        np.testing.assert_array_equal(ensemble_states()[1], states)

    def test_rnn_first_input_counterfactual_persists_three_steps(self):
        import numpy as np
        from zanim_scenes.rnn_memory import counterfactual
        from zanim_scenes.models import ep_08_4 as d

        actual, altered = counterfactual()
        np.testing.assert_allclose(actual, np.asarray(d.HIDDEN_STATES).reshape(4, 2))
        np.testing.assert_allclose(altered[0], d.HIDDEN_START.ravel())
        for t, token in enumerate(d.TOKENS):
            used = np.zeros_like(token) if t == 0 else token
            np.testing.assert_allclose(
                altered[t + 1], d.rnn_step(used, altered[t, :, None]).ravel()
            )
        distances = np.linalg.norm(actual[1:] - altered[1:], axis=1)
        np.testing.assert_allclose(distances, [0.75237051, 0.37991938, 0.22508708], atol=1e-8)
        self.assertGreater(distances[-1], 0)

    def test_rnn_flow_grid_matches_model_recurrence(self):
        import numpy as np
        from zanim_scenes.rnn_flow import stages
        from zanim_scenes.models import ep_08_4 as d

        axis = np.linspace(-1, 1, 13)
        points = np.array([(x, y) for y in axis for x in axis])
        for t, token in enumerate(d.TOKENS, 1):
            expected = np.array([d.rnn_step(token, p[:, None]).ravel() for p in points])
            _, _, points = stages(points, token)
            np.testing.assert_allclose(points, expected)
            np.testing.assert_allclose(points[84], d.HIDDEN_STATES[t].ravel(), atol=1e-14)
            self.assertTrue((np.abs(points) < 1).all())

    def test_continuous_paths_and_adjoint_match_rnn(self):
        import numpy as np
        from zanim_scenes.continuous import trajectories, curves, STEPS
        from zanim_scenes.models import ep_08_4 as d

        tokens, h, gradients = trajectories(.7)
        for j in range(STEPS - 1):
            for i in (0, 9, 27):
                np.testing.assert_allclose(h[i,j+1], d.rnn_step(tokens[i,j,:,None], h[i,j,:,None]).ravel())
        direction = np.array([.3, -.8])
        for i in (0,9):
            for t in (0,6,11):
                losses=[]
                for sign in (-1,1):
                    state=h[i,t]+sign*1e-5*direction
                    for token in tokens[i,t:]:
                        state=d.rnn_step(token[:,None],state[:,None]).ravel()
                    losses.append(.5*float(state@state))
                self.assertAlmostEqual((losses[1]-losses[0])/2e-5,
                                       float(gradients[i,t]@direction),places=7)
        projected=curves(h)
        np.testing.assert_allclose(projected[:,::4,1],h[:,:,0]*2.45+h[:,:,1]*.7-.15)

    def test_continuous_gate_branches_share_weighted_endpoint(self):
        import numpy as np
        from zanim_scenes.continuous import trajectories, gate_paths, COUNT
        _,h,_=trajectories(.7)
        for phase in (0.,1.1):
            a,b,z=gate_paths(h,phase)
            self.assertTrue(((z>0)&(z<1)).all())
            lanes=np.arange(COUNT)
            candidate=np.tanh(np.stack((np.sin(lanes*.31+phase)*1.6,
                                       np.cos(lanes*.43-phase)*1.4),axis=-1))
            mixed=z[:,None]*h[:,-1]+(1-z[:,None])*candidate
            np.testing.assert_allclose(a[:,-1],b[:,-1])
            np.testing.assert_allclose(a[:,-1,1],mixed[:,0]*2.25+mixed[:,1]*.55-.15)

    def test_lstm_memory_matches_unrolled_gated_contributions(self):
        import numpy as np
        from zanim_scenes.continuous import memory_states, memory_paths, COUNT, STEPS
        for phase in (0.,.9):
            c,h,f,i,o,g=memory_states(phase)
            for t in range(1,STEPS):
                total=c[:,0]*np.prod(f[:,:t],axis=1)
                for j in range(t):
                    total+=i[:,j]*g[:,j]*np.prod(f[:,j+1:t],axis=1)
                np.testing.assert_allclose(c[:,t],total,atol=1e-14)
            np.testing.assert_allclose(h[:,1:],o*np.tanh(c[:,1:]))
            self.assertTrue((np.abs(h)<1).all())
            upper,lower=memory_paths(phase)
            lane=(np.arange(COUNT)-(COUNT-1)/2)*.066
            np.testing.assert_allclose(upper[:,::4,1],lane[:,None]+.46*c+1.05)
            np.testing.assert_allclose(lower[:,::4,1],lane[:,None]+.78*h-1.95)

    def test_art_gru_gate_extremes_preserve_or_replace_state(self):
        import numpy as np
        from zanim_scenes.gated_art import values
        from zanim_scenes.models import ep_08_4 as d
        keep=values('gru',0.,{'z':1.})
        replace=values('gru',0.,{'z':0.})
        reset=values('gru',0.,{'r':0.})
        np.testing.assert_allclose(keep['mixed'],keep['old'])
        np.testing.assert_allclose(replace['mixed'],replace['candidate'])
        np.testing.assert_allclose(reset['candidate'],np.tanh(reset['x']@d.WEIGHT_XH.T+d.BIAS.ravel()))

    def test_art_lstm_output_gate_does_not_erase_memory(self):
        import numpy as np
        from zanim_scenes.gated_art import values, frame
        baseline=values('lstm',.4)
        closed=values('lstm',.4,{'o':0.})
        kept=values('lstm',.4,{'f':1.,'i':0.})
        erased=values('lstm',.4,{'f':0.,'i':0.})
        np.testing.assert_allclose(closed['cell'],baseline['cell'])
        np.testing.assert_allclose(closed['hidden'],0.)
        np.testing.assert_allclose(kept['cell'],kept['old'])
        np.testing.assert_allclose(erased['cell'],0.)
        for kind in ('gru','lstm'):
            paths,_=frame(kind,.4)
            np.testing.assert_allclose(paths[0][:,-1],paths[1][:,-1])

    def test_stacked_recurrence_is_causal_and_uses_same_time_lower_layer(self):
        import numpy as np
        from zanim_scenes.recurrent_layers import stacked, inputs
        from zanim_scenes.models import ep_08_4 as d
        tokens = inputs(.3)
        _, states = stacked(tokens=tokens)
        changed = tokens.copy()
        changed[:, 10] += .8
        _, perturbed = stacked(tokens=changed)
        np.testing.assert_allclose(states[:, :, :10], perturbed[:, :, :10])
        self.assertGreater(np.linalg.norm(states[:, :, 10] - perturbed[:, :, 10]), .01)
        current = tokens[:, 0]
        for layer in range(3):
            current = np.tanh(current @ d.WEIGHT_XH.T + d.BIAS.ravel())
            np.testing.assert_allclose(states[layer, :, 0], current)

    def test_bidirectional_context_and_concatenation(self):
        import numpy as np
        from zanim_scenes.recurrent_layers import bidirectional, inputs
        tokens = inputs(.3)
        _, forward, backward, joined = bidirectional(tokens=tokens)
        np.testing.assert_array_equal(joined[..., :2], forward)
        np.testing.assert_array_equal(joined[..., 2:], backward)
        future = tokens.copy()
        future[:, 10] += .8
        _, f, b, _ = bidirectional(tokens=future)
        np.testing.assert_array_equal(f[:, :10], forward[:, :10])
        self.assertGreater(np.linalg.norm(b[:, 4] - backward[:, 4]), 1e-8)
        past = tokens.copy()
        past[:, 2] += .8
        _, f, b, _ = bidirectional(tokens=past)
        np.testing.assert_array_equal(b[:, 3:], backward[:, 3:])
        self.assertGreater(np.linalg.norm(f[:, 8] - forward[:, 8]), 1e-8)

    def test_continuous_beam_keeps_top_scored_extensions_only(self):
        import math
        from zanim_scenes.sequence_art import beam_steps
        from zanim_scenes.models import ep_09_8 as d
        steps = beam_steps()
        self.assertEqual([p[0] for p, _ in steps[0][1]], list(d.K2_T1_KEEP))
        self.assertEqual([p for p, _ in steps[1][1]], d.K2_T2_KEEP)
        parents = {()}
        for candidates, kept in steps:
            self.assertEqual(len(candidates), len(parents) * 3)
            self.assertEqual({p[:-1] for p, _ in candidates}, parents)
            self.assertEqual(kept, sorted(candidates, key=lambda row: (-row[1], row[0]))[:2])
            for path, score in candidates:
                expected = math.log(float(d.P_STEP1[path[0]]))
                expected += sum(math.log(float(d.P_STEP2[a,b])) for a,b in zip(path,path[1:]))
                self.assertAlmostEqual(score, expected)
            parents = {p for p, _ in kept}

    def test_continuous_attention_uses_normalized_matching_weights(self):
        import numpy as np
        from zanim_scenes.attention_art import selection
        from zanim_scenes.models import ep_10_1 as d
        for u in np.linspace(0,1,19):
            query, weights, output = selection(u)
            expected = d.softmax_rows((query @ d.KEYS.T)[None,:])[0]
            np.testing.assert_allclose(weights, expected)
            self.assertAlmostEqual(float(weights.sum()), 1.)
            self.assertTrue((weights > 0).all())
            self.assertAlmostEqual(output, float(weights @ d.VALUES[:,0]))
        for j in range(3):
            np.testing.assert_allclose(selection(j/3)[1], d.WEIGHTS[j])
