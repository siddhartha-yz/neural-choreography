import ast
import json
import subprocess
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from scripts.render import ROOT, PROFILES, command, scene_for
from scripts.qa_video import frame_times, inspect_metadata, review
from scripts.review_all import write_index


class RenderTests(unittest.TestCase):
    def test_every_episode_is_discoverable_and_respects_cli_quality(self):
        paths = list((ROOT / "episodes").glob("*/scene.py"))
        self.assertTrue(paths)
        for path in paths:
            with self.subTest(episode=path.parent.name):
                self.assertEqual(scene_for(path.parent.name)[0], path)
                for node in ast.walk(ast.parse(path.read_text())):
                    if isinstance(node, ast.Assign):
                        for target in node.targets:
                            if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                                self.assertFalse(target.value.id == "config" and target.attr in
                                                 {"pixel_width", "pixel_height", "frame_rate"})

    def test_preview_and_final_have_separate_quality(self):
        preview = command("03.1", "preview", Path("media/preview"))
        final = command("03.1", "final", Path("media/final"))
        self.assertEqual(preview[preview.index("--profile") + 1], "preview")
        self.assertEqual(final[final.index("--profile") + 1], "final")
        self.assertEqual(PROFILES["preview"], (854, 480, 15))
        self.assertEqual(PROFILES["final"], (1920, 1080, 60))
        self.assertNotEqual(preview[-1], final[-1])

    def test_invalid_episode_does_not_resolve_arbitrary_paths(self):
        with self.assertRaises(ValueError):
            scene_for("../03.1")


class MetadataTests(unittest.TestCase):
    def metadata(self):
        return {"streams": [{"codec_type": "video", "codec_name": "h264", "width": 1920,
                             "height": 1080, "avg_frame_rate": "60/1", "duration": "25"}]}

    def test_valid_final(self):
        self.assertEqual(inspect_metadata(self.metadata(), "final")[2], [])

    def test_wrong_quality_and_audio_are_reported(self):
        metadata = self.metadata()
        metadata["streams"][0].update(width=854, height=480, avg_frame_rate="15/1", duration="5", codec_name="vp9")
        metadata["streams"].append({"codec_type": "audio"})
        self.assertEqual(len(inspect_metadata(metadata, "final")[2]), 4)

    def test_duration_is_validity_not_a_creative_limit(self):
        for duration in (0.1, 5, 46, 120):
            metadata = self.metadata()
            metadata["streams"][0]["duration"] = str(duration)
            self.assertEqual(inspect_metadata(metadata, "final")[2], [])
            self.assertTrue(all(0 <= t <= max(0, duration - 1 / 60)
                                for t in frame_times(duration, 60, None)))
        for duration in (0, -1, float("nan"), float("inf")):
            metadata = self.metadata()
            metadata["streams"][0]["duration"] = str(duration)
            with self.assertRaises(ValueError):
                inspect_metadata(metadata, "final")

    def test_unknown_fps_is_reported(self):
        metadata = self.metadata()
        metadata["streams"][0]["avg_frame_rate"] = "0/0"
        self.assertTrue(inspect_metadata(metadata, "final")[2])

    def test_final_frame_precedes_eof(self):
        self.assertAlmostEqual(frame_times(25, 60, [2, 8])[-1], 25 - 1 / 60)
        for value in (-1, 25, float("nan")):
            with self.assertRaises(ValueError):
                frame_times(25, 60, [value])

    def test_decode_failure_cannot_produce_a_pass_report(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "review"
            with patch("scripts.qa_video.subprocess.check_output", return_value=json.dumps(self.metadata())):
                with patch("scripts.qa_video.subprocess.run", side_effect=subprocess.CalledProcessError(1, "ffmpeg")):
                    with self.assertRaises(subprocess.CalledProcessError):
                        review(Path("damaged.mp4"), output, "final", None)
            self.assertFalse((output / "report.json").exists())


class BatchReviewTests(unittest.TestCase):
    def test_final_gallery_is_not_labeled_as_preview(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_index(output, [], profile="final")
            page = (output / "index.html").read_text()
            self.assertIn("1080p60", page)
            self.assertNotIn("480p15", page)

    def test_failed_render_does_not_link_to_old_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_index(output, [{"episode": "03.1", "status": "failed", "error": "<failure>"}])
            page = (output / "index.html").read_text()
            self.assertIn("&lt;failure&gt;", page)
            self.assertNotIn("frame-05.png", page)
            self.assertIn("03.1/run.log", page)

    def test_metadata_failure_is_not_counted_as_passed(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_index(output, [{"episode": "03.1", "status": "metadata-failed",
                                  "duration": 10, "problems": ["Too short"]}])
            page = (output / "index.html").read_text()
            self.assertIn("通过 0 集", page)
            self.assertIn("Too short", page)


if __name__ == "__main__":
    unittest.main()
