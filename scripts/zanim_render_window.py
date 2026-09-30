"""A render-only window view for the pinned Zanim 0.7 renderer.

The authoring scene stays intact. Objects outside the requested lifetime
window and objects with identically zero opacity in that entire window
cannot contribute pixels. Keep parent lookup/channel tables unchanged and
filter only registry traversal and unrelated timeline event traversal.
This adapter uses rc1 internal registry layout; verify against the original
native raster before exporting any video. Never serialize this window view.
"""

from copy import copy
from zanim.timeline import OpacityClip, InterpolationClip


def render_window(scene, start, end):
    view = copy(scene)
    retained = []
    for registered in scene._registry:
        if registered.object_id == 0:
            retained.append(registered)
            continue
        added, removed = scene._effective_lifetime(registered)
        if added >= end or (removed is not None and removed <= start):
            continue
        initial = getattr(registered.initial, "opacity", None)
        if initial is not None:
            lo = max(start, added)
            hi = min(end, removed) if removed is not None else end
            values = [
                scene._opacity_at(registered.object_id, initial, lo),
                scene._opacity_at(registered.object_id, initial, hi),
            ]
            for clip in scene._timeline._channel_clips(
                OpacityClip, registered.object_id
            ):
                if clip.span.start <= hi and clip.span.end >= lo:
                    values += [clip.before, clip.after]
            if all(value == 0 for value in values):
                continue
        retained.append(registered)
    view._registry = retained
    ids = {r.object_id for r in retained} | {-1}
    for r in retained:
        ids.update(r.parent_ids)
    view._timeline = copy(scene._timeline)
    view._timeline.clips = [
        clip
        for clip in scene._timeline.clips
        if isinstance(clip, InterpolationClip) or getattr(clip, "object_id", -1) in ids
    ]
    return view


def verify_window(scene, view, start, end):
    import hashlib
    from zanim.render.frame import render_snapshot_rgb0

    original = bytearray(scene.width * scene.height * 4)
    optimized = bytearray(len(original))
    # Include the last rendered frame, and four interior positions.
    times = [start + (end - start) * q for q in (0, 0.2, 0.4, 0.6, 0.8)] + [
        end - 1 / scene.fps
    ]
    for t in times:
        render_snapshot_rgb0(original, scene.evaluate(t), scene.canvas)
        render_snapshot_rgb0(optimized, view.evaluate(t), view.canvas)
        if hashlib.sha256(original).digest() != hashlib.sha256(optimized).digest():
            raise ValueError(f"Render window changed pixels at {t:.6f}s")
    return len(times)
