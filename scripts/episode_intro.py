"""Chinese opening card shared by all animation scenes."""

import textwrap
from manim import DOWN, FadeOut, Text, VGroup, WHITE, BLUE_A, config, linear
from scripts.episode_info import EPISODES, introduction

INTRO_DURATION = 4.0


def make_intro_card(episode: str) -> VGroup:
    if episode not in EPISODES:
        raise ValueError(f"Missing episode introduction: {episode}")
    title, explanation = introduction(episode)
    heading = Text(title, font="Noto Sans CJK SC", font_size=44, color=WHITE)
    if heading.width > config.frame_width - 1.6:
        heading.scale_to_fit_width(config.frame_width - 1.6)
    description = Text("\n".join(textwrap.wrap(explanation, width=26)),
                       font="Noto Sans CJK SC", font_size=28, color=BLUE_A,
                       line_spacing=1.2)
    if description.width > config.frame_width - 2.0:
        description.scale_to_fit_width(config.frame_width - 2.0)
    return VGroup(heading, description).arrange(DOWN, buff=0.6).move_to([0, 0, 0])


def show_episode_intro(scene, episode: str) -> None:
    card = make_intro_card(episode)
    scene.add(card)
    scene.wait(INTRO_DURATION - 0.2)
    scene.play(FadeOut(card), run_time=0.2, rate_func=linear)
