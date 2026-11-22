"""PushT "worlds": same physics, different appearance (the paper's Parallel Worlds, Eq. 8).
Các "thế giới" PushT: cùng vật lý, khác hình ảnh (Parallel Worlds của paper, Eq. 8).

A world only changes colors (background, target zone, T-block, agent). Object poses and dynamics are
untouched, so replaying a trajectory's actions from the same start state reproduces it exactly.
Một thế giới chỉ đổi màu (nền, vùng đích, khối T, tác tử). Vị trí và động lực học giữ nguyên, nên
chạy lại đúng chuỗi action từ cùng trạng thái đầu sẽ tái tạo y hệt quỹ đạo.
"""
from __future__ import annotations

import numpy as np
import pygame
import pymunk
import pymunk.pygame_util
from gym_pusht.envs.pusht import PushTEnv

from s2s.config import TASK

# name -> (background, goal zone, T-block, agent) as RGB / tên -> (nền, vùng đích, khối T, tác tử)
WORLDS: dict[str, tuple[tuple[int, int, int], ...]] = {
    "source": ((255, 255, 255), (144, 238, 144), (119, 136, 153), (65, 105, 225)),  # gym_pusht defaults
    # seen worlds, used for Parallel Worlds data / thế giới đã thấy, dùng để sinh dữ liệu
    "w1": ((240, 230, 200), (120, 200, 120), (160, 82, 45), (70, 70, 200)),
    "w2": ((200, 220, 240), (250, 200, 120), (90, 90, 90), (200, 60, 60)),
    "w3": ((60, 60, 70), (100, 180, 100), (200, 200, 210), (240, 200, 60)),
    "w4": ((230, 200, 220), (150, 220, 230), (120, 60, 140), (40, 140, 90)),
    "w5": ((210, 240, 210), (255, 170, 170), (70, 110, 160), (230, 120, 30)),
    # unseen worlds, evaluation only / thế giới chưa thấy, chỉ để đánh giá
    "u1": ((255, 250, 205), (173, 216, 230), (178, 34, 34), (34, 139, 34)),
    "u2": ((40, 40, 60), (220, 220, 120), (100, 200, 220), (255, 105, 180)),
    "u3": ((190, 190, 190), (200, 160, 220), (30, 30, 30), (255, 140, 0)),
}


class WorldPushT(PushTEnv):
    """gym_pusht PushT with a configurable color scheme and a task description for LeRobot eval.
    PushT của gym_pusht với bảng màu tùy chỉnh và mô tả nhiệm vụ cho bước đánh giá của LeRobot."""

    def __init__(self, world: str = "source", **kwargs):
        kwargs.setdefault("obs_type", "pixels_agent_pos")
        super().__init__(**kwargs)
        self.world = world
        self.bg, self.goal_color, self.block_color, self.agent_color = WORLDS[world]

    task = TASK  # LeRobot eval looks for `task` / `task_description` / bước đánh giá của LeRobot tìm các thuộc tính này

    def task_description(self) -> str:
        return TASK

    def _setup(self):
        super()._setup()
        for shape in self.block.shapes:
            shape.color = pygame.Color(*self.block_color)
        for shape in self.agent.shapes:
            shape.color = pygame.Color(*self.agent_color)

    def _draw(self):
        screen = pygame.Surface((512, 512))
        screen.fill(self.bg)
        draw_options = pymunk.pygame_util.DrawOptions(screen)
        goal_body = self.get_goal_pose_body(self.goal_pose)
        for shape in self.block.shapes:
            pts = [pymunk.pygame_util.to_pygame(goal_body.local_to_world(v), draw_options.surface)
                   for v in shape.get_vertices()]
            pygame.draw.polygon(screen, pygame.Color(*self.goal_color), pts + [pts[0]])
        self.space.debug_draw(draw_options)
        return screen

    def reset(self, seed=None, options=None):
        """Always reset through an explicit start state and remember it in `self.init_state`.
        Reading the state back after reset does NOT reproduce the episode (gym_pusht sets the angle
        about the centre of mass and runs one physics step), so replays must use `init_state`.
        Luôn reset qua một trạng thái đầu tường minh và lưu nó vào `self.init_state`. Đọc lại
        trạng thái sau reset KHÔNG tái tạo được episode (gym_pusht xoay quanh trọng tâm và chạy
        thêm một bước vật lý), nên khi chạy lại phải dùng `init_state`."""
        if options is not None and options.get("reset_to_state") is not None:
            state = np.asarray(options["reset_to_state"], dtype=np.float64)
            super().reset(seed=seed)  # seeds the RNG / khởi tạo RNG
        else:
            super().reset(seed=seed)
            # same distribution as gym_pusht / cùng phân phối với gym_pusht
            state = np.array([self.np_random.integers(50, 450), self.np_random.integers(50, 450),
                              self.np_random.integers(100, 400), self.np_random.integers(100, 400),
                              self.np_random.uniform(-np.pi, np.pi)], dtype=np.float64)
        self.init_state = state.copy()
        return super().reset(options={"reset_to_state": state})


def make_env(world: str = "source", vlv_size: int = 224) -> WorldPushT:
    """One env. `vlv_size` sets render() resolution used for verifier videos.
    Một env. `vlv_size` là độ phân giải render() dùng cho video của verifier."""
    return WorldPushT(world=world, render_mode="rgb_array",
                      visualization_width=vlv_size, visualization_height=vlv_size)
