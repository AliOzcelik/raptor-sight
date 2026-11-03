"""Bbox-level pan-tilt tracking simulator (PROJECT.md section 4)."""

import gymnasium as gym

from sim.config import SimConfig
from sim.core import BatchSim
from sim.env import PanTiltEnv

gym.register(id = "RaptorSight/PanTilt-v0", entry_point = "sim.env:PanTiltEnv")
