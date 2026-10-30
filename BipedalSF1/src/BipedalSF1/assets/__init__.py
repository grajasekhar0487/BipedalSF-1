# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Paths and configurations for project-owned assets."""

from pathlib import Path

BIPEDALSF1_ASSETS_DIR = Path(__file__).resolve().parent / "data"
"""Path to project-owned asset data."""

# The robot model is maintained alongside the task package so its layered USD
# and the URDF/MJCF source description stay together.
TRON2A_USD_PATH = Path(__file__).resolve().parents[4] / "SF_TRON2A" / "usd" / "robot.usd"
"""Path to the SF-TRON2A robot USD in the repository checkout."""
