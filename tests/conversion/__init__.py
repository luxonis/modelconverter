"""Conversion tests, run inside the per-backend Docker images.

Each test converts a model for one target and checks the result, so a
module here only runs where that target's toolchain is available. Select
a target with its marker, e.g. ``pytest -m rvc2``.
"""
