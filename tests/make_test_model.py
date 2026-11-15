"""Builds a small 'character-like' test mesh with concavities (no UVs -> exercises xatlas)."""
import sys
import numpy as np
import trimesh

parts = [
    trimesh.creation.icosphere(subdivisions=4, radius=0.55),                       # body
    trimesh.creation.icosphere(subdivisions=3, radius=0.35).apply_translation([0, 0.75, 0]),  # head
    trimesh.creation.cylinder(radius=0.5, height=0.08, sections=48).apply_transform(
        trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])).apply_translation([0, 1.02, 0]),  # hat brim
    trimesh.creation.box(extents=[0.3, 0.3, 0.3]).apply_translation([0, 1.2, 0]),   # hat
    trimesh.creation.capsule(radius=0.1, height=0.5).apply_transform(
        trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])).apply_translation([0.45, 0.15, 0]),
    trimesh.creation.capsule(radius=0.1, height=0.5).apply_transform(
        trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])).apply_translation([-0.95, 0.15, 0]),
    trimesh.creation.cone(radius=0.08, height=0.3).apply_translation([0, 0.75, 0.3]),  # nose (+Z = front)
]
trimesh.util.concatenate(parts).export(sys.argv[1])
