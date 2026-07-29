"""Concept analysis: turn a trained BSF into a self-contained explorable artifact.

``compute`` holds the geometry/statistics (torch + numpy); ``build`` drives a
full analysis over a capture root; ``types`` is the on-disk contract the
dashboard reads (numpy only -- no torch needed to view an artifact).
"""
from .types import (
    Analysis, Meta, ConceptExample, ARTIFACT_VERSION,
    AnalysisError, ArtifactVersionError, ArtifactShapeError,
    CheckpointMismatchError,
)

__all__ = [
    'Analysis', 'Meta', 'ConceptExample', 'ARTIFACT_VERSION',
    'AnalysisError', 'ArtifactVersionError', 'ArtifactShapeError',
    'CheckpointMismatchError',
]
