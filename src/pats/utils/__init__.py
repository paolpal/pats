"""
Modulo per il caricamento e processing di dati multimodali (pose + testo) dal dataset PATS.

Struttura dati di output:
{
    "text": "See, C-H-O-H-H-H-H.",
    "pose": array(n_frames, 52, 2),  # Coordinate XY dei 52 joint
    "duration": 2.66,
    "n_frames": 79,
    "words": [
        {"word": "See,", "end": 0.76},
        {"word": "C", "end": 1.12},
        ...
    ]
}
"""

# Export delle funzioni principali
from .config import get_config
from .loaders import load_pose_data, load_text_data, get_interval_metadata
from .data_builder import (
    build_sample_data,
    get_speaker_intervals,
    load_multiple_samples,
    get_all_missing_intervals
)


__all__ = [
    # Configurazione
    'get_config',
    # Loaders
    'load_pose_data',
    'load_text_data',
    'get_interval_metadata',
    # Data builder
    'build_sample_data',
    'get_speaker_intervals',
    'load_multiple_samples',
    'get_all_missing_intervals'
]
