"""
Funzioni di caricamento dati da file HDF5 e CSV.
"""

from pats.data import HDF5
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, Optional

from .config import INTERVALS_CSV


# ============================================================================
# FUNZIONI DI CARICAMENTO DATI
# ============================================================================

def load_pose_data(h5_file: Path) -> np.ndarray:
    """
    Carica i dati di pose da file HDF5 e li converte in formato (frames, 52, 2).
    
    Args:
        h5_file: Path al file HDF5
        
    Returns:
        Array numpy con shape (n_frames, 52, 2) contenente coordinate XY
    """
    h5 = HDF5.h5_open(str(h5_file), 'r')
    raw = h5['pose/data'][:]
    h5.close()
    
    # Converte da [X0...X51, Y0...Y51] a (frames, 52, 2)
    n_frames = raw.shape[0]
    pose = np.zeros((n_frames, 52, 2))
    pose[:, :, 0] = raw[:, :52]  # X coordinates
    pose[:, :, 1] = raw[:, 52:]  # Y coordinates
    
    return pose


def load_text_data(h5_file: Path) -> Optional[pd.DataFrame]:
    """
    Carica i metadati del testo da file HDF5.
    
    Args:
        h5_file: Path al file HDF5
        
    Returns:
        DataFrame con colonne ['Word', 'start_', 'end_', 'start_frame', 'end_frame']
        oppure None se non disponibile
    """
    try:
        text_meta = pd.read_hdf(str(h5_file), key='text/meta')
        if len(text_meta) > 0 and 'Word' in text_meta.columns:
            return text_meta
    except Exception:
        pass
    return None


def get_interval_metadata(speaker: str, interval_id: str) -> Dict:
    """
    Recupera i metadati di un intervallo dal CSV.
    
    Args:
        speaker: Nome dello speaker
        interval_id: ID dell'intervallo
        
    Returns:
        Dizionario con metadati dell'intervallo
    """
    df = pd.read_csv(INTERVALS_CSV)
    df = df[df['speaker'] == speaker]
    row = df[df['interval_id'] == interval_id].iloc[0]
    
    return {
        'interval_id': interval_id,
        'duration': float(row['delta_time']),
        'speaker': speaker
    }
