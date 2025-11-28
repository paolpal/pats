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

from data import HDF5
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


# ============================================================================
# CONFIGURAZIONE
# ============================================================================

DATA_ROOT = Path('/home/paolo/Projects/Gesture/pats/data')
PROCESSED_DIR = DATA_ROOT / 'processed'
INTERVALS_CSV = DATA_ROOT / 'cmu_intervals_df.csv'


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


# ============================================================================
# FUNZIONE PRINCIPALE DI COSTRUZIONE DATI
# ============================================================================

def build_sample_data(speaker: str, interval_id: str, verbose: bool = True) -> Dict:
    """
    Costruisce una struttura dati completa per un sample.
    
    Args:
        speaker: Nome dello speaker (es. 'fallon', 'bee', 'conan')
        interval_id: ID dell'intervallo (es. 'cmu0000004481')
        verbose: Se True, stampa informazioni durante il caricamento
        
    Returns:
        Dizionario con struttura:
        {
            "text": str,
            "pose": np.ndarray (n_frames, 52, 2),
            "duration": float,
            "n_frames": int,
            "words": List[Dict[str, any]]
        }
    """
    # Path al file HDF5
    h5_file = PROCESSED_DIR / speaker / f'{interval_id}.h5'
    
    if not h5_file.exists():
        raise FileNotFoundError(f"File non trovato: {h5_file}")
    
    # Carica pose
    pose = load_pose_data(h5_file)
    n_frames = pose.shape[0]
    
    # Carica testo
    text_meta = load_text_data(h5_file)
    
    # Recupera metadati intervallo
    interval_meta = get_interval_metadata(speaker, interval_id)
    duration = interval_meta['duration']
    
    # Costruisce la struttura dati
    sample = {
        "pose": pose,
        "duration": duration,
        "n_frames": n_frames
    }
    
    # Processa il testo se disponibile
    if text_meta is not None:
        # Testo completo
        sample["text"] = text_meta['Word'].str.cat(sep=' ')
        
        # Lista di parole con timing
        words = []
        for _, row in text_meta.iterrows():
            words.append({
                "word": row['Word'],
                "start": float(row['start_frame']),
                "end": float(row['end_frame'])
            })
        sample["words"] = words
    else:
        sample["text"] = None
        sample["words"] = []
    
    if verbose:
        print(f"✓ Sample caricato: {interval_id}")
        print(f"  Speaker: {speaker}")
        print(f"  Frames: {n_frames}")
        print(f"  Duration: {duration:.2f}s")
        print(f"  FPS: {n_frames/duration:.2f}")
        if sample["text"]:
            print(f"  Text: {sample['text'][:60]}...")
    
    return sample


# ============================================================================
# FUNZIONI UTILITY
# ============================================================================

def get_speaker_intervals(speaker: str, split: Optional[str] = None) -> List[str]:
    """
    Recupera tutti gli interval_id per uno speaker.
    
    Args:
        speaker: Nome dello speaker
        
    Returns:
        Lista di interval_id
    """
    df = pd.read_csv(INTERVALS_CSV)
    if split is not None:
        df = df[ df['dataset'] == split ]
    df = df[df['speaker'] == speaker]
    return df['interval_id'].tolist()


def load_multiple_samples(speaker: str, interval_ids: List[str], 
                         verbose: bool = False) -> List[Dict]:
    """
    Carica multipli sample in batch.
    
    Args:
        speaker: Nome dello speaker
        interval_ids: Lista di interval_id da caricare
        verbose: Se True, stampa info per ogni sample
        
    Returns:
        Lista di dizionari con struttura sample
    """
    samples = []
    for interval_id in interval_ids:
        try:
            sample = build_sample_data(speaker, interval_id, verbose=verbose)
            samples.append(sample)
        except Exception as e:
            print(f"✗ Errore caricamento {interval_id}: {e}")
    
    return samples


# ============================================================================
# ESEMPIO DI UTILIZZO
# ============================================================================

if __name__ == "__main__":
    # Carica un singolo sample
    intervals = get_speaker_intervals('fallon', split='dev')
    print(f"Intervalli disponibili per 'fallon': {len(intervals)}")

    sample = build_sample_data('fallon', intervals[5], verbose=True)
    
    print("\n" + "="*60)
    print("STRUTTURA DATI SAMPLE:")
    print("="*60)
    print(f"Text: {sample['text']}")
    print(f"Pose shape: {sample['pose'].shape}")
    print(f"Duration: {sample['duration']}s")
    print(f"N frames: {sample['n_frames']}")
    print(f"Words: {len(sample['words'])} parole")
    print("\nPrime 3 parole:")
    for word in sample['words'][:3]:
        print(f"  - {word}")
