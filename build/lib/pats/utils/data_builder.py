"""
Funzioni per la costruzione e il caricamento batch di sample dal dataset PATS.

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

from pathlib import Path
import pandas as pd
from typing import Dict, List, Optional

from .config import get_config
from .loaders import load_pose_data, load_text_data, get_interval_metadata, get_missing_intervals


# ============================================================================
# FUNZIONE PRINCIPALE DI COSTRUZIONE DATI
# ============================================================================

def build_sample_data(speaker: str, interval_id: str, data_root: Optional[Path] = None, verbose: bool = True) -> Dict:
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
    config = get_config(data_root)
    PROCESSED_DIR = config['PROCESSED_DIR']
    h5_file = PROCESSED_DIR / speaker / f'{interval_id}.h5'
    
    if not h5_file.exists():
        raise FileNotFoundError(f"File non trovato: {h5_file}")
    
    # Carica pose
    pose = load_pose_data(h5_file)
    n_frames = pose.shape[0]
    
    # Carica testo
    text_meta = load_text_data(h5_file)
    
    # Recupera metadati intervallo
    interval_meta = get_interval_metadata(speaker, interval_id, data_root=data_root)
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

def get_speaker_intervals(speaker: str, split: Optional[str] = None, data_root: Optional[Path] = None) -> List[str]:
    """
    Recupera tutti gli interval_id per uno speaker.
    
    Args:
        speaker: Nome dello speaker
        split: Nome del dataset split (es. 'train', 'dev', 'test')
        
    Returns:
        Lista di interval_id
    """
    config = get_config(data_root)
    INTERVALS_CSV = config['INTERVALS_CSV']
    df = pd.read_csv(INTERVALS_CSV)
    if split is not None:
        df = df[df['dataset'] == split]
    df = df[df['speaker'] == speaker]
    return df['interval_id'].tolist()


def load_multiple_samples(speaker: str, interval_ids: List[str], data_root: Optional[Path] = None,
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
    missing_intervals = get_missing_intervals(data_root)
    samples = []
    for interval_id in interval_ids:
        if interval_id in missing_intervals:
            if verbose:
                print(f"✗ Intervallo mancante: {interval_id}")
            continue
        try:
            sample = build_sample_data(speaker, interval_id, data_root=data_root, verbose=verbose)
            samples.append(sample)
        except Exception as e:
            print(f"✗ Errore caricamento {interval_id}: {e}")
    
    return samples

def get_all_missing_intervals(data_root: Optional[Path] = None):
    """
    Carica gli intervalli mancanti dal file HDF5.
    
    Returns:
        DataFrame con gli intervalli mancanti
    """
    return get_missing_intervals(data_root)
