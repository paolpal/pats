"""
Configurazione per il caricamento dati PATS.
"""

from pathlib import Path
from typing import Optional

# ============================================================================
# CONFIGURAZIONE
# ============================================================================

DATA_ROOT = Path('/home/paolo/Projects/Gesture/pats/data')

def get_config(data_root: Optional[Path] = None) -> dict:
    """
    Restituisce la configurazione aggiornata con un data_root opzionale.
    """
    if data_root is None:
        data_root = DATA_ROOT
    
    return {
        'DATA_ROOT': data_root,
        'PROCESSED_DIR': data_root / 'processed',
        'INTERVALS_CSV': data_root / 'cmu_intervals_df.csv',
        'MISSING_INTERVALS_H5': data_root / 'missing_intervals.h5'
    }