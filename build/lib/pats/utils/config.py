"""
Configurazione per il caricamento dati PATS.
"""

from pathlib import Path


# ============================================================================
# CONFIGURAZIONE
# ============================================================================

DATA_ROOT = Path('/home/paolo/Projects/Gesture/pats/data')
PROCESSED_DIR = DATA_ROOT / 'processed'
INTERVALS_CSV = DATA_ROOT / 'cmu_intervals_df.csv'
