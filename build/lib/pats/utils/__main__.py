"""
Esempio di utilizzo del modulo utils.

Uso:
    python -m utils --speaker fallon --split dev --index 5
    python -m utils --speaker bee --interval-id cmu0000004481
    python -m utils --speaker conan --split train --num-samples 10
"""

import argparse
from .data_builder import build_sample_data, get_speaker_intervals, load_multiple_samples


def main():
    parser = argparse.ArgumentParser(
        description='Carica e visualizza sample dal dataset PATS'
    )
    
    parser.add_argument(
        '--speaker',
        type=str,
        required=True,
        choices=['fallon', 'bee', 'conan', 'oliver', 'rock', 'chemistry', 'seth', 'ellen', 'lec_cosmic', 'lec_evol'],
        help='Nome dello speaker'
    )
    
    parser.add_argument(
        '--interval-id',
        type=str,
        help='ID specifico dell\'intervallo da caricare (es. cmu0000004481)'
    )
    
    parser.add_argument(
        '--split',
        type=str,
        choices=['train', 'dev', 'test'],
        help='Dataset split da utilizzare'
    )
    
    parser.add_argument(
        '--index',
        type=int,
        default=0,
        help='Indice dell\'intervallo da caricare (default: 0)'
    )
    
    parser.add_argument(
        '--num-samples',
        type=int,
        help='Numero di sample da caricare in batch'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Stampa informazioni dettagliate'
    )
    
    args = parser.parse_args()
    
    # Caso 1: interval_id specifico fornito
    if args.interval_id:
        print(f"Caricamento sample: {args.interval_id}")
        sample = build_sample_data(args.speaker, args.interval_id, verbose=args.verbose)
        print_sample_info(sample)
        return
    
    # Caso 2: carica da split
    intervals = get_speaker_intervals(args.speaker, split=args.split)
    print(f"Intervalli disponibili per '{args.speaker}' (split: {args.split or 'tutti'}): {len(intervals)}")
    
    if len(intervals) == 0:
        print("Nessun intervallo trovato.")
        return
    
    # Caso 3: carica multipli sample
    if args.num_samples:
        num_to_load = min(args.num_samples, len(intervals))
        print(f"\nCaricamento di {num_to_load} sample...")
        samples = load_multiple_samples(args.speaker, intervals[:num_to_load], verbose=args.verbose)
        print(f"\nCaricati {len(samples)} sample con successo.")
        
        # Stampa statistiche
        if samples:
            total_frames = sum(s['n_frames'] for s in samples)
            total_duration = sum(s['duration'] for s in samples)
            print(f"\nStatistiche:")
            print(f"  Frames totali: {total_frames}")
            print(f"  Durata totale: {total_duration:.2f}s")
            print(f"  FPS medio: {total_frames/total_duration:.2f}")
        return
    
    # Caso 4: carica singolo sample per indice
    if args.index >= len(intervals):
        print(f"Errore: indice {args.index} fuori range (max: {len(intervals)-1})")
        return
    
    print(f"\nCaricamento intervallo all'indice {args.index}: {intervals[args.index]}")
    sample = build_sample_data(args.speaker, intervals[args.index], verbose=args.verbose)
    print_sample_info(sample)


def print_sample_info(sample):
    """Stampa informazioni dettagliate su un sample."""
    print("\n" + "="*60)
    print("STRUTTURA DATI SAMPLE:")
    print("="*60)
    print(f"Text: {sample['text']}")
    print(f"Pose shape: {sample['pose'].shape}")
    print(f"Duration: {sample['duration']}s")
    print(f"N frames: {sample['n_frames']}")
    print(f"FPS: {sample['n_frames']/sample['duration']:.2f}")
    print(f"Words: {len(sample['words'])} parole")
    
    if sample['words']:
        print("\nPrime 3 parole:")
        for word in sample['words'][:3]:
            print(f"  - {word}")


if __name__ == "__main__":
    main()
