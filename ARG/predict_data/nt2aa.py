#!/usr/bin/env python3
"""
nt2aa.py - Nucleotide to Amino Acid Translation Tool (Preserves Original Headers)

Usage:
  nt2aa.py <input_file> <output_file> [--min_len MIN_LENGTH] [--table TABLE_ID]
  nt2aa.py (-h | --help)

Options:
  -h --help         Show this screen
  --min_len LEN     Minimum ORF length (amino acids) [default: 25]
  --table ID        NCBI translation table ID [default: 11]
"""

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Data import CodonTable
import re
import sys
from docopt import docopt


def create_extended_codon_table(table_id=11):
    """
    Create extended codon table with support for ambiguous bases
    """
    # Get standard codon table
    std_table = CodonTable.generic_by_id[table_id]
    table_dict = std_table.forward_table.copy()
    stop_codons = std_table.stop_codons

    # Add stop codons to table
    for codon in stop_codons:
        table_dict[codon] = "*"

    # Handle ambiguous bases (IUPAC codes)
    ambiguous_bases = {
        'R': ['A', 'G'],  # Purine
        'Y': ['C', 'T'],  # Pyrimidine
        'S': ['G', 'C'],  # Strong
        'W': ['A', 'T'],  # Weak
        'K': ['G', 'T'],  # Keto
        'M': ['A', 'C'],  # Amino
        'B': ['C', 'G', 'T'],  # Not A
        'D': ['A', 'G', 'T'],  # Not C
        'H': ['A', 'C', 'T'],  # Not G
        'V': ['A', 'C', 'G'],  # Not T
        'N': ['A', 'C', 'G', 'T']  # Any
    }

    # Generate all possible ambiguous codons
    for base1 in ambiguous_bases:
        for base2 in ambiguous_bases:
            for base3 in ambiguous_bases:
                codon = base1 + base2 + base3

                # Skip if already in table
                if codon in table_dict:
                    continue

                # Get all possible unambiguous codons
                possible_codons = []
                for b1 in ambiguous_bases[base1]:
                    for b2 in ambiguous_bases[base2]:
                        for b3 in ambiguous_bases[base3]:
                            possible_codons.append(b1 + b2 + b3)

                # Get unique translations
                translations = set()
                for uc in possible_codons:
                    if uc in table_dict:
                        translations.add(table_dict[uc])

                # Assign translation
                if len(translations) == 1:
                    table_dict[codon] = translations.pop()
                else:
                    # If multiple possibilities or no translation, use 'X'
                    table_dict[codon] = 'X'

    return table_dict


def find_longest_orf(aa_seq, min_len=25):
    """
    Find the longest ORF starting with M in an amino acid sequence
    """
    # Split at stop codons
    orf_chunks = aa_seq.split('*')

    longest_orf = ""
    max_length = 0

    for chunk in orf_chunks:
        # Find all M-started ORFs in this chunk
        start_positions = [pos for pos, char in enumerate(chunk) if char == 'M']

        for start in start_positions:
            orf_candidate = chunk[start:]
            length = len(orf_candidate)

            # Check if longer than current longest
            if length >= min_len and length > max_length:
                longest_orf = orf_candidate
                max_length = length

    return longest_orf if max_length > 0 else ""


def translate_sequence(dna_seq, codon_table, min_len=25):
    """
    Translate nucleotide sequence to amino acids and find longest ORF

    Args:
        dna_seq (Seq): DNA sequence
        codon_table (dict): Codon to amino acid mapping
        min_len (int): Minimum ORF length

    Returns:
        str: Longest ORF amino acid sequence
    """
    seq_str = str(dna_seq).upper()
    revcomp_str = str(dna_seq.reverse_complement()).upper()

    longest_orf = ""
    max_length = 0

    # Process all 6 reading frames
    for frame, seq in enumerate([seq_str, seq_str[1:], seq_str[2:],
                                 revcomp_str, revcomp_str[1:], revcomp_str[2:]]):

        # Translate in chunks to handle ambiguous bases
        aa_seq = []
        for i in range(0, len(seq) - 2, 3):
            codon = seq[i:i + 3]
            aa = codon_table.get(codon, 'X')  # Use 'X' for unknown codons
            aa_seq.append(aa)

        aa_seq = ''.join(aa_seq)

        # Find longest ORF in this frame
        frame_orf = find_longest_orf(aa_seq, min_len)
        frame_length = len(frame_orf)

        # Update if longer than current longest
        if frame_length > max_length:
            longest_orf = frame_orf
            max_length = frame_length

    return longest_orf


def main():
    args = docopt(__doc__)

    # Parse arguments
    input_file = args['<input_file>']
    output_file = args['<output_file>']
    min_len = int(args['--min_len'])
    table_id = int(args['--table'])

    # Create codon table
    print(f"Using NCBI translation table {table_id}")
    codon_table = create_extended_codon_table(table_id)

    # Process sequences
    translated_count = 0
    skipped_count = 0

    with open(output_file, 'w') as out_handle:
        for record in SeqIO.parse(input_file, "fasta"):
            # Translate and find longest ORF
            aa_seq = translate_sequence(record.seq, codon_table, min_len)

            if aa_seq:
                # Create new record with same header
                aa_record = SeqRecord(
                    Seq(aa_seq),
                    id=record.id,
                    description=record.description
                )
                SeqIO.write(aa_record, out_handle, "fasta")
                translated_count += 1
            else:
                skipped_count += 1

    print(f"\nTranslation complete!")
    print(f"  Translated sequences: {translated_count}")
    print(f"  Skipped sequences (no ORF found): {skipped_count}")
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
