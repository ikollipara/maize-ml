"""
transpose_vcf.py
Jensina Davis <jdavis132@huskers.unl.edu>
Ian Kollipara <ikollipara2@huskers.unl.edu>

Transpose a VCF file to a CSV file.
"""

# Imports
import io
import os
import re
import sys

try:
    import pandas as pd
except ImportError as e:
    sys.stderr.write('Missing required package: pandas\n')
    exit(1)

def read_vcf(vcf_path):
    """ Read a VCF file into a pandas DataFrame.

    Args:
        vcf_path (str): The path to the VCF file.

    Returns:
        pandas.DataFrame: The VCF file as a DataFrame.
    """

    with open(vcf_path, 'r') as f:
        lines = (l for l in f if not l.startswith('##'))
    return pd.read_csv(
        io.StringIO(''.join(lines)),
        dtype={'#CHROM': str, 'POS': int, 'ID': str, 'REF': str, 'ALT': str,
            'QUAL': str, 'FILTER': str, 'INFO': str},
        sep='\t'
    ).rename(columns={'#CHROM': '#CHROM'})

if len(sys.argv) == 2 and sys.argv[1] in ('-h', '--help'):
    sys.stdout.write('Usage: python transpose_vcf.py <output_dir> ...<vcf_files>\n')
    sys.stdout.write('\t output_dir: The directory to write the transposed CSV files to\n')
    sys.stdout.write('\t vcf_files: The VCF files to transpose\n')
    sys.stdout.write('\tExample: python transpose_vcf.py genotype/ chr_1.vcf chr_2.vcf chr_3.vcf\n')
    exit(0)

if len(sys.argv) < 3:
    sys.stderr.write('Error: Missing required arguments\n')
    sys.stderr.write('Usage: python transpose_vcf.py <output_dir> ...<vcf_files>\n')
    exit(1)

output_dir = sys.argv[1]
vcf_files = sys.argv[2:] if not sys.argv[2] == '*' else [f for f in os.listdir() if f.endswith('.vcf')]

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for vcf_file in vcf_files:
    if not vcf_file.endswith('.vcf'):
        sys.stderr.write(f'Error: {vcf_file} is not a VCF file. Skipping...\n')
        continue

    df = read_vcf(vcf_file)
    df = df.T
    df.to_csv(os.path.join(output_dir, f'transposed_{os.path.basename(vcf_file).replace(".vcf", ".csv")}'))
    sys.stdout.write(f'Transposed {vcf_file}\n')
