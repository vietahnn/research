import pandas as pd
import ast
import numpy as np

# Load dataset
csv_path = "../datasets/WLASL100/WLASL100_train_25fps.csv"
df = pd.read_csv(csv_path, encoding="utf-8")

# Get sequence lengths
sequence_lengths = []

for idx, row in df.iterrows():
    # Get length from any column with coordinate data
    seq_len = len(ast.literal_eval(row["leftEar_X"]))
    sequence_lengths.append(seq_len)

# Calculate statistics
sequence_lengths = np.array(sequence_lengths)

print(f"Total samples: {len(sequence_lengths)}")
print(f"Min length: {sequence_lengths.min()}")
print(f"Max length: {sequence_lengths.max()}")
print(f"Mean length: {sequence_lengths.mean():.2f}")
print(f"Median length: {np.median(sequence_lengths):.2f}")
print(f"Std deviation: {sequence_lengths.std():.2f}")
print(f"\nPercentiles:")
print(f"25th percentile: {np.percentile(sequence_lengths, 25):.2f}")
print(f"50th percentile: {np.percentile(sequence_lengths, 50):.2f}")
print(f"75th percentile: {np.percentile(sequence_lengths, 75):.2f}")
print(f"90th percentile: {np.percentile(sequence_lengths, 90):.2f}")
print(f"95th percentile: {np.percentile(sequence_lengths, 95):.2f}")

# Distribution
print(f"\nDistribution:")
print(f"< 50 frames: {(sequence_lengths < 50).sum()} samples ({(sequence_lengths < 50).sum() / len(sequence_lengths) * 100:.1f}%)")
print(f"50-100 frames: {((sequence_lengths >= 50) & (sequence_lengths < 100)).sum()} samples ({((sequence_lengths >= 50) & (sequence_lengths < 100)).sum() / len(sequence_lengths) * 100:.1f}%)")
print(f"100-150 frames: {((sequence_lengths >= 100) & (sequence_lengths < 150)).sum()} samples ({((sequence_lengths >= 100) & (sequence_lengths < 150)).sum() / len(sequence_lengths) * 100:.1f}%)")
print(f">= 150 frames: {(sequence_lengths >= 150).sum()} samples ({(sequence_lengths >= 150).sum() / len(sequence_lengths) * 100:.1f}%)")
