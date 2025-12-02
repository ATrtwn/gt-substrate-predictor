"""
Prepare substrate data in KPGT format.
KPGT expects: data/Substrate/Substrate.csv with 'smiles' column
"""
import pandas as pd
from pathlib import Path

# Set up paths
ROOT = Path(__file__).parent.parent
data_dir = ROOT / "data"
substrate_dir = data_dir / "Substrate"

# Create Substrate directory if it doesn't exist
substrate_dir.mkdir(exist_ok=True)

# Load substrate SMILES data
df = pd.read_csv(data_dir / "Substrate_SMILES.csv")

# KPGT expects a 'smiles' column (lowercase)
# We'll use the SMILES_isomeric_1 column
# Note: For feature extraction only, we need a dummy label column
df_kpgt = pd.DataFrame({
    'smiles': df['SMILES_isomeric_1'].values,
    'dummy_label': 0  # Dummy label for feature extraction
})

# Remove rows with missing SMILES
df_kpgt = df_kpgt.dropna(subset=['smiles'])

print(f"Prepared {len(df_kpgt)} substrates with valid SMILES")
print(f"First few rows:")
print(df_kpgt.head())

# Save to KPGT format
output_path = substrate_dir / "Substrate.csv"
df_kpgt.to_csv(output_path, index=False)
print(f"\nSaved to: {output_path}")
