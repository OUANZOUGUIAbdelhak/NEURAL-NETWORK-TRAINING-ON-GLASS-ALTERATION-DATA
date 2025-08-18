import pandas as pd
from collections import defaultdict

# >>> 1) Point this to your CSV <<<
INPUT_CSV = "/home/intra.cea.fr/ao280403/Bureau/documents_bd_finale/Convert Composition/Data_From_Faijan_Oxides.csv"         # e.g., "glasses.csv"
OUTPUT_XLSX = "elements_mol_percent.xlsx"

# --- Stoichiometry per oxide/halide: moles of each element per 1 mole of oxide ---
# (Works for mol% since mol% is proportional to moles.)
STOICH = {
    "AL2O3": {"Al": 2, "O": 3},
    "B2O3":  {"B": 2, "O": 3},
    "BAO":   {"Ba": 1, "O": 1},
    "CAO":   {"Ca": 1, "O": 1},
    "CE2O3": {"Ce": 2, "O": 3},
    "CL":    {"Cl": 1},        # halide in input, treated as elemental Cl
    "CR2O3": {"Cr": 2, "O": 3},
    "CS2O":  {"Cs": 2, "O": 1},
    "F":     {"F": 1},         # halide in input, treated as elemental F
    "FE2O3": {"Fe": 2, "O": 3},
    "HFO2":  {"Hf": 1, "O": 2},
    "K2O":   {"K":  2, "O": 1},
    "LA2O3": {"La": 2, "O": 3},
    "LI2O":  {"Li": 2, "O": 1},
    "MGO":   {"Mg": 1, "O": 1},
    "MNO2":  {"Mn": 1, "O": 2},
    "MOO3":  {"Mo": 1, "O": 3},
    "NA2O":  {"Na": 2, "O": 1},
    "NIO":   {"Ni": 1, "O": 1},
    "ND2O3": {"Nd": 2, "O": 3},
    "P2O5":  {"P":  2, "O": 5},
    "PR2O3": {"Pr": 2, "O": 3},
    "RUO2":  {"Ru": 1, "O": 2},
    "SO3":   {"S":  1, "O": 3},
    "SIO2":  {"Si": 1, "O": 2},
    "SRO":   {"Sr": 1, "O": 1},
    "SNO2":  {"Sn": 1, "O": 2},
    "TEO2":  {"Te": 1, "O": 2},
    "TIO2":  {"Ti": 1, "O": 2},
    "V2O5":  {"V":  2, "O": 5},
    "Y2O3":  {"Y":  2, "O": 3},
    "ZNO":   {"Zn": 1, "O": 1},
    "ZRO2":  {"Zr": 1, "O": 2},
}

# --- Optional aliases for common variations (add more if your file uses them) ---
ALIASES = {
    "AL2O3":"AL2O3","B2O3":"B2O3","BAO":"BAO","CAO":"CAO","CE2O3":"CE2O3",
    "CL":"CL","CR2O3":"CR2O3","CS2O":"CS2O","F":"F","FE2O3":"FE2O3","HFO2":"HFO2",
    "K2O":"K2O","LA2O3":"LA2O3","LI2O":"LI2O","MGO":"MGO","MNO2":"MNO2","MOO3":"MOO3",
    "NA2O":"NA2O","NIO":"NIO","ND2O3":"ND2O3","P2O5":"P2O5","PR2O3":"PR2O3","RUO2":"RUO2",
    "SO3":"SO3","SIO2":"SIO2","SRO":"SRO","SNO2":"SNO2","TEO2":"TEO2","TIO2":"TIO2",
    "V2O5":"V2O5","Y2O3":"Y2O3","ZNO":"ZNO","ZRO2":"ZRO2",
    # common misspellings / alternate casings map to normalized keys above:
    "HF02":"HFO2", "HF O2":"HFO2", "HAFNIA":"HFO2",   # examples
    "ZN O":"ZNO",  "ZR O2":"ZRO2",
}

def normalize_key(s: str) -> str:
    """Uppercase, remove spaces/underscores to match our dictionaries."""
    return s.upper().replace(" ", "").replace("_", "")

# --- Load data ---
df = pd.read_csv(INPUT_CSV)

# Build a mapping from normalized -> original column name
norm_to_orig = {normalize_key(c): c for c in df.columns}

# Identify which columns are oxide/halide composition columns (present in STOICH/ALIASES)
comp_cols_norm = []
for nkey, orig in norm_to_orig.items():
    std = ALIASES.get(nkey, nkey)
    if std in STOICH:
        comp_cols_norm.append(nkey)

# Compute elemental moles per row (proportional from mol% of oxides)
element_moles = defaultdict(lambda: pd.Series(0.0, index=df.index))

for nkey in comp_cols_norm:
    orig_col = norm_to_orig[nkey]
    std = ALIASES.get(nkey, nkey)
    oxide_series = pd.to_numeric(df[orig_col], errors="coerce").fillna(0.0)  # mol%
    for elem, coeff in STOICH[std].items():
        element_moles[elem] = element_moles[elem] + oxide_series * coeff

# Normalize to mol% of elements (sum across all elements = 100)
# If a row has zero total element moles (unlikely), leave NaN to highlight data issue.
total_elem = sum(element_moles.values())
element_molpct = {}
for elem, moles in element_moles.items():
    with pd.option_context('mode.use_inf_as_na', True):
        element_molpct[f"{elem}"] = (moles / total_elem) * 100.0

elem_df = pd.DataFrame(element_molpct)

# Optional: a quick quality check that rows sum ~ 100
elem_df["CHECK_SUM_mol%"] = elem_df.filter(like="_mol%").sum(axis=1)

# Keep original non-composition columns + append element columns
comp_orig_cols = [norm_to_orig[n] for n in comp_cols_norm]
metadata_cols = [c for c in df.columns if c not in comp_orig_cols]
out = pd.concat([df[metadata_cols], elem_df], axis=1)

# Save to Excel
out.to_excel(OUTPUT_XLSX, index=False)
print(f"✅ Done. Saved elemental mol% (including O) to: {OUTPUT_XLSX}")
