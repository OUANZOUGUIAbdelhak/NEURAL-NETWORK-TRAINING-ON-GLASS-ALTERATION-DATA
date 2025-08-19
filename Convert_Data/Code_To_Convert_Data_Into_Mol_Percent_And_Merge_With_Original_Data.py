import re
import pandas as pd
from collections import defaultdict

# -------------------- PATHS (edit these) --------------------
OXIDE_INPUT_CSV   = "/home/intra.cea.fr/ao280403/Bureau/ML Model/NEURAL-NETWORK-TRAINING-ON-GLASS-ALTERATION-DATA/Convert_Data/Data_From_Faijan_Oxides.csv"          # your new oxide-mol% file to convert & append
ORIGINAL_DB_CSV   = "/home/intra.cea.fr/ao280403/Bureau/ML Model/NEURAL-NETWORK-TRAINING-ON-GLASS-ALTERATION-DATA/Convert_Data/Data Si ph LnV0.csv"    # your existing DB CSV (with target columns)
OUTPUT_MERGED_CSV = "merged_database.csv"      # final combined CSV
OUTPUT_MERGED_EXCEL = "merged_database.xlsx"  # final combined Excel file

# Drop the first *data* row of the newly converted block before merging
DROP_FIRST_ROW_OF_NEW = True

# -------------------- DB schema (order preserved) --------------------
TARGET_COLUMNS = [
    "Li","B","O","Na","Mg","Al","Si","P","K","Ca","Ti","Cr","Mn","Fe","Ni","Zn",
    "Rb","Sr","Y","Zr","Mo","Ru","Te","I","Cs","Ba","La","Hf","Ce","Pr","Nd",
    "S_autres_TR","Température","ph","Ln(V0)"
]
TARGET_ELEMENTS = [
    "Li","B","O","Na","Mg","Al","Si","P","K","Ca","Ti","Cr","Mn","Fe","Ni","Zn",
    "Rb","Sr","Y","Zr","Mo","Ru","Te","I","Cs","Ba","La","Hf","Ce","Pr","Nd"
]

# -------------------- Stoichiometry (moles of element per mole of oxide/halide) --------------------
STOICH = {
    "AL2O3": {"Al": 2, "O": 3},
    "B2O3":  {"B":  2, "O": 3},
    "BAO":   {"Ba": 1, "O": 1},
    "CAO":   {"Ca": 1, "O": 1},
    "CE2O3": {"Ce": 2, "O": 3},
    "CL":    {"Cl": 1},     # not in DB schema -> ignored during normalization
    "CR2O3": {"Cr": 2, "O": 3},
    "CS2O":  {"Cs": 2, "O": 1},
    "F":     {"F": 1},      # not in DB schema -> ignored during normalization
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
    "SIO2":  {"Si": 1, "O": 2},
    "SNO2":  {"Sn": 1, "O": 2},  # Sn not in DB schema -> ignored
    "SRO":   {"Sr": 1, "O": 1},
    "TIO2":  {"Ti": 1, "O": 2},
    "TEO2":  {"Te": 1, "O": 2},
    "V2O5":  {"V":  2, "O": 5},  # V not in DB schema -> ignored
    "Y2O3":  {"Y":  2, "O": 3},
    "ZNO":   {"Zn": 1, "O": 1},
    "ZRO2":  {"Zr": 1, "O": 2},
    "RB2O":  {"Rb": 2, "O": 1},
    "I":     {"I": 1},           # allow elemental I if present as halide column
}
ALIASES = {
    "FEMON": "FE2O3",
    "HF O2": "HFO2", "HF02": "HFO2",
    "SI O2": "SIO2", "ZN O": "ZNO", "ZR O2": "ZRO2",
}

# -------------------- Helpers --------------------
def normalize_key(s: str) -> str:
    """Uppercase + strip non-alphanumerics for robust header matching."""
    return re.sub(r"[^A-Za-z0-9]", "", str(s)).upper()

def pct(numer: pd.Series, denom: pd.Series) -> pd.Series:
    return numer.mul(100).div(denom.where(denom != 0, pd.NA)).fillna(0.0)

# -------------------- 1) Load & convert the OXIDE file to elemental mol% --------------------
df_ox = pd.read_csv(OXIDE_INPUT_CSV)
norm_to_orig = {normalize_key(c): c for c in df_ox.columns}

# Which oxide/halide composition columns are present?
present_oxides = []
for nkey, orig in norm_to_orig.items():
    std = ALIASES.get(nkey, nkey)
    if std in STOICH:
        present_oxides.append((std, orig))

# Elemental moles (proportional from mol%)
zero = pd.Series(0.0, index=df_ox.index)
element_moles = defaultdict(lambda: zero.copy())
for std, col in present_oxides:
    series = pd.to_numeric(df_ox[col], errors="coerce").fillna(0.0)
    for elem, coeff in STOICH[std].items():
        element_moles[elem] = element_moles[elem] + series * coeff

# Normalize ONLY over elements in your DB schema so they sum to 100
total_selected = zero.copy()
for e in TARGET_ELEMENTS:
    total_selected = total_selected + element_moles.get(e, zero)
elem_pct = {e: pct(element_moles.get(e, zero), total_selected) for e in TARGET_ELEMENTS}
converted = pd.DataFrame(elem_pct).astype(float)

# S_autres_TR (set to 0 unless you compute something specific)
converted["S_autres_TR"] = 0.0

# Metadata: Température, ph, Ln(V0)
# Température from Temprature/Temperature if present
if "TEMPRATURE" in norm_to_orig:
    converted["Température"] = df_ox[norm_to_orig["TEMPRATURE"]]
elif "TEMPERATURE" in norm_to_orig:
    converted["Température"] = df_ox[norm_to_orig["TEMPERATURE"]]
else:
    converted["Température"] = pd.NA

# ph from pH/ph
if "PH" in norm_to_orig:
    converted["ph"] = df_ox[norm_to_orig["PH"]]
else:
    converted["ph"] = pd.NA

# Ln(V0) from ln rate / ln_rate / Ln(V0)
if "LNRATE" in norm_to_orig:
    converted["Ln(V0)"] = df_ox[norm_to_orig["LNRATE"]]
elif "LNV0" in norm_to_orig:
    converted["Ln(V0)"] = df_ox[norm_to_orig["LNV0"]]
else:
    converted["Ln(V0)"] = pd.NA

# Ensure exact column order/coverage (fill missing with 0 or NA accordingly)
for col in TARGET_COLUMNS:
    if col not in converted.columns:
        # elements -> 0; metadata -> NA
        converted[col] = 0.0 if col in TARGET_ELEMENTS + ["S_autres_TR"] else pd.NA
converted = converted[TARGET_COLUMNS]

# >>> Drop first row of *newly converted* block if requested <<<
if DROP_FIRST_ROW_OF_NEW and len(converted) > 0:
    converted = converted.iloc[1:].reset_index(drop=True)

# -------------------- 2) Load the ORIGINAL database CSV --------------------
db = pd.read_csv(ORIGINAL_DB_CSV)

# If DB is missing any columns, add them so we can align (filled with 0 or NA)
for col in TARGET_COLUMNS:
    if col not in db.columns:
        db[col] = 0.0 if col in TARGET_ELEMENTS + ["S_autres_TR"] else pd.NA

# Drop any extra columns not in schema, then order columns
db = db[TARGET_COLUMNS]

# -------------------- 3) Merge & save --------------------
merged = pd.concat([db, converted], ignore_index=True, sort=False)
merged = merged[TARGET_COLUMNS]  # enforce final order

# Save as CSV (UTF-8)
merged.to_csv(OUTPUT_MERGED_CSV, index=False)

# Save as Excel
merged.to_excel(OUTPUT_MERGED_EXCEL, index=False, engine="openpyxl")

print(f"✅ Merged CSV saved to: {OUTPUT_MERGED_CSV}")
print(f"✅ Merged Excel saved to: {OUTPUT_MERGED_EXCEL}")
