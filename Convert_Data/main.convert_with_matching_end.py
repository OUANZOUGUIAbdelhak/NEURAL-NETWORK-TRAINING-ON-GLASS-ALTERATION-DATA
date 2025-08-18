import re
import pandas as pd
from collections import defaultdict

# -------------------- CONFIG --------------------
INPUT_CSV = "/home/intra.cea.fr/ao280403/Bureau/documents_bd_finale/Convert Composition/Data_From_Faijan_Oxides.csv"             # your oxide-mol% file
OUTPUT_XLSX = "elements_for_database.xlsx"

TARGET_COLUMNS = [
    "Li","B","O","Na","Mg","Al","Si","P","K","Ca","Ti","Cr","Mn","Fe","Ni","Zn",
    "Rb","Sr","Y","Zr","Mo","Ru","Te","I","Cs","Ba","La","Hf","Ce","Pr","Nd",
    "S_autres_TR","Température","ph","Ln(V0)"
]

TARGET_ELEMENTS = [
    "Li","B","O","Na","Mg","Al","Si","P","K","Ca","Ti","Cr","Mn","Fe","Ni","Zn",
    "Rb","Sr","Y","Zr","Mo","Ru","Te","I","Cs","Ba","La","Hf","Ce","Pr","Nd"
]

STOICH = {
    "AL2O3": {"Al": 2, "O": 3},
    "B2O3":  {"B":  2, "O": 3},
    "BAO":   {"Ba": 1, "O": 1},
    "CAO":   {"Ca": 1, "O": 1},
    "CE2O3": {"Ce": 2, "O": 3},
    "CL":    {"Cl": 1},      # not in DB schema; ignored by normalization
    "CR2O3": {"Cr": 2, "O": 3},
    "CS2O":  {"Cs": 2, "O": 1},
    "F":     {"F": 1},       # not in DB schema; ignored by normalization
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
    "SNO2":  {"Sn": 1, "O": 2},    # Sn not in DB schema; ignored
    "SRO":   {"Sr": 1, "O": 1},
    "TIO2":  {"Ti": 1, "O": 2},
    "TEO2":  {"Te": 1, "O": 2},
    "V2O5":  {"V":  2, "O": 5},    # V not in DB schema; ignored
    "Y2O3":  {"Y":  2, "O": 3},
    "ZNO":   {"Zn": 1, "O": 1},
    "ZRO2":  {"Zr": 1, "O": 2},
    "RB2O":  {"Rb": 2, "O": 1},    # if Rb2O appears
}

ALIASES = {
    "FEMON": "FE2O3",
    "HF O2": "HFO2", "HF02": "HFO2",
    "SI O2": "SIO2", "ZN O": "ZNO", "ZR O2": "ZRO2",
}

# Normalize any column header to an alphanumeric uppercase token
def normalize_key(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", str(s)).upper()

# -------------------- LOAD --------------------
df = pd.read_csv(INPUT_CSV)
norm_to_orig = {normalize_key(c): c for c in df.columns}

# -------------------- ELEMENT CONVERSION --------------------
present_oxides = []
for nkey, orig in norm_to_orig.items():
    std = ALIASES.get(nkey, nkey)
    if std in STOICH:
        present_oxides.append((std, orig))

zero = pd.Series(0.0, index=df.index)
element_moles = defaultdict(lambda: zero.copy())

for std, orig_col in present_oxides:
    series = pd.to_numeric(df[orig_col], errors="coerce").fillna(0.0)
    for elem, coeff in STOICH[std].items():
        element_moles[elem] = element_moles[elem] + series * coeff

# Normalize ONLY over elements in your DB schema so they sum to 100
total_selected = zero.copy()
for e in TARGET_ELEMENTS:
    total_selected = total_selected + element_moles.get(e, zero)

def pct(numer, denom):
    return numer.mul(100).div(denom.where(denom != 0, pd.NA)).fillna(0.0)

elem_pct = {e: pct(element_moles.get(e, zero), total_selected) for e in TARGET_ELEMENTS}
elem_df = pd.DataFrame(elem_pct).astype(float)

# Add S_autres_TR (0 unless you want to compute something specific)
elem_df["S_autres_TR"] = 0.0

# -------------------- METADATA COPY (robust) --------------------
out = pd.DataFrame(index=df.index)
out[TARGET_ELEMENTS] = elem_df[TARGET_ELEMENTS]
out["S_autres_TR"] = elem_df["S_autres_TR"]

# Température
for cand in ("TEMPRATURE", "TEMPERATURE"):
    if cand in norm_to_orig:
        out["Température"] = df[norm_to_orig[cand]]
        break
else:
    out["Température"] = pd.NA

# pH
if "PH" in norm_to_orig:
    out["ph"] = df[norm_to_orig["PH"]]
else:
    out["ph"] = pd.NA

# Ln(V0): copy from any of these if present, in this priority
ln_candidates = ("LNRATE", "LNV0")
for cand in ln_candidates:
    if cand in norm_to_orig:
        out["Ln(V0)"] = df[norm_to_orig[cand]]
        break
else:
    out["Ln(V0)"] = pd.NA  # (leave blank if none found)

# -------------------- ORDER & SAVE --------------------
out = out[TARGET_COLUMNS]
out.to_excel(OUTPUT_XLSX, index=False)
print(f"✅ Saved: {OUTPUT_XLSX}")
