# Dictionary of atomic masses (AMU)
ATOMIC_MASSES = {
    "H1": 1.007825,
    "D2": 2.014101,
    "C12": 12.0,
    "C13": 13.003355,
    "O16": 15.994915,
    "O17": 16.999132,
    "O18": 17.999161,
    "S32": 31.972071,
    "S34": 33.967867,
}

# Definitions of isotopologues.
# Format: { ISO_CODE: [Atom1, CenterAtom, Atom2] }
ISOTOPOLOGUE_DEFINITIONS = {
    # --- H2O (1xx) ---
    # Standard notation 161 = H(1)-O(16)-H(1)
    161: ["H1", "O16", "H1"],
    162: ["H1", "O16", "D2"],
    171: ["H1", "O17", "H1"],
    181: ["H1", "O18", "H1"],
    262: ["D2", "O16", "D2"],
    # --- CO (2x, 3x) ---
    26: ["C12", "O16"],
    27: ["C12", "O17"],
    28: ["C12", "O18"],
    36: ["C13", "O16"],
    37: ["C13", "O17"],
    38: ["C13", "O18"],
    # --- CO2 (6xx, 7xx, 8xx) ---
    626: ["O16", "C12", "O16"],
    627: ["O16", "C12", "O17"],
    628: ["O16", "C12", "O18"],
    636: ["O16", "C13", "O16"],
    828: ["O18", "C12", "O18"],
}


def get_mass_list(iso_code):
    """Returns list of numerical masses [m1, m2, m3] for a given iso code."""
    if iso_code not in ISOTOPOLOGUE_DEFINITIONS:
        raise ValueError(
            f"Isotopologue code {iso_code} not defined in isotopologue_masses.py"
        )

    atom_keys = ISOTOPOLOGUE_DEFINITIONS[iso_code]
    return [ATOMIC_MASSES[key] for key in atom_keys]
