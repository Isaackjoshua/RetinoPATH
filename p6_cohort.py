"""P6 — 3-class (R0/R1/R2) cohort helpers. Pure functions, unit-tested off-GPU."""


def filter_r0r2_patients(df, code_col='code', grade_col='grade_int'):
    """Drop every patient whose MAX grade > 2 (R3A); keep all rows of the rest."""
    pmax = df.groupby(code_col)[grade_col].max()
    keep = pmax[pmax <= 2].index
    return df[df[code_col].isin(keep)].copy()


def inverse_freq_weights(counts):
    """Majority-normalised inverse-frequency weights.

    counts: dict {class_index: n}. Returns a list ordered by sorted class index,
    with the majority (most frequent) class -> 1.0 and rarer classes -> max/n.
    """
    keys = sorted(counts)
    vals = [float(counts[k]) for k in keys]
    maxc = max(vals)
    return [maxc / v for v in vals]
