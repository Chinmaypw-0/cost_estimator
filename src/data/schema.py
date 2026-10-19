HOURS_PER_PM = 152                 # person-month -> person-hours (COCOMO convention)
ALBRECHT_HOURS_PER_UNIT = 1000     # VERIFY from the ARFF header (thousands of hours?)

UNIFIED_COLS = [
    "dataset", "domain_group", "size_value", "size_unit",
    "duration_months", "team_experience", "effort_hours",
]

# Coarse labels based on dataset descriptions; edit if you know better.
DOMAIN_GROUP = {
    "albrecht": "business_it",
    "desharnais": "business_it",
    "kemerer": "business_it",
    "maxwell": "business_it",
    "nasa93": "aerospace_engineering",
    "cocomo811": "mixed_engineering",
}

# Standard COCOMO81 effort multiplier values
MULTIPLIERS = {
    "rely": {"vl": .75, "l": .88, "n": 1.0, "h": 1.15, "vh": 1.40},
    "data": {"l": .94, "n": 1.0, "h": 1.08, "vh": 1.16},
    "cplx": {"vl": .70, "l": .85, "n": 1.0, "h": 1.15, "vh": 1.30, "xh": 1.65},
    "time": {"n": 1.0, "h": 1.11, "vh": 1.30, "xh": 1.66},
    "stor": {"n": 1.0, "h": 1.06, "vh": 1.21, "xh": 1.56},
    "virt": {"l": .87, "n": 1.0, "h": 1.15, "vh": 1.30},
    "turn": {"l": .87, "n": 1.0, "h": 1.07, "vh": 1.15},
    "acap": {"vl": 1.46, "l": 1.19, "n": 1.0, "h": .86, "vh": .71},
    "aexp": {"vl": 1.29, "l": 1.13, "n": 1.0, "h": .91, "vh": .82},
    "pcap": {"vl": 1.42, "l": 1.17, "n": 1.0, "h": .86, "vh": .70},
    "vexp": {"vl": 1.21, "l": 1.10, "n": 1.0, "h": .90},
    "lexp": {"vl": 1.14, "l": 1.07, "n": 1.0, "h": .95},
    "modp": {"vl": 1.24, "l": 1.10, "n": 1.0, "h": .91, "vh": .82},
    "tool": {"vl": 1.24, "l": 1.10, "n": 1.0, "h": .91, "vh": .83},
    "sced": {"vl": 1.23, "l": 1.08, "n": 1.0, "h": 1.04, "vh": 1.10},
}