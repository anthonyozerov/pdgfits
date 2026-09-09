import re
import numpy as np
import pandas as pd


# Parser for measurement strings as written in the internal PDG database.
# Written by AI and probably fragile.
# Doesn't handle some branching ratios and special cases which I don't understand.
def parse_measurement(s):
    if "@" in s:
        # print(s)
        s = s.split('@')[0]
        # return (np.nan, np.nan, np.nan)
    s = s.strip()

    # get the trailing exponent, and the corresponding multiplicative
    # 'scale' which we will apply to all values
    exp_match = re.search(r"[Ee]([+-]?\d+)$", s)
    scale = 1.0
    if exp_match:
        scale = 10 ** int(exp_match.group(1))
        s = s[: exp_match.start()]
    # strip opening and closing parentheses
    s = s.strip('()')

    # tokenise the string
 
    # Each error token is preceded by one or two sign characters, possibly with a space between the sign(s) and number.
    # Regex for one token: (sign-chars)[optional space](unsigned number)
    #   sign-chars: one of  +  |  -  |  +-  |  -+
    #   unsigned number: digits with optional decimal point
    token_re = re.compile(r"([+-](?:\s*[+-])?)\s*(\d+(?:\.\d+)?)")

    # Central value: everything before the first token_re match that
    # does NOT start at position 0.
    first = token_re.search(s, pos=1)
    if first:
        value = float(s[: first.start()].strip())
        rest = s[first.start() :]
    else:
        value = float(s)
        rest = ""

    pos_errors: list[float] = []
    neg_errors: list[float] = []

    for signs, mag in token_re.findall(rest):
        x = float(mag)
        if "+" in signs:
            pos_errors.append(x)
        if "-" in signs:
            neg_errors.append(x)

    # combine stat and syst in quadrature
    def quad(errs: list[float]) -> float:
        return np.sqrt(sum(e**2 for e in errs)) if errs else 0.0
    
    last_err = scale*(pos_errors[-1] + neg_errors[-1]) / 2 if pos_errors and neg_errors else 0.0

    # return all values scaled by the magnitude
    return (value * scale, quad(pos_errors) * scale, quad(neg_errors) * scale, last_err)

# function to format a measurement into a string
def measurement_string(meas, error_p, error_n):
    return f'{meas}+{error_p}-{error_n}'


# ---------------------------------------------------------------------------
# Parsers for the structured measurement-annotation strings used in the PDG DB.
#
# These strings are comma-separated records whose fields are space-separated
# tokens. A "node key" is the dotted form of a space-separated
# '<par_code> <param>' field, e.g. 'M001 8' -> 'M001.8'.
# ---------------------------------------------------------------------------

def _node_key(field):
    """Turn a space-separated '<par_code> <param>' field into a 'par_code.param' key."""
    return '.'.join(field.strip().split(' '))


# Format: 'br_adjust: <meas>; <rel>, <adjust_meas>, <par_code> <param>; ...'
# Returns the node key of the (last) adjustment target, or None.
def br_adjust_node(measurement):
    if 'br_adjust' in measurement:
        return _node_key(measurement.split(',')[-1])
    else:
        return None


# Format: 'dep_meas: <value>, <slope> <const> <node>, <slope> <const> <node>, ...'
# Returns (nodes, slopes, constants) for the dependent measurements, or None.
def get_dep_meas_data(measurement):
    if 'dep_meas' in measurement:
        records = [r.strip().split(' ') for r in measurement.split(',')[1:]]
        nodes = [r[2] for r in records]
        slopes = [float(r[0]) for r in records]
        constants = [float(r[1]) for r in records]
        return (nodes, slopes, constants)
    else:
        return None


# Each adjustment record (as produced by unadjust_measurement) has the form
# '<rel>, ADJUST, <par_code> <param>'. Returns [node_keys, rels], or None.
def get_adjust_data(adjustments):
    if isinstance(adjustments, list):
        nodes = [_node_key(data.split(',')[2]) for data in adjustments]
        rels = [data.split(',')[0].strip() for data in adjustments]
        assert all(rel in ['/', '*'] for rel in rels), f"Unknown relationships: {rels}"
        return [nodes, rels]
    else:
        return None

# function to get the scale for a measurement
# TODO: DOES THIS HANDLE ALL CASES???
def get_scale(text):
    scale = 1
    if text == 'keV':
        scale *= 1e-3
    elif text == 'eV':
        scale *= 1e-6
    return scale

# function to make a key for a parameter
def parameter_key(par_code, parameter):
    if pd.isna(par_code):
        return parameter
    else:
        return par_code+'.'+parameter