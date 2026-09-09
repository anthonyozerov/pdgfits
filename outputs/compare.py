import re, sys

def parse(path):
    with open(path) as f:
        lines = f.read().splitlines()
    blocks, cur = [], []
    for ln in lines:
        if set(ln) == {'='} and len(ln) >= 50:
            if cur: blocks.append(cur)
            cur = []
        else:
            cur.append(ln)
    if cur: blocks.append(cur)
    fits = {}
    for b in blocks:
        if not b: continue
        label = b[0].strip()
        d = {'chi2': None, 'pdg': None, 'valid': None, 'hess': None,
             'success': None, 'msg': None}
        for ln in b:
            mm = re.search(r'chi2 obtained:\s*([0-9.Ee+-]+)', ln)
            if mm: d['chi2'] = float(mm.group(1))
            mm = re.search(r'chi2 obtained by PDG:\s*([0-9.Ee+-]+)', ln)
            if mm: d['pdg'] = float(mm.group(1))
            mm = re.search(r'fit valid:\s*(\w+).*hessian accurate:\s*(\w+)', ln)
            if mm: d['valid'], d['hess'] = mm.group(1), mm.group(2)
            mm = re.search(r'scipy success:\s*(\w+), message:\s*(.*)', ln)
            if mm: d['success'], d['msg'] = mm.group(1), mm.group(2).strip()
        if d['chi2'] is not None:
            fits[label] = d
    return fits

mn, sc = parse(sys.argv[1]), parse(sys.argv[2])
labels = list(dict.fromkeys(list(mn)+list(sc)))
print(f"minuit fits={len(mn)}  scipy fits={len(sc)}  union={len(labels)}")

REL = 1e-3  # relative chi2 threshold to flag
chi2_flags, valid_flags, scipy_fail = [], [], []
for L in labels:
    a, b = mn.get(L), sc.get(L)
    ca = a['chi2'] if a else None
    cb = b['chi2'] if b else None
    rel = None
    if ca is not None and cb is not None:
        rel = abs(ca-cb)/max(abs(ca), abs(cb), 1e-12)
        if rel > REL:
            chi2_flags.append((rel, L, ca, cb))
    if a and a['valid'] == 'False':
        valid_flags.append(('minuit', L, a))
    if b and b['success'] == 'False':
        scipy_fail.append((L, b))

print(f"\n=== scipy fits with success: False  ({len(scipy_fail)}) ===")
for L, b in scipy_fail:
    a = mn.get(L)
    ca = a['chi2'] if a else float('nan')
    rel = abs(ca-b['chi2'])/max(abs(ca), abs(b['chi2']), 1e-12)
    print(f"  {L:30} chi2 m={ca:.4g} s={b['chi2']:.4g} rel={rel:.1e}  | {b['msg'][:60]}")

print(f"\n=== minuit fits flagged invalid ({len(valid_flags)}) ===")
for who, L, a in valid_flags:
    print(f"  {L:30} valid={a['valid']} hess_accurate={a['hess']}")

print(f"\n=== chi2 differs > {REL:.0e} relative  ({len(chi2_flags)}) ===")
print(f"  {'label':30} {'minuit':>11} {'scipy':>11} {'rel.diff':>9}")
for rel, L, ca, cb in sorted(chi2_flags, reverse=True):
    note = '  (scipy success:False)' if sc.get(L) and sc[L]['success']=='False' else ''
    print(f"  {L:30} {ca:>11.5g} {cb:>11.5g} {rel:>9.1e}{note}")

# summary numbers
agree = sum(1 for L in labels if mn.get(L) and sc.get(L)
            and abs(mn[L]['chi2']-sc[L]['chi2'])/max(abs(mn[L]['chi2']),abs(sc[L]['chi2']),1e-12) <= REL)
print(f"\n=== summary ===")
print(f"chi2 agree (rel<= {REL:.0e}): {agree}/{len(labels)}")
print(f"chi2 differ:                 {len(chi2_flags)}")
print(f"scipy success:False:         {len(scipy_fail)}")
print(f"minuit invalid:              {len(valid_flags)}")
