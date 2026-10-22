"""patch_hsig.py -- give the MLP-based families the same hidden-scale knob.

DeLP's forward is out(kwta(t * sigmoid(d))) with sigmoid(d) ~ 0.576, while the
MLP families are out(kwta(sigmoid(t))) -- a different, also arbitrary, operating
point. Tuning one and not the other makes "DeLP gains 8-10 points from alpha"
meaningless: the MLP families have an equally free gain in their sigmoid.

HSIG is that gain, defaulting to 1.0 so every earlier result stays bit-identical.
"""
import ast
import sys

p = "bench_cl.py"
s = open(p).read()

old = "HGAIN = 1.0"
new = (
    "HGAIN = 1.0\n"
    "# Sigmoid gain for the MLP-based families. Their hidden layer is\n"
    "# kwta(sigmoid(t)); sigmoid's operating point is as arbitrary as the gate\n"
    "# scale above. 1.0 reproduces every earlier result bit-for-bit.\n"
    "HSIG = 1.0"
)
assert old in s, "HGAIN anchor missing"
s = s.replace(old, new, 1)

old2 = "        h = kwta(torch.sigmoid(self.ff(x)), self.k)"
new2 = (
    "        t = self.ff(x)\n"
    "        h = kwta(torch.sigmoid(t) if HSIG == 1.0 else torch.sigmoid(HSIG * t), self.k)"
)
assert old2 in s, "MLP forward anchor missing"
s = s.replace(old2, new2, 1)

old3 = '                    help="hidden-layer gain for the dendritic arms; 1.0 = original")'
new3 = (
    '                    help="hidden-layer gain for the dendritic arms; 1.0 = original")\n'
    '    ap.add_argument("--hsig", type=float, default=1.0,\n'
    '                    help="sigmoid gain for the MLP families; 1.0 = original")'
)
assert old3 in s, "hgain argparse anchor missing"
s = s.replace(old3, new3, 1)

old4 = "    global HGAIN\n    HGAIN = float(a.hgain)"
new4 = "    global HGAIN, HSIG\n    HGAIN = float(a.hgain)\n    HSIG = float(a.hsig)"
assert old4 in s, "global anchor missing"
s = s.replace(old4, new4, 1)

old5 = '    res["hgain"] = HGAIN'
new5 = '    res["hgain"] = HGAIN\n    res["hsig"] = HSIG'
assert old5 in s, "record anchor missing"
s = s.replace(old5, new5, 1)

open(p, "w").write(s)
ast.parse(s)
print("patched + syntax_ok")
