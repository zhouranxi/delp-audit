"""Is a uniform hidden gain a capacity knob or a pure re-parameterisation?

kwta(h, k) = h * (h >= kth_largest(h)).  If that is scale-equivariant, then
kwta(alpha*t) == alpha*kwta(t) for any alpha > 0, so a network with gain alpha
is functionally identical to the same network with gain 1 and a readout whose
weights are scaled by alpha.  Under that reading the 10-13 point gap in the
ladder is NOT extra capacity -- it is an optimisation / regularisation scale
effect, and DeLP's gate sitting at sigma(d)=0.58 is a fixable mis-scaling
rather than a structural cost.

This script settles it numerically.  Run from ~/dendritic-lab.
"""
import torch
from exp4_branch_protection import kwta

torch.manual_seed(0)
for shape in [(8, 256), (1, 256), (64, 256)]:
    h = torch.randn(*shape)
    for a in (0.5, 2.0, 4.0, 16.0):
        lhs = kwta(a * h, 64)
        rhs = a * kwta(h, 64)
        print("shape=%-10s alpha=%-5s max|diff|=%.3e  mask_equal=%s"
              % (str(shape), a, (lhs - rhs).abs().max().item(),
                 bool(((lhs != 0) == (rhs != 0)).all())))
