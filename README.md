# MC-PDFT & L-PDFT with the polarizable continuum model
 
A PySCF-based implementation of polarizable continuum model (PCM) solvation for
multiconfiguration pair-density functional theory (MC-PDFT) and linearized PDFT
(L-PDFT), together with the input files and data supporting the accompanying
manuscript.
 
## Paper
 
A. Choudhury, M. R. Hermes, D. G. Truhlar, L. Gagliardi,
*Vertical Excitation Energies in Solution via a Two-Time-Scale Solvation Model Coupled to
Pair-Density Functional Theory* (submitted).

 
## Code
 
The implementation lives in a fork of PySCF, on the `pdft-pcm` branch:
 
**https://github.com/arpanchoudhury/pyscf/tree/pdft-pcm**
 
This repository holds the geometries, some input example scripts, and `chk` files from PySCF calculations.
 
## Features
 
Solvation modes implemented for SA-CASSCF, MC-PDFT, and L-PDFT:
 
- state-specific equilibrium solvation
- state-specific nonequilibrium solvation
- state-averaged equilibrium solvation
- state-averaged nonequilibrium solvation

Nonequilibrium solvation uses Pekar's partition of the surface charges into fast
and slow components, with the slow charges obtained from the ground state and the
fast charges from a chosen state.
 

## Usage
 
Attach PCM to an MC-PDFT or L-PDFT object and select the solvation mode through
the `with_solvent` attributes:
 
```python
from pyscf import gto, scf, mcpdft, solvent
 
mol = gto.M(atom='...', basis='jul-cc-pVTZ')
mf = scf.RHF(mol).run()
 
# state-averaged L-PDFT over 2 states
mc = mcpdft.CASSCF(mf, 'tPBE', ncas, nelecas).state_average_([0.5, 0.5])
mc = solvent.PCM(mc)
 
# state-averaged nonequilibrium solvation:
# slow charges from state 0, fast charges from state rfroot
mc.with_solvent.state_id = None
mc.with_solvent.equilibrium_solvation = False
mc.with_solvent.rfroot = 1
 
mc.run()
```
 
### Selecting the solvation mode
 
The mode is determined by `state_id` and `equilibrium_solvation`:
 
| `state_id` | `equilibrium_solvation` | mode | additional attribute |
|---|---|---|---|
| `0` (default) | — | ground-state equilibrium | — |
| `int > 0` | `True` | state-specific equilibrium | — |
| `int > 0` | `False` | state-specific nonequilibrium | `refdm` (reference density) |
| `None` | `True` | state-averaged equilibrium | `rfroot` |
| `None` | `False` | state-averaged nonequilibrium | `rfroot` |
 
`refdm` is the density matrix defining the slow charges (typically the converged
ground-state density). `rfroot` selects the state whose density defines the fast
charges in a state-averaged calculation.
 
## Repository contents
 
```
inputs/       PySCF input scripts for each molecule and solvent
geometries/   MP2-optimized ground-state structures (gas phase and solvent)
outputs/      raw calculation logs
scripts/      analysis and figure-generation scripts
```
 
<!-- Adjust to match the actual layout. -->
 
## Computational details
 
Settings used for the results in the paper:
 
- **Basis sets**: jul-cc-pVTZ for all molecules; cc-pVTZ for cytosine
- **Solvation**: IEF-PCM with SMD intrinsic Coulomb radii
- **On-top functional**: tPBE
- **Integration grid**: PySCF grid level 6
- **Geometries**: optimized at MP2 with Gaussian 16 (equilibrium solvation in solvent)
- **Reference calculations**: XMS-CASPT2 with OpenMolcas v26.02 (IPEA shift 0.25,
  imaginary level shift 0.2i); nonequilibrium LR-TDDFT with Q-Chem v6.2
Active spaces for each system are listed in the manuscript.
 
## License
 
<!-- Match PySCF's license (Apache-2.0) if the code is intended for reuse. -->
 
## Contact
 
Arpan Choudhury — Department of Chemistry, University of Chicago.
