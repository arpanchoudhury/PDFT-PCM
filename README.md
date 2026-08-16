# MC-PDFT & L-PDFT with the polarizable continuum model
 
A PySCF-based implementation of polarizable continuum model (PCM) solvation for
multiconfiguration pair-density functional theory (MC-PDFT) and linearized PDFT
(L-PDFT), together with data and example input files supporting the accompanying
manuscript.

 ![Alt text](<img width="2600" height="1256" alt="toc" src="https://github.com/user-attachments/assets/94886f91-024b-4dad-a617-92eeb689544c" />
)
 
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
 
mol = gto.M(atom='...', basis='aug-cc-pVTZ')
mf = solvent.PCM(scf.RHF(mol)).run()
 
# L-PDFT over 2 states
mc = mcpdft.CASSCF(mf, 'tPBE', ncas, nelecas)
mc.fix_spin_(ss=0) 

mc =  mc.multi_state([0.5, 0.5], "lin")
mc = solvent.PCM(mc)
# state-averaged nonequilibrium solvation:
# slow charges from state 0, fast charges from state rfroot
mc.with_solvent.equilibrium_solvation = False
mc.with_solvent.rfroot = 1 
mc.kernel()
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
ground-state density). `rfroot` selects the state whose density defines the total/fast
charges in an equilibrium/nonequilibrium state-averaged calculation.
 
## Repository contents
 
```
data/         checkpoint files for final results 
              MP2-optimized ground-state structures (gas phase and solvent)
              PySCF input scripts
              OpenMolcas input scripts for nonequilibrium XMS-CASPT2 calculations
pyscf/        required codes
test/         unit test
```
  
## Computational details
 
Settings used for the results in the paper:
 
- **Basis sets**: jul-cc-pVTZ for all molecules; cc-pVTZ for cytosine
- **Solvation**: IEF-PCM with SMD intrinsic Coulomb radii
- **On-top functional**: tPBE
- **Integration grid**: PySCF grid level 6
- **Geometries**: optimized at MP2 with Gaussian 16 (equilibrium solvation in solvent)
- **Other calculations**: XMS-CASPT2 with OpenMolcas v26.02 (IPEA shift 0.25,
  imaginary level shift 0.2i); nonequilibrium TDDFT/ptSS with Q-Chem v6.2
Active spaces for each system are listed in the manuscript.
 
## License
 
<!-- Match PySCF's license (Apache-2.0) if the code is intended for reuse. -->
 
## Contact
 
Arpan Choudhury — Department of Chemistry, University of Chicago.
(arpanchoudhury@uchicago.edu)
