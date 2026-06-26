import numpy 
from pyscf import gto, scf, mcscf, mcpdft, lib
from pyscf.mcscf import avas
from pyscf.tools.molden import load



xyz = 'methanal_sa2casscf_s0opt_gas.xyz'


mol = gto.M(
    atom=xyz,
    basis = 'aug-ccpvtz',
    symmetry =False,
    verbose = 4,
)


mf = scf.RHF(mol)
mf.kernel()

ao_labels = ['0 H 1s',  '1 C 2s', '1 C 2px', '1 C 2py', '1 C 2pz',  '2 H 1s', '3 O 2s', '3 O 2px', '3 O 2py', '3 O 2pz']
ncas, nelec, orbs = avas.avas(mf, ao_labels)


mc = mcscf.CASSCF(mf, ncas, nelec).state_average(numpy.ones(2)/2)
mc.mo_coeff = orbs

mc.fix_spin_(ss=0) # often necessary!
mc.kernel()

