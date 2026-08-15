from pyscf import gto, lib, scf, mcpdft, mcscf
from pyscf.mcscf import avas
import numpy
from pyscf.tools import molden
from pyscf.tools.molden import load


solute_xyz ='mcp_mp2-Opt_pentane.xyz'


ao_labels =['0 C 2pz', '1 C 2pz', '4 C 2pz', '5 C 2pz']



weights = numpy.ones(2)/2
my_otxc = 'tPBE'



mol = gto.M(
    atom=solute_xyz,
    basis = 'aug-cc-pvtz', 
    symmetry =False,
    verbose = 4,
)

mf = scf.RHF(mol)
mf.kernel()


ncas, nelec, orbs = avas.avas(mf, ao_labels)



#  -------------------- SA-MCPDFT  --------------------
mc = mcpdft.CASSCF (mf, my_otxc, ncas, nelec, grids_level=6)
mc.mo_coeff = orbs
mc.fix_spin_(ss=0) # often necessary!

mc_sa = mc.state_average(weights)
mc_sa.kernel()


