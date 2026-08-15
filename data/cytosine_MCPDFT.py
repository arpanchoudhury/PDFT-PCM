from pyscf import gto, lib, scf, mcpdft, mcscf
from pyscf import solvent
from pyscf.mcscf import avas
import numpy
from pyscf.solvent.smd import smd_radii
from pyscf.tools import molden
from pyscf.tools.molden import load


chkfilename = 'cytosine_mp2-Opt_water-phistate2.chk'
solute_xyz ='cytosine_mp2-Opt_water.xyz'



ao_labels = ['0 N 2pz',  '1 C 2pz', '3 C 2pz', '5 C 2pz', '6 N 2p', '9 N 2p', '10 C 2pz', '11 O 2p'] 


ncas, nelec = 10,14

pcm_method = 'IEF-PCM'
solvent_eps =78.355 
solvent_refidx = 1.3328 
solvent_alpha =  0.82 


phi_state = 2
weights = numpy.ones(10)/10
my_otxc = 'tPBE'


radii_table = smd_radii(solvent_alpha)

mol = gto.M(
    atom=solute_xyz,
    basis = 'cc-pvtz', 
    symmetry =False,
    verbose = 4,
)

mf = solvent.PCM(scf.RHF(mol))
mf.with_solvent.eps = solvent_eps
mf.with_solvent.method = pcm_method
mf.with_solvent.radii_table = radii_table
mf.kernel()


_, _, orbs = avas.avas(mf, ao_labels)



#  -------------------- SA-MCPDFT  --------------------
mc2 = mcpdft.CASSCF (mf, my_otxc, ncas, nelec, grids_level=6)
mc2.mo_coeff = orbs
mc2.max_cycle_macro = 150
mc2.fix_spin_(ss=0) # often necessary!
mc2.chkfile=chkfilename

mc_sa = mc2.state_average(weights)
mc_sa = solvent.PCM(mc_sa)
mc_sa.with_solvent.method = pcm_method
mc_sa.with_solvent.eps = solvent_eps
mc_sa.with_solvent.refidx = solvent_refidx
mc_sa.with_solvent.radii_table = radii_table
mc_sa.with_solvent.equilibrium_solvation = False
mc_sa.with_solvent.rfroot = phi_state

mc_sa.kernel()


