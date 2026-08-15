from pyscf import gto, lib, scf, mcpdft, mcscf
from pyscf import solvent
from pyscf.mcscf import avas
import numpy
from pyscf.solvent.smd import smd_radii
from NWChemBasisLibrary import  jul_ccpvtz
from pyscf.tools import molden
from pyscf.tools.molden import load


chkfilename = 'mcp_mp2-Opt_pentane-phistate.chk'
solute_xyz ='mcp_mp2-Opt_pentane.xyz'


ao_labels =['0 C 2pz', '1 C 2pz', '4 C 2pz', '5 C 2pz']


pcm_method = 'IEF-PCM'
solvent_eps = 1.8371 
solvent_refidx = 1.3575 
solvent_alpha =  0.0 #Abraham’s hydrogen bond acidity parameter for solvent


phi_state = 1
weights = numpy.ones(2)/2
my_otxc = 'tPBE'


radii_table = smd_radii(solvent_alpha)

mol = gto.M(
    atom=solute_xyz,
    basis = jul_ccpvtz, 
    symmetry =False,
    verbose = 4,
)

mf = solvent.PCM(scf.RHF(mol))
mf.with_solvent.eps = solvent_eps
mf.with_solvent.method = pcm_method
mf.with_solvent.radii_table = radii_table
mf.kernel()


ncas, nelec, orbs = avas.avas(mf, ao_labels)



#  -------------------- SA-MCPDFT  --------------------
mc = mcpdft.CASSCF (mf, my_otxc, ncas, nelec, grids_level=6)
mc.mo_coeff = orbs
mc.fix_spin_(ss=0) # often necessary!
mc.chkfile=chkfilename

mc_sa = mc.state_average(weights)
mc_sa = solvent.PCM(mc_sa)
mc_sa.with_solvent.method = pcm_method
mc_sa.with_solvent.eps = solvent_eps
mc_sa.with_solvent.refidx = solvent_refidx
mc_sa.with_solvent.radii_table = radii_table
mc_sa.with_solvent.equilibrium_solvation = False
mc_sa.with_solvent.rfroot = phi_state
mc_sa.kernel()


