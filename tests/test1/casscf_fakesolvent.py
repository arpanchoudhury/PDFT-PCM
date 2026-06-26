from pyscf import gto, lib, scf, mcpdft, mcscf
from pyscf import solvent
from pyscf.mcscf import avas
import numpy
from pyscf.solvent.smd import smd_radii
from pyscf.tools import molden




solute_xyz = 'methanal_sa2casscf_s0opt_gas.xyz'
solute_ao_labels = ['0 H 1s',  '1 C 2s', '1 C 2px', '1 C 2py', '1 C 2pz',  '2 H 1s', '3 O 2s', '3 O 2px', '3 O 2py', '3 O 2pz']


# solvent related ----
pcm_method = 'IEF-PCM'
solvent_eps =  1.0 # epsilon=1.0 means vacuum
solvent_refidx =  1.0 #refractive index = 1.0 means vacuum
solvent_alpha =  0.0 # Abraham’s hydrogen bond acidity parameter (0.0 means vacuum)
es_method = 'NONEQ'  # nonequilibrium solvation
phi_state = 1 # root used to obtain fast charges;
# (slow charges are always obtained from ground state in NONEQ)
# -------------------


radii_table = smd_radii(solvent_alpha)

mol = gto.M(
    atom=solute_xyz,
    basis ='aug-ccpvtz', 
    symmetry =False,
    verbose = 4,
)

mf = solvent.PCM(scf.RHF(mol))
mf.with_solvent.eps = solvent_eps
mf.with_solvent.method = pcm_method
mf.with_solvent.radii_table = radii_table
mf.kernel()



ncas, nelec, orbs = avas.avas(mf, solute_ao_labels)

# ------------------- SA-CASSCF --------------------
mc = mcscf.CASSCF(mf, ncas, nelec).state_average(numpy.ones(2)/2)

mc.mo_coeff = orbs
mc.fix_spin_(ss=0) # often necessary!

mc = solvent.PCM(mc)
mc.with_solvent.method = pcm_method
mc.with_solvent.eps = solvent_eps
mc.with_solvent.refidx = solvent_refidx
mc.with_solvent.radii_table = radii_table
mc.with_solvent.es_method = es_method
mc.with_solvent.phi_state = phi_state
mc.with_solvent.partition = 'pekar'
mc.kernel()



