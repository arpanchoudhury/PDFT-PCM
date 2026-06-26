#!/usr/bin/env python
# Copyright 2014-2023 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Xiaojie Wu <wxj6000@gmail.com>
#

'''
PCM family solvent models
'''

import numpy
import scipy
from pyscf import lib
from pyscf.lib import logger
from pyscf import gto, df
from pyscf.dft import gen_grid
from pyscf.data import radii
from pyscf.solvent import ddcosmo
from pyscf.solvent import _attach_solvent
from scipy.special import erf


@lib.with_doc(_attach_solvent._for_scf.__doc__)
def pcm_for_scf(mf, solvent_obj=None, dm=None):
    if solvent_obj is None:
        solvent_obj = PCM(mf.mol)
    return _attach_solvent._for_scf(mf, solvent_obj, dm)

@lib.with_doc(_attach_solvent._for_casscf.__doc__)
def pcm_for_casscf(mc, solvent_obj=None, dm=None):
    if solvent_obj is None:
        if isinstance(getattr(mc._scf, 'with_solvent', None), PCM):
            solvent_obj = mc._scf.with_solvent
        else:
            solvent_obj = PCM(mc.mol)
    return _attach_solvent._for_casscf(mc, solvent_obj, dm)

@lib.with_doc(_attach_solvent._for_casci.__doc__)
def pcm_for_casci(mc, solvent_obj=None, dm=None):
    if solvent_obj is None:
        if isinstance(getattr(mc._scf, 'with_solvent', None), PCM):
            solvent_obj = mc._scf.with_solvent
        else:
            solvent_obj = PCM(mc.mol)
    return _attach_solvent._for_casci(mc, solvent_obj, dm)

@lib.with_doc(_attach_solvent._for_post_scf.__doc__)
def pcm_for_post_scf(method, solvent_obj=None, dm=None):
    if solvent_obj is None:
        if isinstance(getattr(method._scf, 'with_solvent', None), PCM):
            solvent_obj = method._scf.with_solvent
        else:
            solvent_obj = PCM(method.mol)
    return _attach_solvent._for_post_scf(method, solvent_obj, dm)

pcm_for_tdscf = _attach_solvent._for_tdscf

@lib.with_doc(_attach_solvent._for_mcpdft.__doc__)
def pcm_for_mcpdft(mc, solvent_obj=None, dm=None):
    if solvent_obj is None:
        if isinstance(getattr(mc._scf, 'with_solvent', None), PCM):
            solvent_obj = mc._scf.with_solvent
        else:
            solvent_obj = PCM(mc.mol)
    return _attach_solvent._for_mcpdft(mc, solvent_obj, dm)

@lib.with_doc(_attach_solvent._for_lpdft.__doc__)
def pcm_for_lpdft(mc, solvent_obj=None, dm=None):
    if solvent_obj is None:
        if isinstance(getattr(mc._scf, 'with_solvent', None), PCM):
            solvent_obj = mc._scf.with_solvent
        else:
            solvent_obj = PCM(mc.mol)
    return _attach_solvent._for_lpdft(mc, solvent_obj, dm)

'''@lib.with_doc(_attach_solvent._for_mrpt.__doc__)
def pcm_for_mrpt(mc, solvent_obj=None, dm=None):
    if solvent_obj is None:
        if isinstance(getattr(mc._scf, 'with_solvent', None), PCM):
            solvent_obj = mc._scf.with_solvent
        else:
            solvent_obj = PCM(mc.mol)
    return _attach_solvent._for_mrpt(mc, solvent_obj, dm)'''

# Inject PCM to other methods
from pyscf import scf
from pyscf import mcscf, mcpdft
from pyscf import mp, ci, cc
from pyscf import tdscf
scf.hf.SCF.PCM    = scf.hf.SCF.PCM    = pcm_for_scf
mp.mp2.MP2.PCM    = mp.mp2.MP2.PCM    = pcm_for_post_scf
ci.cisd.CISD.PCM  = ci.cisd.CISD.PCM  = pcm_for_post_scf
cc.ccsd.CCSD.PCM  = cc.ccsd.CCSD.PCM  = pcm_for_post_scf
tdscf.rhf.TDBase.PCM = tdscf.rhf.TDBase.PCM = pcm_for_tdscf
mcscf.casci.CASCI.PCM = mcscf.casci.CASCI.PCM = pcm_for_casci
mcscf.mc1step.CASSCF.PCM = mcscf.mc1step.CASSCF.PCM = pcm_for_casscf
mcpdft.MultiStateMCPDFTSolver.PCM = mcpdft.MultiStateMCPDFTSolver.PCM = pcm_for_lpdft

# TABLE II,  J. Chem. Phys. 122, 194110 (2005)
XI = {
    6: 4.84566077868,
    14: 4.86458714334,
    26: 4.85478226219,
    38: 4.90105812685,
    50: 4.89250673295,
    86: 4.89741372580,
    110: 4.90101060987,
    146: 4.89825187392,
    170: 4.90685517725,
    194: 4.90337644248,
    302: 4.90498088169,
    350: 4.86879474832,
    434: 4.90567349080,
    590: 4.90624071359,
    770: 4.90656435779,
    974: 4.90685167998,
    1202: 4.90704098216,
    1454: 4.90721023869,
    1730: 4.90733270691,
    2030: 4.90744499142,
    2354: 4.90753082825,
    2702: 4.90760972766,
    3074: 4.90767282394,
    3470: 4.90773141371,
    3890: 4.90777965981,
    4334: 4.90782469526,
    4802: 4.90749125553,
    5294: 4.90762073452,
    5810: 4.90792902522,
}

modified_Bondi = radii.VDW.copy()

modified_Bondi[1] = 1.1/radii.BOHR      
# modified H radius due to Rowland-Taylor 

PI = numpy.pi

def switch_h(x):
    '''
    switching function (eq. 3.19)
    J. Chem. Phys. 133, 244111 (2010)
    notice the typo in the paper
    '''
    y = x**3 * (10.0 - 15.0*x + 6.0*x**2)
    y[x<0] = 0.0
    y[x>1] = 1.0
    return y

def gen_surface(mol, ng=302, rad=modified_Bondi, surface_discretization_method="SWIG"):
    '''J. Phys. Chem. A 1999, 103, 11060-11079'''
    unit_sphere = gen_grid.MakeAngularGrid(ng)
    atom_coords = mol.atom_coords(unit='B')
    N_J = ng * numpy.ones(mol.natm)
    from pyscf.data.elements import charge as charge_of_element
    element_index = [charge_of_element(e) for e in mol.elements]
    R_J = numpy.asarray([rad[chg] for chg in element_index])
    if surface_discretization_method.upper() == "SWIG":
        R_sw_J = R_J * (14.0 / N_J)**0.5
        alpha_J = 1.0/2.0 + R_J/R_sw_J - ((R_J/R_sw_J)**2 - 1.0/28)**0.5
        R_in_J = R_J - alpha_J * R_sw_J

    grid_coords = []
    weights = []
    charge_exp = []
    switch_fun = []
    R_vdw = []
    norm_vec = []
    area = []
    gslice_by_atom = []
    p0 = p1 = 0
    for ia in range(mol.natm):
        r_vdw = R_J[ia]

        atom_grid = r_vdw * unit_sphere[:,:3] + atom_coords[ia,:]
        riJ = scipy.spatial.distance.cdist(atom_grid[:,:3], atom_coords)

        w = unit_sphere[:,3] * 4.0 * PI
        xi = XI[ng] / (r_vdw * w**0.5)

        if surface_discretization_method.upper() == "SWIG":
            diJ = (riJ - R_in_J) / R_sw_J
            diJ[:,ia] = 1.0
            diJ[diJ < 1e-8] = 0.0
            fiJ = switch_h(diJ)
        elif surface_discretization_method.upper() == "ISWIG":
            fiJ = 1 - 0.5 * (erf(xi[:, None] * (R_J[None, :] - riJ)) + erf(xi[:, None] * (R_J[None, :] + riJ)))
            fiJ[:,ia] = 1.0
            fiJ[fiJ < 1e-8] = 0
        else:
            raise NotImplementedError(f"surface_discretization_method = {surface_discretization_method} not recognized")

        swf = numpy.prod(fiJ, axis=1)

        idx = w*swf > 1e-16

        p0, p1 = p1, p1+sum(idx)
        gslice_by_atom.append([p0,p1])
        grid_coords.append(atom_grid[idx,:3])
        weights.append(w[idx])
        switch_fun.append(swf[idx])
        norm_vec.append(unit_sphere[idx,:3])
        charge_exp.append(xi[idx])
        R_vdw.append(numpy.ones(sum(idx)) * r_vdw)
        area.append(w[idx]*r_vdw**2*swf[idx])

    grid_coords = numpy.vstack(grid_coords)
    norm_vec = numpy.vstack(norm_vec)
    weights = numpy.concatenate(weights)
    charge_exp = numpy.concatenate(charge_exp)
    switch_fun = numpy.concatenate(switch_fun)
    area = numpy.concatenate(area)
    R_vdw = numpy.concatenate(R_vdw)

    surface = {
        'ng': ng,
        'gslice_by_atom': gslice_by_atom,
        'grid_coords': grid_coords,
        'weights': weights,
        'charge_exp': charge_exp,
        'switch_fun': switch_fun,
        'R_vdw': R_vdw,
        'norm_vec': norm_vec,
        'area': area,
        'atom_coords': atom_coords
    }
    if surface_discretization_method.upper() == "SWIG":
        surface.update({
            'R_in_J': R_in_J,
            'R_sw_J': R_sw_J,
        })
    elif surface_discretization_method.upper() == "ISWIG":
        surface.update({
            'R_J': R_J,
        })
    return surface

def get_F_A(surface):
    '''
    generate F and A matrix in  J. Chem. Phys. 133, 244111 (2010)
    '''
    R_vdw = surface['R_vdw']
    switch_fun = surface['switch_fun']
    weights = surface['weights']
    A = weights*R_vdw**2*switch_fun
    return switch_fun, A


def get_D_S(surface, with_S=True, with_D=False):
    '''
    generate D and S matrix in  J. Chem. Phys. 133, 244111 (2010)
    The diagonal entries of S is not filled
    '''
    charge_exp  = surface['charge_exp']
    grid_coords = surface['grid_coords']
    switch_fun  = surface['switch_fun']
    norm_vec    = surface['norm_vec']
    R_vdw       = surface['R_vdw']

    xi_i, xi_j = numpy.meshgrid(charge_exp, charge_exp, indexing='ij')
    xi_ij = xi_i * xi_j / (xi_i**2 + xi_j**2)**0.5
    rij = scipy.spatial.distance.cdist(grid_coords, grid_coords)
    xi_r_ij = xi_ij * rij
    numpy.fill_diagonal(rij, 1)
    S = erf(xi_r_ij) / rij
    numpy.fill_diagonal(S, charge_exp * (2.0 / PI)**0.5 / switch_fun)

    D = None
    if with_D:
        drij = numpy.expand_dims(grid_coords, axis=1) - grid_coords
        nrij = numpy.sum(drij * norm_vec, axis=-1)

        D = S*nrij/rij**2 -2.0*xi_r_ij/PI**0.5*numpy.exp(-xi_r_ij**2)*nrij/rij**3
        numpy.fill_diagonal(D, -charge_exp * (2.0 / PI)**0.5 / (2.0 * R_vdw))

    return D, S



class PCM(lib.StreamObject):
    '''
    PCM Solvent Model

    This class implements the Polarizable Continuum Model (PCM) for solvent effects.

    Input Attributes:
    -----------------
    method : str
        The PCM model. Options include 'C-PCM', 'IEF-PCM', 'COSMO', and 'SS(V)PE'.
        Default is 'C-PCM'.

    vdw_scale : float
        A scaling factor for van der Waals radii. Default is 1.2, consistent with Q-Chem settings.

    r_probe : float
        An additional radius (in Angstrom) added to the van der Waals radii.
        Default is 0.0.

    radii_table : dict
        Custom van der Waals radii for each element. By default, scaled van der Waals radii
        from `vdw_scale` and `r_probe` are used.

    lebedev_order : int  (unused – kept for API compatibility only)
        Ignored. The OpenMolcas geodesic tessellation is used instead of
        Lebedev grids. Use `tsnum` to control surface resolution.

    tsnum : int
        Target number of tesserae per atomic sphere for the OpenMolcas
        geodesic polyhedron tessellation. The actual count is the nearest
        achievable value for the chosen seed polyhedron (icosahedron,
        pentakisdodecahedron, or tetrahedron) at integer subdivision
        frequency. Default is 60.

    eps : float
        The dielectric constant of the solvent. Default is 78.3553, the dielectric constant
        for water.

    frozen : bool
        Whether to freeze the potential produced by the solvent during SCF iterations or
        other convergence processes. When frozen=True is set, the solvent is
        assumed to respond slowly, while the electron density relaxes quickly.
        Default is False.

    max_cycle : int
        The maximum number of iterations to relax the solvent.

    conv_tol : float
        The convergence tolerance for total energy during solvent relaxation.

    equilibrium_solvation : bool
        Affects TDDFT and other excited state computations. Controls whether the solvent
        relaxes rapidly with respect to the electron density of the excited state.
        For vertical excitations, it is recommended to set this to False, as the solvent
        typically does not fully relax. In some software packages (e.g., Q-Chem),
        non-equilibrium solvation is applied with an optical dielectric constant of
        eps=1.78. Default is False.

    state_id : int
        Specifies the target state in excited state calculations.
        `state_id=0` corresponds to the ground state, while `state_id=1` corresponds
        to the first excited state. Default is 0.

    surface_discretization_method : str  (unused, kept for API compatibility only)
        Ignored. Sphere overlap is handled by exact geometric clipping

    Saved Results:
    --------------
    e_tot : float
        The energy contribution from the solvent.

    v : ndarray
        The potential matrix generated by the solvent.

    Intermediate Attributes:
    ------------------------
    These attributes are generated during calculations and should not be modified.
    Additionally, they may not be compatible between GPU and CPU implementations.

    - surface
    - _intermediates
    - v_grids_n
    '''


    _keys = {
        'method', 'vdw_scale', 'surface', 'r_probe',
        'mol', 'radii_table', 'lebedev_order',
        'eps', 'max_cycle', 'conv_tol', 'state_id', 'frozen',
        'equilibrium_solvation', 'e', 'v', 'v_grids_n',
        'surface_discretization_method',
        'conv_energy', 'es_method', 'refdm',
        'refidx', 'partition', 'rf_root', 'phi_state',
        'neq_method', 'conv_potential'}


    def __init__(self, mol):
        self.mol = mol
        self.stdout = mol.stdout
        self.verbose = mol.verbose
        self.max_memory = mol.max_memory
        self.method = 'C-PCM'

        self.vdw_scale = 1.2 # default value in qchem
        self.r_probe = 0.0
        self.radii_table = None
        self.lebedev_order = 29
        self.eps = 78.3553
        self.surface_discretization_method = "SWIG"

        self.max_cycle = 20
        self.conv_tol = 1e-7
        self.state_id = 0

        self.frozen = False
        self.equilibrium_solvation = False

        self.surface = {}
        self._intermediates = {}
        self.v_grids_n = None # nuclear potential on grids

        self.e = None
        self.v = None
        self._dm = None

        # Some aditional attributes
        self.conv_energy = None
        self.conv_potential = None
        self.es_method = None
        self.refdm = None
        self.refidx = 0.0
 
        self.partition = 'pekar'
        self.rf_root = None

        self.phi_state = None
        self.neq_method = None

    def dump_flags(self, verbose=None):
        logger.info(self, '******** %s (In testing) ********', self.__class__)
        logger.warn(self, 'PCM is an experimental feature. It is '
                    'still in testing.\nFeatures and APIs may be changed '
                    'in the future.')
        logger.info(self, 'lebedev_order = %s (%d grids per sphere)',
                    self.lebedev_order, gen_grid.LEBEDEV_ORDER[self.lebedev_order])
        logger.info(self, 'eps = %s'          , self.eps)
        logger.info(self, 'frozen = %s'       , self.frozen)
        #logger.info(self, 'equilibrium_solvation = %s', self.equilibrium_solvation)
        return self

    def to_gpu(self):
        from pyscf.lib import to_gpu
        obj = to_gpu(self)
        return obj.reset()

    def reset(self, mol=None):
        if mol is not None:
            self.mol = mol
        self._intermediates = None
        self.surface = None
        self.v_grids_n = None
        return self
    

    def build(self, ng=None):
        if self.radii_table is None:
            vdw_scale = self.vdw_scale
            radii_table = vdw_scale * modified_Bondi + self.r_probe/radii.BOHR
            print('radii_table', radii_table)
        else:
            radii_table = self.radii_table
        print('radii_table in build', radii_table)
        logger.debug2(self, 'radii_table %s', radii_table)
        mol = self.mol
        if ng is None:
            ng = gen_grid.LEBEDEV_ORDER[self.lebedev_order]

        self.surface = gen_surface(mol, rad=radii_table, ng=ng,
                                   surface_discretization_method = self.surface_discretization_method)
        self._intermediates = {}
        F, A = get_F_A(self.surface)
        D, S = get_D_S(self.surface, with_S=True, with_D=True)

        epsilon = self.eps
        if self.method.upper() in ['C-PCM', 'CPCM']:
            f_epsilon = (epsilon-1.)/epsilon if epsilon != float('inf') else 1.0
            K = S
            R = -f_epsilon * numpy.eye(K.shape[0])
        elif self.method.upper() == 'COSMO':
            f_epsilon = (epsilon - 1.0)/(epsilon + 1.0/2.0) if epsilon != float('inf') else 1.0
            K = S
            R = -f_epsilon * numpy.eye(K.shape[0])
        elif self.method.upper() in ['IEF-PCM', 'IEFPCM']:
            f_epsilon = (epsilon - 1.0)/(epsilon + 1.0) if epsilon != float('inf') else 1.0
            DA = D*A
            DAS = numpy.dot(DA, S)
            K = S - f_epsilon/(2.0*PI) * DAS
            R = -f_epsilon * (numpy.eye(K.shape[0]) - 1.0/(2.0*PI)*DA)
        elif self.method.upper() == 'SS(V)PE':
            f_epsilon = (epsilon - 1.0)/(epsilon + 1.0) if epsilon != float('inf') else 1.0
            DA = D*A
            DAS = numpy.dot(DA, S)
            K = S - f_epsilon/(4.0*PI) * (DAS + DAS.T)
            R = -f_epsilon * (numpy.eye(K.shape[0]) - 1.0/(2.0*PI)*DA)
        else:
            raise RuntimeError(f"Unknown implicit solvent model: {self.method}")

        intermediates = {
            'S': S,
            'D': D,
            'A': A,
            'K': K,
            'R': R,
            'f_epsilon': f_epsilon
        }
        self._intermediates.update(intermediates)

        charge_exp  = self.surface['charge_exp']
        grid_coords = self.surface['grid_coords']
        atom_coords = mol.atom_coords(unit='B')
        atom_charges = mol.atom_charges()

        int2c2e = mol._add_suffix('int2c2e')
        fakemol = gto.fakemol_for_charges(grid_coords, expnt=charge_exp**2)
        fakemol_nuc = gto.fakemol_for_charges(atom_coords)
        v_ng = gto.mole.intor_cross(int2c2e, fakemol_nuc, fakemol)
        self.v_grids_n = numpy.dot(atom_charges, v_ng)


    def _get_vind(self, dms):
        if not self._intermediates:
            self.build()

        nao = dms.shape[-1]
        #print('dms.shape', dms.shape)
        dms = dms.reshape(-1,nao,nao)
        if dms.shape[0] == 2:
            dms = (dms[0] + dms[1]).reshape(-1,nao,nao)

        K = self._intermediates['K']
        R = self._intermediates['R']
        v_grids_e = self._get_v(dms)
        v_grids = self.v_grids_n - v_grids_e

        b = numpy.dot(R, v_grids.T)
        q = numpy.linalg.solve(K, b).T

        vK_1 = numpy.linalg.solve(K.T, v_grids.T)
        qt = numpy.dot(R.T, vK_1).T
        q_sym = (q + qt)/2.0



        vmat = self._get_vmat(q_sym)
        epcm = 0.5 * numpy.dot(q_sym[0], v_grids[0])


        self._intermediates['q'] = q[0]
        self._intermediates['q_sym'] = q_sym[0]
        self._intermediates['v_grids'] = v_grids[0]
        self._intermediates['dm'] = dms
        return epcm, vmat[0], q_sym, v_grids

    def _get_vind_CAS(self, dms):
        if not self._intermediates:
            self.build()

        nao = dms.shape[-1]
        #print('dms.shape', dms.shape)
        dms = dms.reshape(-1,nao,nao)
        if dms.shape[0] == 2:
            dms = (dms[0] + dms[1]).reshape(-1,nao,nao)

        K = self._intermediates['K']
        R = self._intermediates['R']
        v_grids_e = -self._get_v(dms)
        #v_grids = self.v_grids_n - v_grids_e

        # separate q_N and q_ele calculation
        b_N = numpy.dot(R, self.v_grids_n.T)
        b_ele = numpy.dot(R, v_grids_e.T)

        q_N = numpy.linalg.solve(K, b_N).T
        q_ele = numpy.linalg.solve(K, b_ele).T

        #b = numpy.dot(R, v_grids.T)
        #q = numpy.linalg.solve(K, b).T

        vk_1_N = numpy.linalg.solve(K.T, self.v_grids_n.T)
        vk_1_ele = numpy.linalg.solve(K.T, v_grids_e.T)

        qt_N = numpy.dot(R.T, vk_1_N).T
        qt_ele = numpy.dot(R.T, vk_1_ele).T
        q_sym_N = (q_N + qt_N)/2.0
        q_sym_ele = (q_ele + qt_ele)/2.0

        #vK_1 = numpy.linalg.solve(K.T, v_grids.T)
        #qt = numpy.dot(R.T, vK_1).T
        #q_sym = (q + qt)/2.0

        vmat_N = self._get_vmat(q_sym_N)
        vmat_ele = self._get_vmat(q_sym_ele)
        epcm_N = 0.5 * numpy.dot(q_sym_N[0], self.v_grids_n[0])
        epcm_ele = 0.5 * numpy.dot(q_sym_ele[0], v_grids_e[0])

        #vmat = self._get_vmat(q_sym)
        #epcm = 0.5 * numpy.dot(q_sym[0], v_grids[0])


        #self._intermediates['q'] = q[0]
        #self._intermediates['q_sym'] = q_sym[0]
        #self._intermediates['v_grids'] = v_grids[0]
        #self._intermediates['dm'] = dms
        #return epcm, vmat[0], q_sym, v_grids
        return (epcm_N, vmat_N[0], q_sym_N, self.v_grids_n), (epcm_ele , vmat_ele[0],  q_sym_ele,  v_grids_e)

    def _get_K_opt_R_opt(self):
        '''
        Return K_opt and R_opt for the optical dielectric constant epsilon_opt = refidx**2.

        These matrices are NOT stored in build() because refidx is typically
        set after the HF/ground-state build (it is only needed for nonequilibrium
        excited-state calculations).  Instead they are computed on demand from
        the current self.refidx and cached in _intermediates under 'K_opt' and
        'R_opt'.  The cache is invalidated whenever refidx changes by checking
        the stored 'refidx_cached' value.
        '''
        if not self._intermediates:
            self.build()

        refidx = self.refidx
        epsilon_opt = refidx**2

        # Use cached version if refidx hasn't changed
        if (self._intermediates.get('refidx_cached') == refidx
                and 'K_opt' in self._intermediates
                and 'R_opt' in self._intermediates):
            return self._intermediates['K_opt'], self._intermediates['R_opt']

        D = self._intermediates['D']
        A = self._intermediates['A']
        S = self._intermediates['S']

        if epsilon_opt <= 0.0:
            raise ValueError(
                'refidx must be set before calling nonequilibrium methods. '
                'Set mc.with_solvent.refidx = <refractive index> (e.g. 1.3328 for water).')
        
        if self.method.upper() in ['C-PCM', 'CPCM']:
            f_epsilon_opt = (epsilon_opt - 1.0) / epsilon_opt
            K_opt = S
            R_opt = -f_epsilon_opt * numpy.eye(S.shape[0])
        elif self.method.upper() == 'COSMO':
            f_epsilon_opt = (epsilon_opt - 1.0) / (epsilon_opt + 1.0/2.0)
            K_opt = S
            R_opt = -f_epsilon_opt * numpy.eye(S.shape[0])
        elif self.method.upper() in ['IEF-PCM', 'IEFPCM']:
            f_epsilon_opt = (epsilon_opt - 1.0)/(epsilon_opt + 1.0) 
            DA = D*A
            DAS = numpy.dot(DA, S)
            K_opt = S - f_epsilon_opt/(2.0*PI) * DAS
            R_opt = -f_epsilon_opt * (numpy.eye(K_opt.shape[0]) - 1.0/(2.0*PI)*DA)

        else:
            raise NotImplementedError(
                f'K_opt/R_opt in nonequilibrium not yet implemented for {self.method}.')


        self._intermediates['K_opt']        = K_opt
        self._intermediates['R_opt']        = R_opt
        self._intermediates['refidx_cached'] = refidx
        return K_opt, R_opt

    def _get_vind_pekar(self, dms):
        if not self._intermediates:
            self.build()

        nao = dms.shape[-1]
        dms = dms.reshape(-1,nao,nao)
        if dms.shape[0] == 2:
            dms = (dms[0] + dms[1]).reshape(-1,nao,nao)

        K_opt, R_opt = self._get_K_opt_R_opt()   # lazy — uses current refidx
        v_grids_e = self._get_v(dms)
        v_grids = self.v_grids_n - v_grids_e

        b_N = numpy.dot(R_opt, self.v_grids_n.T)
        b_ele = numpy.dot(R_opt, v_grids_e.T)

        q_N = numpy.linalg.solve(K_opt, b_N).T
        q_ele = numpy.linalg.solve(K_opt, b_ele).T

        b = numpy.dot(R_opt, v_grids.T)
        q = numpy.linalg.solve(K_opt, b).T

        vk_1_N = numpy.linalg.solve(K_opt.T, self.v_grids_n.T)
        vk_1_ele = numpy.linalg.solve(K_opt.T, v_grids_e.T)

        qt_N = numpy.dot(R_opt.T, vk_1_N).T
        qt_ele = numpy.dot(R_opt.T, vk_1_ele).T
        q_sym_N = (q_N + qt_N)/2.0
        q_sym_ele = (q_ele + qt_ele)/2.0

        vK_1 = numpy.linalg.solve(K_opt.T, v_grids.T)
        qt = numpy.dot(R_opt.T, vK_1).T
        q_sym = (q + qt)/2.0

        vmat_N = self._get_vmat(q_sym_N)
        vmat_ele = self._get_vmat(q_sym_ele)
        epcm_N = 0.5 * numpy.dot(q_sym_N[0], self.v_grids_n[0])
        epcm_ele = 0.5 * numpy.dot(q_sym_ele[0], v_grids_e[0])

        return q_sym, v_grids
        #return (q_sym_N, self.v_grids_n), (q_sym_ele, v_grids_e)  

    def _get_vind_pekar2(self, dms):
        if not self._intermediates:
            self.build()

        nao = dms.shape[-1]
        dms = dms.reshape(-1,nao,nao)
        if dms.shape[0] == 2:
            dms = (dms[0] + dms[1]).reshape(-1,nao,nao)

        K_opt, R_opt = self._get_K_opt_R_opt()   # lazy — uses current refidx
        v_grids_e = -self._get_v(dms)
        #v_grids = self.v_grids_n - v_grids_e

        b_N = numpy.dot(R_opt, self.v_grids_n.T)
        b_ele = numpy.dot(R_opt, v_grids_e.T)

        q_N = numpy.linalg.solve(K_opt, b_N).T
        q_ele = numpy.linalg.solve(K_opt, b_ele).T


        vk_1_N = numpy.linalg.solve(K_opt.T, self.v_grids_n.T)
        vk_1_ele = numpy.linalg.solve(K_opt.T, v_grids_e.T)

        qt_N = numpy.dot(R_opt.T, vk_1_N).T
        qt_ele = numpy.dot(R_opt.T, vk_1_ele).T
        q_sym_N = (q_N + qt_N)/2.0
        q_sym_ele = (q_ele + qt_ele)/2.0

        return (q_sym_N, self.v_grids_n), (q_sym_ele, v_grids_e)  

    def _get_vind_marcus(self, dm_es, q_or):
        '''
        Marcus partition 


        Parameters
        ----------
        dm_es  : ndarray (nao, nao)  — excited-state density matrix
        q_or   : ndarray (ngrids,)   — Marcus orientational (slow) charges

        Returns
        -------
        q_el   : ndarray (1, ngrids) — fast charges (symmetrised)
        v_grids_mol : ndarray (1, ngrids) — molecular ESP (without V_slow)
        '''
        if not self._intermediates:
            self.build()

        nao = dm_es.shape[-1]
        dms = dm_es.reshape(-1, nao, nao)
        if dms.shape[0] == 2:
            dms = (dms[0] + dms[1]).reshape(-1, nao, nao)

        K_opt, R_opt = self._get_K_opt_R_opt()   # lazy — uses current refidx
        S     = self._intermediates['S']

        v_grids_e = self._get_v(dms)
        v_grids_mol = self.v_grids_n - v_grids_e   


        v_slow = S @ q_or                      

        v_grids_total = v_grids_mol.copy()
        v_grids_total[0] += v_slow

        b = numpy.dot(R_opt, v_grids_total.T)      
        q = numpy.linalg.solve(K_opt, b).T

        vK_1 = numpy.linalg.solve(K_opt.T, v_grids_total.T)
        qt   = numpy.dot(R_opt.T, vK_1).T
        q_el = (q + qt) / 2.0                       

        return q_el, v_grids_mol


    def _get_v(self, dms):
        '''
        return electrostatic potential on surface
        '''
        mol = self.mol
        nao = dms.shape[-1]
        grid_coords = self.surface['grid_coords']
        exponents   = self.surface['charge_exp']
        ngrids = grid_coords.shape[0]
        nset = dms.shape[0]
        v_grids_e = numpy.empty([nset, ngrids])
        max_memory = self.max_memory - lib.current_memory()[0]
        blksize = int(max(max_memory*.9e6/8/nao**2, 400))
        int3c2e = mol._add_suffix('int3c2e')
        cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, int3c2e)
        for p0, p1 in lib.prange(0, ngrids, blksize):
            fakemol = gto.fakemol_for_charges(grid_coords[p0:p1], expnt=exponents[p0:p1]**2)
            fakemol.cart = mol.cart
            v_nj = df.incore.aux_e2(mol, fakemol, intor=int3c2e, aosym='s1', cintopt=cintopt)
            for i in range(nset):
                v_grids_e[i,p0:p1] = numpy.einsum('ijL,ij->L',v_nj, dms[i])

        return v_grids_e
    
    def _get_vmat(self, q):
        mol = self.mol
        nao = mol.nao
        grid_coords = self.surface['grid_coords']
        exponents   = self.surface['charge_exp']
        ngrids = grid_coords.shape[0]
        q = q.reshape([-1,ngrids])
        nset = q.shape[0]
        vmat = numpy.zeros([nset,nao,nao])
        max_memory = self.max_memory - lib.current_memory()[0]
        blksize = int(max(max_memory*.9e6/8/nao**2, 400))

        int3c2e = mol._add_suffix('int3c2e')
        cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, int3c2e)
        for p0, p1 in lib.prange(0, ngrids, blksize):
            fakemol = gto.fakemol_for_charges(grid_coords[p0:p1], expnt=exponents[p0:p1]**2)
            fakemol.cart = mol.cart
            v_nj = df.incore.aux_e2(mol, fakemol, intor=int3c2e, aosym='s1', cintopt=cintopt)
            for i in range(nset):
                vmat[i] += -numpy.einsum('ijL,L->ij', v_nj, q[i,p0:p1])
        return vmat


    def kernel(self, dm, state_id=None, es_method=None, refdm=None):
        '''A single shot solvent effects for given density matrix.
        '''
        
        self._dm = dm
        #self.refdm = refdm
        #self.state_id = state_id        
        #self.es_method = es_method

        if state_id in [0, None]:
            sol_energy, sol_pot, _, _ = self._get_vind(dm)
            logger.info(self, 'Ground state equilibrium E(pol) = %.15g', sol_energy)
            #print("self.equilibrium_solvation", self.equilibrium_solvation)
            return sol_energy, sol_pot

        elif isinstance(state_id, int) and state_id > 0 and es_method in ['nonequilibrium', 'non-equilibrium', 'non_equilibrium', 
                                            'non-eq', 'non_eq', 'NEQ', 'neq', 'NONEQ',
                                            'VEM', 'vem']:
            
            #print("state id ", state_id)
            #print("self.state_id", self.state_id)

            # ------------------------------------------------------------------
            # Decide which charge partition to use
            # ------------------------------------------------------------------
            partition = getattr(self, 'partition', 'pekar').lower()
            epsilon     = self.eps
            epsilon_opt = self.refidx ** 2
            if partition == 'marcus':
                # ==============================================================
                # Marcus partition (Partition I / OpenMolcas) — eqs 13, 15, 18
                # ==============================================================

               
                _, _, _q_ref_total, _vgrid_ref = self._get_vind(refdm)

                fact_or = (epsilon - epsilon_opt) / (epsilon - 1.0)
                q_or = fact_or * _q_ref_total[0]          # shape (ngrids,)

                q_el_gs = _q_ref_total[0] - q_or       

                q_el_es, v_grids_mol = self._get_vind_marcus(dm, q_or)

                q_neq = q_el_es.copy()
                q_neq[0] += q_or
                sol_pot = self._get_vmat(q_neq)[0]

                S = self._intermediates['S']
                v_slow = S @ q_or                         # V_slow at tesserae

                v_es = v_grids_mol[0]                     
                v_gs = _vgrid_ref[0]                      


                G_el   = 0.5 * numpy.dot(q_el_es[0], v_es)

                G_or   = numpy.dot(q_or, v_es - 0.5 * v_gs)


                delta_q_el = q_el_es[0] - q_el_gs       
                G_el_or = 0.5 * numpy.dot(delta_q_el, v_slow)

                epcm = G_el + G_or + G_el_or

                logger.info(self, 'Marcus partition:  epsilon = %.4f,  epsilon_opt = %.4f', epsilon, epsilon_opt)
                logger.info(self, '  G_P total= %.15g', epcm)

            else:
                # ==============================================================
                # Pekar partition (default, original implementation)
                # ==============================================================
                _q_ref_dyn, _vgrid_ref = self._get_vind_pekar(refdm)  # GS dynamic charge
                _, _, _q_ref, _ = self._get_vind(refdm)              # GS total charge
                _q_ref_in = _q_ref - _q_ref_dyn                      # inertial charge

                _q_dyn, _vgrid = self._get_vind_pekar(dm)              # ES dynamic charge
                _q_neq = _q_ref_in + _q_dyn                          # NEQ total charge

                sol_pot = self._get_vmat(_q_neq)[0]

                epcm_dyn = 0.5 * numpy.dot(_q_dyn[0], _vgrid[0])
                epcm_in  = numpy.dot((_vgrid[0] - 0.5 * _vgrid_ref[0]), _q_ref_in[0])
                epcm = epcm_dyn + epcm_in

                logger.info(self, 'Pekar partition:  eps_opt = %s', epsilon_opt)
                logger.info(self, 'Excited state %d : NEQ E(pol) = %.15g (dyn) + %.15g (in) = %.15g',
                            self.state_id, epcm_dyn, epcm_in, epcm)

            return epcm, sol_pot
        
        elif (isinstance(state_id, str)
              and state_id.lower() == 'sa_casscf'
              and es_method in ['nonequilibrium', 'non-equilibrium', 'non_equilibrium',
                                'non-eq', 'non_eq', 'NEQ', 'neq', 'NONEQ', 'VEM', 'vem']
              and not self.equilibrium_solvation
              and refdm is None
              and getattr(self, 'phi_state', None) is not None):
            # ==================================================================
            # SA-CASSCF NEQ — Pekar partition, slow+fast from same SA-CASSCF
            # ------------------------------------------------------------------
 
            phi       = self.phi_state
            _sa_dm    = dm[-1]
            state_dms = dm[:-1]
            n_states  = len(state_dms)

            #np.savetxt('sadm_kernel.txt', _sa_dm)
            if phi >= n_states:
                raise ValueError(
                    f'phi_state={phi} out of range for {n_states} SA states.')

            partition = getattr(self, 'partition', 'pekar').lower()
            if partition != 'pekar':
                raise NotImplementedError(
                    f'SA-CASSCF NEQ only implemented for Pekar partition, not {partition}.')

            # STEP 1: total equilibrium charges from state 0
            #(_, _, _q0_N_total, _vgrid0_N), (_, _, _q0_ele_total, _vgrid0_ele) = self._get_vind_CAS(state_dms[0])
            _, _, _q0_total, _vgrid0 = self._get_vind(state_dms[0])

            # fast charges of state 0 
            #(_q0_N_fast, _), (_q0_ele_fast, _) = self._get_vind_pekar2(state_dms[0])
            _q0_fast, _ = self._get_vind_pekar(state_dms[0])

            # slow charges of state 0 
            #_q0_N_slow = _q0_N_total - _q0_N_fast
            #_q0_ele_slow = _q0_ele_total - _q0_ele_fast
            _q0_slow = _q0_total - _q0_fast   


            # STEP 2: fast charges from state phi 
            #(_qphi_N_fast, _vgrid_N_phi), (_qphi_ele_fast, _vgrid_ele_phi) = self._get_vind_pekar2(state_dms[phi])
            _qphi_fast, _vgrid_phi = self._get_vind_pekar(state_dms[phi])
   
            #slow_self_N = -0.5 * numpy.dot(_vgrid0_N[0], _q0_N_slow[0])
            #slow_self_ele = -0.5 * numpy.dot(_vgrid0_ele[0], _q0_ele_slow[0])

            slow_self = -0.5 * numpy.dot(_vgrid0[0], _q0_slow[0])

            logger.info(self, 'SA-CASSCF NEQ: slow charges from ground state, \n ' \
                                'fast charges from state %d (phi_state)', phi)

            # STEP 3: reaction field from total NEQ charge = Q^slow(0) + Q^fast(phi)
            #_q_neq  = _q0_ele_slow + _qphi_ele_fast 
            _q_neq  = _q0_slow + _qphi_fast 

            sol_pot = self._get_vmat(_q_neq)[0]

            # STEP 4: per-state free energies
            epcms = []
            for i, d in enumerate(state_dms):

                if i == 0:
                    # G_0 
                    '''epcm_i = ( -0.5 * numpy.dot(_vgrid0_ele[0], _q0_ele_total[0]) 
                              + 0.5 * numpy.dot(_vgrid0_N[0], _q0_N_total[0]) )'''
                    epcm_i = 0.5 * numpy.dot(_vgrid0[0], _q0_total[0])

                else:
                    # general formula for state > 0:
                    '''(_qk_N_fast, _vgrid_N_k), (_qk_ele_fast, _vgrid_ele_k) = self._get_vind_pekar2(state_dms[i])
                    fast_self_N_k  = 0.5 * numpy.dot(_vgrid_N_k[0], _qk_N_fast[0])          # +1
                    fast_self_ele_k  = -0.5 * numpy.dot(_vgrid_ele_k[0], _qk_ele_fast[0])   # +1'''   
                    _qk_fast, _vgrid_k = self._get_vind_pekar(state_dms[i])                 
                    fast_self_k  = 0.5 * numpy.dot(_vgrid_k[0], _qk_fast[0])               

                    slow_inter_k = numpy.dot(_vgrid_k[0], _q0_slow[0])                     # already included
                    #slow_inter_N_k = numpy.dot(_vgrid_N_k[0], _q0_N_slow[0])     

                    epcm_i = fast_self_k + slow_inter_k + slow_self
                    #epcm_i = fast_self_N_k + fast_self_ele_k + slow_inter_N_k + slow_self_N + slow_self_ele
                    

                epcms.append(epcm_i)
                logger.info(self, '  G_P state %d = %.15g', i, epcm_i)

            logger.info(self, 'SA-CASSCF NEQ Pekar G_P per state = %s', epcms)
            return epcms, sol_pot

        elif (isinstance(state_id, str)
              and state_id.lower() == 'sa_casscf'
              and es_method in ['nonequilibrium', 'non-equilibrium', 'non_equilibrium',
                                'non-eq', 'non_eq', 'NEQ', 'neq', 'NONEQ', 'VEM', 'vem']
              and not self.equilibrium_solvation
              and getattr(self, 'neq_method', None) is not None
              and str(self.neq_method).upper() == 'II'):

            _sa_dm    = dm[-1]
            state_dms = dm[:-1]
 

            _, _, _q0_total, _vgrid0 = self._get_vind(state_dms[0])

            _q0_fast, _ = self._get_vind_pekar(state_dms[0])
 

            _q0_slow = _q0_total - _q0_fast 

            _qfast_sa, _ = self._get_vind_pekar(_sa_dm)   

            slow_self = -0.5 * numpy.dot(_vgrid0[0], _q0_slow[0])
 

 

            _q_neq  = _q0_slow + _qfast_sa
            sol_pot = self._get_vmat(_q_neq)[0]
 
            epcms = []
            
            #_qfast_sa =  np.zeros((1, _q0_slow.shape[1]))

            for i, d in enumerate(state_dms):
                _, _, _, _vgrid_i = self._get_vind(d)
                #_qfast_i, _vgrid_i = self._get_vind_pekar(d)
                v_i = _vgrid_i[0]
 
                fast_self_k  = 0.5 * numpy.dot(v_i, _qfast_sa[0])  
                #fast_self_k  = 0.5 * numpy.dot(v_i, _qfast_i[0])
                slow_inter_k = numpy.dot(v_i, _q0_slow[0])           
                epcm_i       = fast_self_k + slow_inter_k + slow_self
 
                epcms.append(epcm_i)
                
                #_qfast_sa = _qfast_sa + _qfast_i 

 
            #_qfast_sa = _qfast_sa / len(state_dms)  # average (or state averaged?) fast charge 
            #_q_neq  = _q0_slow + _qfast_sa
            #sol_pot = self._get_vmat(_q_neq)[0]

            logger.info(self, 'SA-CASSCF NEQ Method II G_P per state = %s', epcms)
            return epcms, sol_pot
 
        elif isinstance(state_id, str) and state_id.lower() == 'sa_casscf' and es_method in [
                'nonequilibrium', 'non-equilibrium', 'non_equilibrium',
                'non-eq', 'non_eq', 'NEQ', 'neq', 'NONEQ', 'VEM', 'vem']:
            # ==================================================================
            # OpenMolcas style
            # ==================================================================

            partition   = getattr(self, 'partition', 'pekar').lower()
            epsilon     = self.eps
            epsilon_opt = self.refidx ** 2
            rf_root     = getattr(self, 'rf_root', None)
 
            # Split dm list
            _sa_dm    = dm[-1]           # SA-averaged density
            state_dms = dm[:-1]          # list of N_states individual densities
            n_states  = len(state_dms)
 
            # Select density for fast (electronic) charges
            if rf_root is None:
                dm_fast = _sa_dm
                logger.info(self, 'SA-CASSCF NEQ: fast charges from SA-averaged density')
            else:
                dm_fast = state_dms[rf_root]   # 0-indexed
                logger.info(self, 'SA-CASSCF NEQ: fast charges from state %d density (rf_root=%d)',
                            rf_root, rf_root)
 
 
            if partition == 'marcus':

                # --- GS quantities -------------------------------------------
                _, _, _q_ref_total, _vgrid_ref = self._get_vind(refdm)
                v_gs    = _vgrid_ref[0]              
 
                fact_or = (epsilon - epsilon_opt) / (epsilon - 1.0)
                q_or    = fact_or * _q_ref_total[0] 
 
                q_el_gs = _q_ref_total[0] - q_or     
 
                S, K_opt, R_opt = (self._intermediates['S'],
                                   *self._get_K_opt_R_opt())  # lazy — uses current refidx
                v_slow = S @ q_or                   
 
                v_nuc_slow       = numpy.zeros((1, len(q_or)))
                v_nuc_slow[0]    = self.v_grids_n + v_slow
                b_fn             = numpy.dot(R_opt, v_nuc_slow.T)
                q_fn_raw         = numpy.linalg.solve(K_opt, b_fn).T
                vK_fn            = numpy.linalg.solve(K_opt.T, v_nuc_slow.T)
                q_fn_t           = numpy.dot(R_opt.T, vK_fn).T
                q_fast_nuc       = (q_fn_raw + q_fn_t) / 2.0 
 
                q_fast_total, _ = self._get_vind_marcus(dm_fast, q_or) 
 
                q_fast_el = q_fast_total - q_fast_nuc               
 
                q_neq      = q_fast_total.copy()
                q_neq[0]  += q_or
                sol_pot    = self._get_vmat(q_neq)[0]
 

                half_ENN        = 0.5 * numpy.dot(q_fast_nuc[0], self.v_grids_n)
                half_W_or_nuc   = 0.5 * numpy.dot(q_or,          self.v_grids_n)
                half_W_InfNuc   = 0.5 * numpy.dot(q_fast_nuc[0], v_slow)
                W_0_or_el       = numpy.dot(q_or, v_gs - self.v_grids_n)  # q_or·V_el_GS
                W_0_or_Inf      = numpy.dot(q_el_gs, v_slow)
                const = (half_ENN + half_W_or_nuc + half_W_InfNuc
                         - 0.5 * W_0_or_el - 0.5 * W_0_or_Inf)
 
                logger.info(self, 'SA-CASSCF Marcus:  ε=%.4f  ε_opt=%.4f  fact_or=%.6f',
                            epsilon, epsilon_opt, fact_or)
                logger.info(self, '  1/2 ENN        = %.10g', half_ENN)
                logger.info(self, '  1/2 W_or_nuc   = %.10g', half_W_or_nuc)
                logger.info(self, '  1/2 W_or_InfNuc= %.10g', half_W_InfNuc)
                logger.info(self, '  -1/2 W_0_or_el = %.10g', -0.5 * W_0_or_el)
                logger.info(self, '  -1/2 W_0_or_Inf= %.10g', -0.5 * W_0_or_Inf)
                logger.info(self, '  RepNuc const = %.10g', const)
 

                epcms = []
                for i, d in enumerate(state_dms):
                    if rf_root is not None:
                        _, _, _, _vgrid_k = self._get_vind(dm_fast)
                    else:
                        _, _, _, _vgrid_k = self._get_vind(d)

                    v_el_k = _vgrid_k[0] - self.v_grids_n   # V_el^(k) (negative)
 
                    h1_k     = numpy.dot(q_fast_nuc[0] + q_or, v_el_k)  # line 278
                    twoham_k = 0.5 * numpy.dot(q_fast_el[0],  v_el_k)   # line 299
                    epcm_k   = const + h1_k + twoham_k
 
                    epcms.append(epcm_k)
                    logger.info(self, '  State %d:  h1=%.10g  TwoHam=%.10g  G_P=%.10g',
                                i, h1_k, twoham_k, epcm_k)
 
            else:
               
                # Pekar partition
 
                _, _, _q_ref, _vgrid_ref = self._get_vind(refdm)
                _q_ref_dyn, _            = self._get_vind_pekar(refdm)
                _q_ref_in                = _q_ref - _q_ref_dyn    # slow (fixed)
 
                # Fast (dynamic) charges from dm_fast
                _q_dyn, _ = self._get_vind_pekar(dm_fast)
                _q_neq    = _q_ref_in + _q_dyn
 
                # RF matrix contribution (same for all states)
                sol_pot   = self._get_vmat(_q_neq)[0]
 
                # State-independent term:  
                const_gs_term = -0.5 * numpy.dot(_q_ref_in[0], _vgrid_ref[0])
 
                # Per-state energies:
                
                epcms = []
                _q_dyn_sum =  numpy.zeros((1, _q_ref_in.shape[1]))
                for i, d in enumerate(state_dms):
                    if rf_root is not None:
                        _, _, _, _vgrid_k = self._get_vind(dm_fast)
                    else:
                        _q_dyn, _vgrid_k = self._get_vind_pekar(d)
                        _q_dyn_sum = _q_dyn_sum + _q_dyn 
                        #_, _, _, _vgrid_k = self._get_vind(d)

                    v_k    = _vgrid_k[0]
                    epcm_k = (0.5 * numpy.dot(_q_dyn[0], v_k)
                              + numpy.dot(_q_ref_in[0], v_k)
                              + const_gs_term)
                    epcms.append(epcm_k)

                if rf_root is None:
                    _q_dyn_sa = _q_dyn_sum / len(state_dms)  # average (or state averaged?) fast charge 
                    _q_neq  = _q_ref_in + _q_dyn_sa
                    sol_pot = self._get_vmat(_q_neq)[0]

                logger.info(self, 'SA-CASSCF Pekar:  eps_opt=%.6f',
                            epsilon_opt)
 
            logger.info(self, 'SA-CASSCF NEQ G_P per state = %s', epcms)
            return epcms, sol_pot
        
        elif isinstance(state_id, str) and state_id.lower() == 'sa_casscf' and es_method is None and self.equilibrium_solvation:
            # ==================================================================
            # SA-CASSCF equilibrium branch
            # ==================================================================
            _sa_dm    = dm[-1]
            state_dms = dm[:-1]
            rf_root   = getattr(self, 'rf_root', None)
 
            # Select density for the equilibrium charges
            if rf_root is None:
                dm_charges = _sa_dm
                logger.info(self, 'SA-CASSCF EQ: charges from SA-averaged density')
            else:
                dm_charges = state_dms[rf_root]   #  0-indexed
                logger.info(self, 'SA-CASSCF EQ: charges from state %d density (rf_root=%d)',
                            rf_root, rf_root)
 
            # Compute equilibrium charges from the selected density
            _, _, _q, _ = self._get_vind(dm_charges)   # _q shape: (1, ngrids)
 
            # RF matrix is the same for all states
            sol_pot = self._get_vmat(_q)[0]
 
            # Per-state energies:
            epcms = []
            for i, d in enumerate(state_dms):
                #_, _, _, _vgrid_k = self._get_vind(d)
                if rf_root is not None:
                        _, _, _, _vgrid_k = self._get_vind(dm_charges)
                else:
                        _, _, _, _vgrid_k = self._get_vind(d)

                v_k    = _vgrid_k[0]
                epcm_k = 0.5 * numpy.dot(_q[0], v_k)
                epcms.append(epcm_k)
                logger.info(self, '  SA-CASSCF EQ G_P state %d = %.15g', i, epcm_k)
 
            logger.info(self, 'SA-CASSCF EQ G_P per state = %s', epcms)
            return epcms, sol_pot



    def nuc_grad_method(self, grad_method):
        raise DeprecationWarning('Use the make_grad_object function from '
                                 'pyscf.solvent.grad.pcm or '
                                 'pyscf.solvent._ddcosmo_tdscf_grad instead.')

    def grad(self, dm):
        '''Computes the Jacobian for the energy associated with the solvent,
        including the derivatives of the solvent itsself and the interactions
        between the solvent and the charge density of the solute.
        '''
        from pyscf.solvent.grad.pcm import grad_qv, grad_nuc, grad_solver
        de_solvent = grad_qv(self, dm)
        de_solvent+= grad_nuc(self, dm)
        de_solvent+= grad_solver(self, dm)
        return de_solvent

    def Hessian(self, hess_method):
        raise DeprecationWarning('Use the make_hessian_object function from '
                                 'pyscf.solvent.hessian.pcm instead.')

    def hess(self, dm):
        '''Computes the Hessian for the energy associated with the solvent,
        including the derivatives of the solvent itsself and the interactions
        between the solvent and the charge density of the solute.
        '''
        from pyscf.solvent.hessian.pcm import (
            analytical_hess_nuc, analytical_hess_qv, analytical_hess_solver)
        de_solvent  =    analytical_hess_nuc(self, dm, verbose=self.verbose)
        de_solvent +=     analytical_hess_qv(self, dm, verbose=self.verbose)
        de_solvent += analytical_hess_solver(self, dm, verbose=self.verbose)
        return de_solvent

    def _B_dot_x(self, dms):
        if not self._intermediates:
            self.build()
        out_shape = dms.shape
        nao = dms.shape[-1]
        dms = dms.reshape(-1,nao,nao)

        K = self._intermediates['K']
        R = self._intermediates['R']
        v_grids = -self._get_v(dms)

        b = numpy.dot(R, v_grids.T)
        q = numpy.linalg.solve(K, b).T

        vK_1 = numpy.linalg.solve(K.T, v_grids.T)
        qt = numpy.dot(R.T, vK_1).T
        q_sym = (q + qt)/2.0

        vmat = self._get_vmat(q_sym)
        return vmat.reshape(out_shape)