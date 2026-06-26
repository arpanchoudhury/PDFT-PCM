#!/usr/bin/env python
# Copyright 2014-2022 The PySCF Developers. All Rights Reserved.
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
from pyscf.lib import logger
from functools import reduce
from scipy import linalg
import numpy as np
from pyscf.data import nist
from pyscf import lib
from pyscf.grad import mcpdft as mcpdft_grad

def get_guage_origin(mol,origin):
    charges = mol.atom_charges()
    coords  = mol.atom_coords()
    mass    = mol.atom_mass_list()
    if isinstance(origin,str):
        if origin.upper() == 'COORD_CENTER':
            center = (0,0,0)
        elif origin.upper() == 'MASS_CENTER':
            center = mass.dot(coords) / mass.sum()
        elif origin.upper() == 'CHARGE_CENTER':
            center = charges.dot(coords)/charges.sum()
        else:
            raise RuntimeError ("Gauge origin is not recognized")
    elif isinstance(origin, tuple):
        center = origin
    else:
        raise RuntimeError ("Gauge origin must be a string or tuple")
    return center

# TODO: docstring?
def mcpdft_HellmanFeynman_dipole (mc, mo_coeff=None, ci=None, origin='Coord_Center'):
    if mo_coeff is None: mo_coeff = mc.mo_coeff
    if ci is None: ci = mc.ci
    if mc.frozen is not None:
        raise NotImplementedError

    mol = mc.mol
    ncore = mc.ncore
    ncas = mc.ncas
    nocc = ncore + ncas
    nelecas = mc.nelecas

    mo_core = mo_coeff[:,:ncore]
    mo_cas = mo_coeff[:,ncore:nocc]

    casdm1 = mc.fcisolver.make_rdm1(ci, ncas, nelecas)

    dm_core = np.dot(mo_core, mo_core.T) * 2
    dm_cas = reduce(np.dot, (mo_cas, casdm1, mo_cas.T))
    dm = dm_core + dm_cas

    center = get_guage_origin(mol,origin)
    with mol.with_common_orig(center):
        ao_dip = mol.intor_symmetric('int1e_r', comp=3)
    elec_term = -np.tensordot(ao_dip, dm).real
    return elec_term

def nuclear_dipole(mc,origin='Coord_Center'):
    '''Compute nuclear contribution wrt gauge origin of the dipole moment'''
    mol = mc.mol
    center = get_guage_origin(mol,origin)
    charges = mol.atom_charges()
    coords  = mol.atom_coords()
    coords -= center
    nucl_term = charges.dot(coords)
    return nucl_term

# TODO: docstring?
class ElectricDipole (mcpdft_grad.Gradients):

    def kernel (self, unit='Debye', origin='Coord_Center', **kwargs):
        ''' Cache the effective Hamiltonian terms so you don't have to
            calculate them twice
        '''
        print('Calculating PDFT dipole moment from original kernel')
        state = kwargs['state'] if 'state' in kwargs else self.state
        if state is None:
            raise NotImplementedError ('Dipole of PDFT state-average energy')
        self.state = state # Not the best code hygiene maybe
        #self.level_shift = 1e-8
        #self.max_cycle = 300
        #self.conv_atol = 1e-10
        #self.conv_rtol = 1e-8
        mo = kwargs['mo'] if 'mo' in kwargs else self.base.mo_coeff
        ci = kwargs['ci'] if 'ci' in kwargs else self.base.ci
        if isinstance (ci, np.ndarray): ci = [ci] # hack hack hack...
        kwargs['ci'] = ci
        if ('veff1' not in kwargs) or ('veff2' not in kwargs):
            kwargs['veff1'], kwargs['veff2'] = self.base.get_pdft_veff (mo,
                ci, incl_coul=True, paaa_only=True, state=state)

        conv, Lvec, bvec, Aop, Adiag = self.solve_lagrange (**kwargs)
        self.debug_lagrange (Lvec, bvec, Aop, Adiag, **kwargs)

        '''Lorb, Lci = self.unpack_uniq_var(Lvec)
        borb, bci = self.unpack_uniq_var(bvec)

        print(f"  state {state} Lorb norm: {np.linalg.norm(Lorb):.6f}")
        print(f"  state {state} Lci  norm: {np.linalg.norm(np.concatenate([c.ravel() for c in Lci])):.6f}")
        print(f"  state {state} borb norm: {np.linalg.norm(borb):.6f}")
        print(f"  state {state} bci  norm: {np.linalg.norm(np.concatenate([c.ravel() for c in bci])):.6f}")

        # flatten lists to vectors for dot products
        Lci_flat = np.concatenate([c.ravel() for c in Lci])
        bci_flat  = np.concatenate([c.ravel() for c in bci])
        Lorb_flat = Lorb.ravel()
        borb_flat = borb.ravel()

        print(f"  state {state} borb dot Lorb: {np.dot(borb_flat, Lorb_flat):.6f}")
        print(f"  state {state} bci  dot Lci:  {np.dot(bci_flat,  Lci_flat):.6f}")
        print(f"  state {state} total bvec dot Lvec: {np.dot(borb_flat, Lorb_flat) + np.dot(bci_flat, Lci_flat):.6f}")'''

        ham_response = self.get_ham_response (origin=origin, **kwargs)

        LdotJnuc = self.get_LdotJnuc (Lvec, origin=origin, **kwargs)


        #Lorb, Lci = self.unpack_uniq_var(Lvec)

        #print(f"  Lorb norm: {np.linalg.norm(Lorb)}")
        #print(f"  Lci  norm: {np.linalg.norm(Lci)}")

        '''# which is dominating LdotJnuc?
        # temporarily zero one out
        Lvec_orb_only = self.pack_uniq_var(Lorb, np.zeros_like(Lci))
        Lvec_ci_only  = self.pack_uniq_var(np.zeros_like(Lorb), Lci)

        LdotJnuc_orb = self.get_LdotJnuc(Lvec_orb_only, origin=origin, **kwargs)
        LdotJnuc_ci  = self.get_LdotJnuc(Lvec_ci_only,  origin=origin, **kwargs)

        print(f"  LdotJnuc orbital part (Debye): {LdotJnuc_orb * nist.AU2DEBYE}")
        print(f"  LdotJnuc CI part      (Debye): {LdotJnuc_ci  * nist.AU2DEBYE}")

        #ham_response = self.get_ham_response(origin=origin, **kwargs)
        #LdotJnuc = self.get_LdotJnuc(Lvec, origin=origin, **kwargs)

        #print(f"  state {state} ham_response (Debye): {ham_response * nist.AU2DEBYE}")
        #print(f"  state {state} LdotJnuc    (Debye): {LdotJnuc    * nist.AU2DEBYE}")


        #print(f"  bvec norm: {np.linalg.norm(bvec)}")
        #print(f"  Lvec norm: {np.linalg.norm(Lvec)}")
        #print(f"  bvec dot Lvec: {np.dot(bvec, Lvec)}")  # should be negative
        #print(f"  state {state} ham_response (Debye): {ham_response * nist.AU2DEBYE}")
        #print(f"  state {state} LdotJnuc    (Debye): {LdotJnuc    * nist.AU2DEBYE}")'''

        mol_dip = ham_response + LdotJnuc

        mol_dip = self.convert_dipole (ham_response, LdotJnuc, mol_dip, unit=unit)
        return mol_dip

    def convert_dipole (self, ham_response, LdotJnuc, mol_dip, unit='Debye'):
        i = self.state
        if unit.upper() == 'DEBYE':
            for x in [ham_response, LdotJnuc, mol_dip]: x *= nist.AU2DEBYE
        log = lib.logger.new_logger(self, self.verbose)
        log.note('SA-CASSCF PDM <{}|mu|{}>         {:>10} {:>10} {:>10}'.format(i,i,'X','Y','Z'))
        log.note('Permanent Dipole Moment  (%s) : %9.5f, %9.5f, %9.5f', unit, *ham_response)

        log.note('MC-PDFT PDM <{}|mu|{}>           {:>10} {:>10} {:>10}'.format(i,i,'X','Y','Z'))
        log.note('Hamiltonian Contribution (%s) : %9.5f, %9.5f, %9.5f', unit, *ham_response)
        log.note('Lagrange Contribution    (%s) : %9.5f, %9.5f, %9.5f', unit, *LdotJnuc)
        log.note('Permanent Dipole Moment  (%s) : %9.5f, %9.5f, %9.5f', unit, *mol_dip)
        return mol_dip

    def get_ham_response (self, state=None, verbose=None, mo=None,
            ci=None, origin='Coord_Center', **kwargs):
        if state is None: state   = self.state
        if verbose is None: verbose = self.verbose
        if mo is None: mo      = self.base.mo_coeff
        if ci is None: ci      = self.base.ci

        fcasscf = self.make_fcasscf (state)
        #fcasscf_test = self.make_fcasscf(state)
        #print(type(fcasscf_test))          
        # → <class 'pyscf.mcscf.mc1step.CASSCF'>  ← plain, no PCM

        #print(hasattr(fcasscf_test, 'with_solvent'))  
        # → True  ← attribute is there but dead code, not attached to anything  
        fcasscf.mo_coeff = mo
        fcasscf.ci = ci[state]

        #print('fcasscf.mo_coeff', fcasscf.mo_coeff)
        #print('fcasscf.ci', fcasscf.ci)
        elec_term = mcpdft_HellmanFeynman_dipole (fcasscf, mo_coeff=mo, ci=ci[state], origin=origin)
        nucl_term = nuclear_dipole(fcasscf, origin=origin)
        total = nucl_term + elec_term
        return total

    def get_LdotJnuc (self, Lvec, state=None, verbose=None,
            mo=None, ci=None, origin='Coord_Center', **kwargs):
        if state is None: state   = self.state
        if verbose is None: verbose = self.verbose
        if mo is None: mo      = self.base.mo_coeff
        if ci is None: ci      = self.base.ci[state]
        mc = self.base

        Lorb, Lci = self.unpack_uniq_var (Lvec)

        mol = mc.mol
        ncore = mc.ncore
        ncas = mc.ncas
        nocc = ncore + ncas
        nelecas = mc.nelecas

        mo_core = mo[:,:ncore]
        mo_cas  = mo[:,ncore:nocc]

        #New 'effective' MO coeffs incl contraction from Lag multipliers
        moL_coeff = np.dot (mo, Lorb)
        moL_core  = moL_coeff[:,:ncore]
        moL_cas   = moL_coeff[:,ncore:nocc]

        casdm1 = mc.fcisolver.make_rdm1(ci, ncas, nelecas)

        dmL_core = np.dot(moL_core, mo_core.T) * 2
        dmL_cas  = reduce(np.dot, (moL_cas, casdm1, mo_cas.T))
        dmL_core += dmL_core.T
        dmL_cas  += dmL_cas.T

        casdm1_transit, _ = mc.fcisolver.trans_rdm12 (Lci, ci, ncas, nelecas)
        casdm1_transit += casdm1_transit.transpose (1,0)

        dm_cas_transit = reduce(np.dot, (mo_cas, casdm1_transit, mo_cas.T))

        dm = dmL_core + dmL_cas + dm_cas_transit

        center = get_guage_origin(mol, origin)
        with mol.with_common_orig(center):
            ao_dip = mol.intor_symmetric('int1e_r', comp=3)
        mol_dip_L = -np.tensordot(ao_dip, dm).real

        return mol_dip_L

