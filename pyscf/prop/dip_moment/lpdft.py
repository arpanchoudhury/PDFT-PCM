#!/usr/bin/env python
# Copyright 2014-2025 The PySCF Developers. All Rights Reserved.
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
# Author: Helen Clifford <helenclifford@uchicago.edu>

from pyscf.lib import logger
from functools import reduce
from scipy import linalg
import numpy as np
from pyscf.data import nist
from pyscf import lib
from pyscf.grad import lpdft as lpdft_grad
from pyscf.prop.dip_moment.mcpdft import mcpdft_HellmanFeynman_dipole, get_guage_origin, nuclear_dipole
from pyscf.mcscf import mc1step
from pyscf.mcscf.df import _DFCASSCF
from pyscf.mcpdft.lpdft import _LPDFT

class ElectricDipole (lpdft_grad.Gradients):

    def kernel (self, unit='Debye', origin='Coord_Center', **kwargs):
        state = kwargs['state'] if 'state' in kwargs else self.state
        if state is None:
            raise NotImplementedError ('Dipole of PDFT state-average energy')
        self.state = state
        #self.level_shift = 1e-8
        #self.max_cycle = 300
        #self.conv_atol = 1e-10
        #self.conv_rtol = 1e-8
        mo = kwargs['mo'] if 'mo' in kwargs else self.base.mo_coeff
        ci = kwargs['ci'] if 'ci' in kwargs else self.base.ci
        if isinstance (ci, np.ndarray): ci = [ci]
        kwargs['ci'] = ci
        if ("feff1" not in kwargs) or ("feff2" not in kwargs):
            kwargs["feff1"], kwargs["feff2"] = self.get_otp_gradient_response(
                mo, ci, state
            )



        # DIAGNOSTIC ===========
        print(f"\n=== Gas phase L-PDFT dipole CI diagnostic ===")
        print(f"si_pdft:\n{self.base.si_pdft}")
        print(f"Is si_pdft identity?: {np.allclose(self.base.si_pdft, np.eye(len(ci)))}")
        print(f"ci[0] norm: {np.linalg.norm(ci[0]):.6e}")
        print(f"ci[1] norm: {np.linalg.norm(ci[1]):.6e}")
        # check if ci is rotated or not by comparing with e_mcscf
        print(f"e_mcscf: {self.base.e_mcscf}")
        print(f"e_states: {self.base.e_states}")
        print(f"==============================================\n")

        '''# add to SolvatedLPDFTDipole.kernel or directly in lpdft.py kernel
        print(f"type(self.base): {type(self.base)}")
        print(f"self.base.get_hcore norm: {np.linalg.norm(self.base.get_hcore()):.6e}")
        print(f"solvent.v norm: {np.linalg.norm(self.base.with_solvent.v):.6e}")

        # check what feff1 contains
        feff1 = kwargs.get('feff1', None)
        if feff1 is not None:
            print(f"feff1 norm: {np.linalg.norm(feff1):.6e}")
            # does feff1 already contain solvent.v?
            diff = feff1 - (feff1 - self.base.with_solvent.v)
            print(f"feff1 - (feff1 - solvent.v) norm: {np.linalg.norm(diff):.6e}")
            # check if solvent.v is inside feff1
            h0 = self.base.get_hcore()
            print(f"feff1 contains solvent.v?: "
                f"{np.linalg.norm(feff1 - self.base.with_solvent.v - (feff1 - self.base.with_solvent.v)):.6e}")'''
        #=======================
        conv, Lvec, bvec, Aop, Adiag = self.solve_lagrange (**kwargs)
        self.debug_lagrange (Lvec, bvec, Aop, Adiag, **kwargs)

        # DIAGNOSTIC ===============
        # add to gas phase dipole kernel after solve_lagrange
        # after solve_lagrange
        geff = bvec + Aop(Lvec)
        gorb_res, gci_res = self.unpack_uniq_var(geff)

        # gci_res is the stuck direction — what does it look like?
        print(f"gci_res norm: {np.linalg.norm(np.concatenate([c.ravel() for c in gci_res])):.6e}")
        for i, g in enumerate(gci_res):
            print(f"  gci_res[{i}] norm: {np.linalg.norm(g):.6e}")

        # what is Aop(gci_res_packed)?
        gci_res_packed = geff.copy()
        gci_res_packed[:self.ngorb] = 0.0  # zero out orbital part
        Aop_gci = Aop(gci_res_packed)
        print(f"Aop(gci_res) norm: {np.linalg.norm(Aop_gci):.6e}")
        print(f"ratio |Aop(gci_res)|/|gci_res|: {np.linalg.norm(Aop_gci)/np.linalg.norm(gci_res_packed):.6e}")

        # does gci_res contribute to LdotJnuc?
        LdotJnuc_res = self.get_LdotJnuc(gci_res_packed, origin='Coord_Center', **kwargs)
        print(f"LdotJnuc from residual direction: {LdotJnuc_res * nist.AU2DEBYE}")
        geff = bvec + Aop(Lvec)
        gorb_part, gci_part = self.unpack_uniq_var(geff)
        print(f"gas |gorb| final: {np.linalg.norm(gorb_part):.6e}")
        print(f"gas |gci|  final: {np.linalg.norm(np.concatenate([c.ravel() for c in gci_part])):.6e}")
        print(f"gas |geff| final: {np.linalg.norm(geff):.6e}")
        print(f"bvec norm: {np.linalg.norm(bvec):.6e}")
        print(f"convergence threshold: {max(self.conv_rtol*np.linalg.norm(bvec), self.conv_atol):.6e}")
        print(f"e_mcscf: {self.base.e_mcscf}")
        print(f"e_gap:   {self.base.e_mcscf - self.base.e_mcscf[0]}")
        print(f"sing_tol: {getattr(self, 'sing_tol_sasa', 1e-8)}")
        # ==============================

        ham_response = self.get_ham_response (origin=origin, **kwargs)

        LdotJnuc = self.get_LdotJnuc (Lvec, origin=origin, **kwargs)

        mol_dip = ham_response + LdotJnuc

        mol_dip = self.convert_dipole (ham_response, LdotJnuc, mol_dip, unit=unit)
        return mol_dip

    def convert_dipole (self, ham_response, LdotJnuc, mol_dip, unit='Debye'):
        i = self.state
        if unit.upper() == 'DEBYE':
            for x in [ham_response, LdotJnuc, mol_dip]: x *= nist.AU2DEBYE
        log = lib.logger.new_logger(self, self.verbose)
        log.note('Permanent Dipole Moment  (%s) : %9.5f, %9.5f, %9.5f', unit, *ham_response)

        log.note('L-PDFT PDM <{}|mu|{}>           {:>10} {:>10} {:>10}'.format(i,i,'X','Y','Z'))
        log.note('Hamiltonian Contribution (%s) : %9.5f, %9.5f, %9.5f', unit, *ham_response)
        log.note('Lagrange Contribution    (%s) : %9.5f, %9.5f, %9.5f', unit, *LdotJnuc)
        log.note('Permanent Dipole Moment  (%s) : %9.5f, %9.5f, %9.5f', unit, *mol_dip)
        return mol_dip

    def get_ham_response(self, state=None, verbose=None, mo=None,
            ci=None, origin='Coord_Center', **kwargs):
        if state is None: state   = self.state
        if verbose is None: verbose = self.verbose
        if mo is None: mo      = self.base.mo_coeff
        if ci is None: ci      = self.base.ci

        fcasscf = self.make_fcasscf (state)
        fcasscf.mo_coeff = mo
        fcasscf.ci = ci[state]

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



class _LPDFTDipole(_LPDFT):
    def dip_moment (self, unit='Debye', origin='Coord_Center', state=None):
        if not isinstance (self, mc1step.CASSCF):
            raise NotImplementedError ("CASCI-based PDFT dipole moments")
        elif getattr (self, 'frozen', None) is not None:
            raise NotImplementedError ("PDFT dipole moments with frozen orbitals")
        elif isinstance (self, _DFCASSCF):
            raise NotImplementedError ("PDFT dipole moments with density-fitting ERIs")
        if not lib.isinteger (state):
            raise RuntimeError ('Permanent dipole requires a single state')
        dip_obj =  ElectricDipole(self)
        mol_dipole = dip_obj.kernel (state=state, unit=unit, origin=origin)
        return mol_dipole
_LPDFT.dip_moment = _LPDFTDipole.dip_moment
