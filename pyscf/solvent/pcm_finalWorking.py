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
import numpy as np
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


# Inject PCM to other methods
from pyscf import scf
from pyscf import mcscf
from pyscf import mp, ci, cc
from pyscf import tdscf
scf.hf.SCF.PCM    = scf.hf.SCF.PCM    = pcm_for_scf
mp.mp2.MP2.PCM    = mp.mp2.MP2.PCM    = pcm_for_post_scf
ci.cisd.CISD.PCM  = ci.cisd.CISD.PCM  = pcm_for_post_scf
cc.ccsd.CCSD.PCM  = cc.ccsd.CCSD.PCM  = pcm_for_post_scf
tdscf.rhf.TDBase.PCM = tdscf.rhf.TDBase.PCM = pcm_for_tdscf
mcscf.casci.CASCI.PCM = mcscf.casci.CASCI.PCM = pcm_for_casci
mcscf.mc1step.CASSCF.PCM = mcscf.mc1step.CASSCF.PCM = pcm_for_casscf

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
modified_Bondi[1] = 1.1/radii.BOHR      # modified version
PI = numpy.pi

# diagscale = 1.0694 from pcm_arrays.F90 line 37.
# Used for S matrix diagonal: S_ii = diagscale * sqrt(4π/area_i)
_DIAGSCALE = 1.0694

# ---------------------------------------------------------------------------
# Pauling radii — OpenMolcas ITypRad=2 (keyword PAULing in RF-input)
#
# Source: src/pcm_util/solvent_data.F90, parameter prad(104)
# Units:  Angstrom.  Zero means "no Pauling radius defined for this element".
#
# Algorithm (fndsph.F90, case 2):
#   for every atom i (including H):
#       nord(i) = i
#       rad(i)  = pauling(Z_i)      ! prad(Z_i) Å
#   alpha   = 1.2  (= rslpar(9) default in pcmdef.F90)
#   nsinit  = nat                   ! ALL atoms get spheres
#   rad     = rad * alpha           ! scale by alpha
#
# Key differences from UATM:
#   - H atoms DO get their own sphere (unlike UATM where H is merged)
#   - Radius is a plain per-element lookup — no topology dependence
# ---------------------------------------------------------------------------

# prad[Z] = Pauling radius in Å for element Z (1-indexed; prad[0] unused).
# Directly transcribed from solvent_data.F90 prad(104) parameter array.
# Zero entries: no Pauling radius defined → fallback to modified_Bondi.

_PRAD_ANG = numpy.array([
    # idx  Z   element
    0.00,  # 0  (unused padding)
    1.20,  # 1  H
    1.20,  # 2  He
    1.37,  # 3  Li
    1.45,  # 4  Be
    1.45,  # 5  B
    1.50,  # 6  C
    1.50,  # 7  N
    1.40,  # 8  O
    1.35,  # 9  F
    1.30,  # 10 Ne
    1.57,  # 11 Na
    1.36,  # 12 Mg
    1.24,  # 13 Al
    1.17,  # 14 Si
    1.90,  # 15 P
    1.85,  # 16 S
    1.80,  # 17 Cl
    1.88,  # 18 Ar
    2.75,  # 19 K
    0.00,  # 20 Ca
    # 21–29  Sc..Cu  — no Pauling radius in source
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
    1.63,  # 30 Zn
    1.40,  # 31 Ga
    1.39,  # 32 Ge
    1.87,  # 33 As
    1.86,  # 34 Se
    2.00,  # 35 Br
    2.00,  # 36 Kr
    1.95,  # 37 Rb
    2.02,  # 38 Sr
    # 39–49  Y..In   — no Pauling radius in source
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
    1.63,  # 50 Sn
    1.72,  # 51 Sb
    1.58,  # 52 Te
    1.93,  # 53 I
    2.17,  # 54 Xe
    2.20,  # 55 Cs
    2.20,  # 56 Ba
    2.15,  # 57 La
    2.16,  # 58 Ce
    # 59–78  Pr..Pt  — no Pauling radius in source (20 zeros)
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
    # 79–81  Au, Hg, Tl — no Pauling radius in source
    0.00, 0.00, 0.00,
    1.72,  # 82 Pb
    1.66,  # 83 Bi
    1.55,  # 84 Po
    1.96,  # 85 At
    1.02,  # 86 Rn
    # 87–95  Fr..Am  — no Pauling radius in source (9 zeros)
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
    1.86,  # 96 Cm
    # 97–104 Bk..Rf  — no Pauling radius in source (8 zeros)
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
])  # shape (105,), index 0 unused; prad[Z] for Z = 1..104



def pauling_radii(mol, alpha=1.2):
    '''
    Compute OpenMolcas Pauling cavity radii (bohr) for every atom.

    Exact translation of fndsph.F90 case(2):
        rad(i) = pauling(ian(i))   [Å, from prad table]
        alpha  = 1.2               [rslpar(9) default in pcmdef.F90]
        rad    = rad * alpha        [applied after all assignments]

    Unlike UATM:
      - H atoms receive their own sphere.
      - Radius depends only on atomic number Z, not on bonding topology.
      - When prad[Z]=0 (element not parametrised), falls back to
        modified_Bondi[Z] (PySCF's default vdW table) so that the
        function never returns zero.

    Parameters
    ----------
    mol   : pyscf.gto.Mole
    alpha : float   scaling factor (default 1.2 = rslpar(9) in pcmdef.F90)

    Returns
    -------
    R : ndarray (natm,)  radii in bohr
    '''
    from pyscf.data.elements import charge as charge_of_element

    elem_z = numpy.array([charge_of_element(e) for e in mol.elements])
    R      = numpy.zeros(mol.natm)

    for ia in range(mol.natm):
        z = int(elem_z[ia])
        # Primary: OpenMolcas prad table (Å)
        if 1 <= z < len(_PRAD_ANG) and _PRAD_ANG[z] > 0.0:
            base_ang = _PRAD_ANG[z]
        # Fallback: PySCF modified_Bondi (already in bohr → convert to Å)
        elif z < len(modified_Bondi) and modified_Bondi[z] > 0.0:
            base_ang = float(modified_Bondi[z]) * radii.BOHR   # bohr → Å
        else:
            base_ang = 2.0 * radii.BOHR                         # 2 bohr → Å
        R[ia] = alpha * base_ang / radii.BOHR                   # Å → bohr

    return R


def _rad_lookup(rad, z):
    '''
    Safe radius lookup that works whether rad is a numpy array or a dict.
    Falls back to modified_Bondi, then 2.0 bohr.
    '''
    try:
        val = rad[z]
        if val > 0:
            return float(val)
    except (IndexError, KeyError, TypeError):
        pass
    # fallback
    try:
        val = modified_Bondi[z]
        if val > 0:
            return float(val)
    except (IndexError, KeyError, TypeError):
        pass
    return 2.0

# =============================================================================
# OpenMolcas-style cavity construction & tessellation
# Translated from src/pcm_util/:
#   polygen.F90   – geodesic spherical polyhedron (Pomelli algorithm)
#   inter_pcm.F90 – arc / sphere intersection (bisection)
#   tessera.F90   – tessera clipping against neighbouring spheres
#   gaubon.F90    – Gauss-Bonnet area + representative point
#
# Reference:
#   C. S. Pomelli, J. Tomasi, V. Barone, Theor. Chem. Acc. 105 (2001) 446.
# =============================================================================

# ---------------------------------------------------------------------------
# Seed polyhedra: unit-sphere vertex coordinates and face connectivity
# (hardcoded from polygen.F90)
# ---------------------------------------------------------------------------

# Icosahedron (20 faces, 12 vertices)
# Vertex coordinates: polygen.F90 icove(3,12) — hardcoded, NOT computed.
# Using different vertex coordinates produces a different cavity orientation
# and different tessera positions even with the same algorithm.
_ICOSA_VERTS = np.array([
    [ 0.000000000, 0.000000000, 1.000000000],
    [ 0.276393202, 0.850650808, 0.447213595],
    [-0.723606798, 0.525731112, 0.447213595],
    [-0.723606798,-0.525731112, 0.447213595],
    [ 0.276393202,-0.850650808, 0.447213595],
    [ 0.894427191, 0.000000000, 0.447213595],
    [-0.276393202, 0.850650808,-0.447213595],
    [-0.894427191, 0.000000000,-0.447213595],
    [-0.276393202,-0.850650808,-0.447213595],
    [ 0.723606798,-0.525731112,-0.447213595],
    [ 0.723606798, 0.525731112,-0.447213595],
    [ 0.000000000, 0.000000000,-1.000000000],
])  # shape (12, 3), all on unit sphere

# Face connectivity: polygen.F90 icotrv(3,20) — vertex indices (1-based in Fortran → 0-based here)
_ICOSA_FACES = np.array([
    [ 0, 1, 2], [ 0, 1, 5], [ 0, 2, 3], [ 0, 3, 4], [ 0, 4, 5],
    [ 1, 2, 6], [ 1, 5,10], [ 1, 6,10], [ 2, 3, 7], [ 2, 6, 7],
    [ 3, 4, 8], [ 3, 7, 8], [ 4, 5, 9], [ 4, 8, 9], [ 5, 9,10],
    [ 6, 7,11], [ 6,10,11], [ 7, 8,11], [ 8, 9,11], [ 9,10,11],
], dtype=int)  # shape (20, 3)

# Pentakisdodecahedron (60 faces, 32 vertices) – vertex coords from polygen.F90
_PENTAK_VERTS = np.array([
    [ 0.000000000, 0.000000000, 1.000000000],
    [ 0.491123473, 0.356822090, 0.794654472],
    [-0.187592474, 0.577350269, 0.794654472],
    [-0.607061998, 0.000000000, 0.794654472],
    [-0.187592474,-0.577350269, 0.794654472],
    [ 0.491123473,-0.356822090, 0.794654472],
    [ 0.894427191, 0.000000000, 0.447213595],
    [ 0.276393202, 0.850650808, 0.447213595],
    [-0.723606798, 0.525731112, 0.447213595],
    [-0.723606798,-0.525731112, 0.447213595],
    [ 0.276393202,-0.850650808, 0.447213595],
    [ 0.794654472, 0.577350269, 0.187592474],
    [-0.303530999, 0.934172359, 0.187592474],
    [-0.982246946, 0.000000000, 0.187592474],
    [-0.303530999,-0.934172359, 0.187592474],
    [ 0.794654472,-0.577350269, 0.187592474],
    [ 0.982246946, 0.000000000,-0.187592474],
    [ 0.303530999, 0.934172359,-0.187592474],
    [-0.794654472, 0.577350269,-0.187592474],
    [-0.794654472,-0.577350269,-0.187592474],
    [ 0.303530999,-0.934172359,-0.187592474],
    [ 0.723606798, 0.525731112,-0.447213595],
    [-0.276393202, 0.850650808,-0.447213595],
    [-0.894427191, 0.000000000,-0.447213595],
    [-0.276393202,-0.850650808,-0.447213595],
    [ 0.723606798,-0.525731112,-0.447213595],
    [ 0.607061998, 0.000000000,-0.794654472],
    [ 0.187592474, 0.577350269,-0.794654472],
    [-0.491123473, 0.356822090,-0.794654472],
    [-0.491123473,-0.356822090,-0.794654472],
    [ 0.187592474,-0.577350269,-0.794654472],
    [ 0.000000000, 0.000000000,-1.000000000],
])

def _build_pentak_faces():
    v = _PENTAK_VERTS
    dists = {}
    for i in range(len(v)):
        for j in range(i+1, len(v)):
            dists[(i,j)] = np.linalg.norm(v[i]-v[j])
    min_d = min(dists.values())
    tol   = min_d * 0.05
    faces = []
    for i in range(len(v)):
        for j in range(i+1, len(v)):
            if dists[(i,j)] > min_d + tol: continue
            for k in range(j+1, len(v)):
                if dists[(i,k)] > min_d + tol: continue
                if dists[(j,k)] > min_d + tol: continue
                faces.append([i, j, k])
    return np.array(faces)

_PENTAK_FACES = _build_pentak_faces()   # (60, 3)

# Tetrahedron (4 faces, 4 vertices)
_s3 = 1.0 / 3.0**0.5
_TETRA_VERTS = np.array([
    [-_s3,  _s3,  _s3],
    [ _s3, -_s3,  _s3],
    [-_s3, -_s3, -_s3],
    [ _s3,  _s3, -_s3],
])
_TETRA_FACES = np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]])


# ---------------------------------------------------------------------------
# polygen.F90 – geodesic subdivision
# ---------------------------------------------------------------------------

def _geodesic_subdivide(verts, faces, nf):
    '''
    Subdivide each triangular face of a seed polyhedron into nf^2 smaller
    triangles using great-circle interpolation, projected onto the unit sphere.
    Direct translation of the core loop in polygen.F90.

    Returns
    -------
    cv  : (N,3)  all vertices on the unit sphere
    jtr : (M,3)  vertex indices of every sub-triangle (0-based)
    '''
    cv = list(verts)
    edge_map = {}   # frozenset({ia,ib}) → list of nf+1 vertex indices

    def _edge_pts(ia, ib):
        v1 = np.array(cv[ia])
        v2 = np.array(cv[ib])
        ct = np.clip(np.dot(v1, v2), -1.0, 1.0)
        th = np.arccos(ct)
        st = np.sin(th)
        idx_list = [ia]
        for l in range(1, nf):
            if st < 1e-12:
                pt = v1.copy()
            else:
                c1 = np.cos(th * l / nf)
                c2 = np.cos(th * (nf - l) / nf)
                al = (c1 - ct * c2) / st**2
                be = (c2 - ct * c1) / st**2
                pt = al * v1 + be * v2
                pt /= np.linalg.norm(pt)
            cv.append(pt.copy())
            idx_list.append(len(cv) - 1)
        idx_list.append(ib)
        return idx_list   # length nf+1

    for face in faces:
        for k in range(3):
            ia, ib = int(face[k]), int(face[(k+1) % 3])
            key = frozenset({ia, ib})
            if key not in edge_map:
                edge_map[key] = _edge_pts(ia, ib)

    jtr = []
    for face in faces:
        ia, ib, ic = int(face[0]), int(face[1]), int(face[2])

        def _row(va, vb):
            lst = edge_map[frozenset({va, vb})]
            return lst if lst[0] == va else lst[::-1]

        e_ab = _row(ia, ib)   # ia→ib, nf+1 pts
        e_ac = _row(ia, ic)   # ia→ic, nf+1 pts
        e_bc = _row(ib, ic)   # ib→ic, nf+1 pts

        # Build triangular index grid oldtr[i][j], i=0..nf, j=0..i
        oldtr = [[None]*(nf+2) for _ in range(nf+2)]
        for l in range(nf+1):
            oldtr[l][0] = e_ab[l]
            oldtr[l][l] = e_ac[l]
            oldtr[nf][l] = e_bc[l]

        # Interior vertices (rows 2..nf-1)
        for l in range(2, nf):
            v1 = np.array(cv[e_ab[l]])
            v2 = np.array(cv[e_ac[l]])
            ct = np.clip(np.dot(v1, v2), -1.0, 1.0)
            th = np.arccos(ct)
            st = np.sin(th)
            for m in range(1, l):
                if st < 1e-12:
                    pt = v1.copy()
                else:
                    c1 = np.cos(th * m / l)
                    c2 = np.cos(th * (l-m) / l)
                    al = (c1 - ct*c2) / st**2
                    be = (c2 - ct*c1) / st**2
                    pt = al*v1 + be*v2
                    pt /= np.linalg.norm(pt)
                cv.append(pt.copy())
                oldtr[l][m] = len(cv) - 1

        # Emit nf^2 triangles
        for i in range(nf):
            for j in range(i+1):
                jtr.append([oldtr[i][j], oldtr[i+1][j], oldtr[i+1][j+1]])
        for i in range(1, nf):
            for j in range(i):
                jtr.append([oldtr[i][j], oldtr[i][j+1], oldtr[i+1][j+1]])

    return np.array(cv), np.array(jtr, dtype=int)


def _polygen(tsnum, centre, radius, pflag=0, tsare=0.4):
    '''
    Translation of polygen.F90.
    Choose best seed polyhedron + subdivision frequency, subdivide, scale.

    Parameters
    ----------
    tsnum  : int    target tessera count (used when pflag=0)
    centre : (3,)   sphere centre in bohr
    radius : float  sphere radius in bohr
    pflag  : int    0 = use tsnum directly,
                    1 = derive tsnum from target area tsare (pcmdef default)
    tsare  : float  target tessera area in Å²  (used when pflag=1).
                    Default 0.4 Å², matching rslpar(7)=0.4 in pcmdef.F90
                    and the OpenMolcas output "Average area ... 0.4000 angstrom^2".

    Returns
    -------
    cv  : (N,3)  vertex positions on this sphere (bohr)
    jtr : (M,3)  triangle face connectivity (0-based indices into cv)
    '''
    # area-driven mode was handled by gen_surface before calling here;
    # pflag=1 branch is kept for compatibility but gen_surface now always
    # calls with pflag=0 and a pre-computed tsnum in Å units.
    if pflag == 1:
        # tsare in Å², radius in Å — same units as polygen.F90
        tsnum = max(4, int(4.0 * PI * radius**2 / tsare + 0.5))

    nti, ntp, ntt = 20, 60, 4

    nfi = max(1, int((tsnum / nti)**0.5 + 0.5))
    nfp = max(1, int((tsnum / ntp)**0.5 + 0.5))
    nft = max(1, int((tsnum / ntt)**0.5 + 0.5))

    ndi = abs(tsnum - nti * nfi**2)
    ndp = abs(tsnum - ntp * nfp**2)
    ndt = abs(tsnum - ntt * nft**2)

    if ndp <= ndi and ndp <= ndt:
        seed_v, seed_f, nf = _PENTAK_VERTS, _PENTAK_FACES, nfp
    elif ndi <= ndt:
        seed_v, seed_f, nf = _ICOSA_VERTS,  _ICOSA_FACES,  nfi
    else:
        seed_v, seed_f, nf = _TETRA_VERTS,  _TETRA_FACES,  nft

    if nf == 1:
        cv  = seed_v.copy()
        jtr = seed_f.copy()
    else:
        cv, jtr = _geodesic_subdivide(seed_v, seed_f, nf)

    cv = cv * radius + centre
    return cv, jtr


# ---------------------------------------------------------------------------
# inter_pcm.F90 – arc / sphere boundary intersection (bisection)
# ---------------------------------------------------------------------------

def _inter_pcm(xe, ye, ze, re, p1, p2, p3, flag):
    '''
    Find where the great-circle arc p1→p2 (on a sphere centred at p3)
    crosses the surface of sphere (xe,ye,ze,re).
    flag=0: p1 outside, p2 inside.  flag=1: p1 inside, p2 outside.
    Returns intersection point p4.

    Exact translation of inter_pcm.F90:
      - TOL = 1e-12 (same units as input; bohr here)
      - ALPHA starts at 0.5, DELTA accumulates ±1/2^(M+1)
      - Exits when abs(|p4 - sphere_centre| - radius) < TOL
      - Max 100 iterations
    '''
    centre_cut = np.array([xe, ye, ze])
    r2 = np.dot(p1 - p3, p1 - p3)
    r = np.sqrt(r2)

    # TOL = 1e-12 (same as inter_pcm.F90 line 23: parameter TOL = 1.0e-12_wp)
    # Since we now run tessellation in Å (matching OpenMolcas), the tolerance
    # is applied directly in Å, matching the Fortran identically.
    TOL = 1.0e-12

    alpha = 0.5
    delta = 0.0
    m = 1
    p4 = p1.copy()
    for _ in range(100):
        alpha = alpha + delta
        v = p1 + alpha * (p2 - p1) - p3
        dnorm = np.linalg.norm(v)
        if dnorm < 1e-300:
            dnorm = 1e-300
        p4 = v * (r / dnorm) + p3
        diff = np.linalg.norm(p4 - centre_cut) - re
        if abs(diff) < TOL:
            break
        if flag == 0:
            delta = (1.0 / 2**(m + 1)) if diff > 0.0 else -(1.0 / 2**(m + 1))
        else:
            delta = -(1.0 / 2**(m + 1)) if diff > 0.0 else (1.0 / 2**(m + 1))
        m += 1
    return p4


# ---------------------------------------------------------------------------
# gaubon.F90 – Gauss-Bonnet area and representative point
# ---------------------------------------------------------------------------

def _gaubon(ns, atom_coords, atom_radii, intsph, pts, ccc):
    '''
    Compute tessera area via the Gauss-Bonnet theorem:
        Area = R^2 * [2π + Σ φ_n cosθ_n  − Σ β_n]
    and the representative point (normalised centroid on sphere surface).
    Translation of gaubon.F90.
    '''
    nv = len(pts)
    if nv < 3:
        return np.zeros(3), 0.0

    re_ns = atom_radii[ns]

    # sum1 = Σ φ_n cos θ_n
    sum1 = 0.0
    for n in range(nv):
        u1 = pts[n]       - ccc[n]
        u2 = pts[(n+1)%nv] - ccc[n]
        cos_phi = np.clip(np.dot(u1, u2) /
                          (np.linalg.norm(u1)*np.linalg.norm(u2)), -1.0, 1.0)
        phi_n = np.arccos(cos_phi)

        sp = intsph[n]
        ax = atom_coords[sp] - atom_coords[ns]
        ax_n = np.linalg.norm(ax)
        if ax_n < 1e-30: ax_n = 1.0
        rv = pts[n] - atom_coords[ns]
        cos_t = np.dot(ax, rv) / (ax_n * np.linalg.norm(rv))
        sum1 += phi_n * cos_t

    # sum2 = Σ exterior angles β_n  (via double cross-product tangent vectors)
    sum2 = 0.0
    for n in range(nv):
        n0 = (n-1) % nv
        n2 = (n+1) % nv

        # incoming tangent at vertex n
        p1v = pts[n]  - ccc[n0]
        p2v = pts[n0] - ccc[n0]
        t1  = np.cross(p1v, np.cross(p1v, p2v))
        d1  = np.linalg.norm(t1)
        if d1 < 1e-35: d1 = 1.0
        u1  = t1 / d1

        # outgoing tangent at vertex n
        p1v = pts[n]  - ccc[n]
        p2v = pts[n2] - ccc[n]
        t2  = np.cross(p1v, np.cross(p1v, p2v))
        d2  = np.linalg.norm(t2)
        if d2 < 1e-35: d2 = 1.0
        u2  = t2 / d2

        beta_n = np.arccos(np.clip(np.dot(u1, u2), -1.0, 1.0))
        sum2  += (PI - beta_n)

    area = re_ns**2 * (2.0*PI + sum1 - sum2)
    if area < 0.0:
        area = 0.0

    centroid = np.mean(np.array(pts), axis=0) - atom_coords[ns]
    d = np.linalg.norm(centroid)
    if d < 1e-30: d = 1.0
    pp = atom_coords[ns] + centroid * re_ns / d
    return pp, area


# ---------------------------------------------------------------------------
# tessera.F90 – clip one triangle against all other spheres
# ---------------------------------------------------------------------------

_SMALL_TS = 1e-12
_TOL_TS   = -1e-10

def _tessera(ns, atom_coords, atom_radii, pts_in, ccc_in, intsph_in):
    '''
    Clip the triangular tessera (on sphere ns) against every other sphere.
    Returns clipped polygon, representative point, area, and validity flag.
    Translation of tessera.F90.
    '''
    pts    = [np.array(p) for p in pts_in]
    ccc    = [np.array(c) for c in ccc_in]
    intsph = list(intsph_in)

    xe = atom_coords[:, 0]
    ye = atom_coords[:, 1]
    ze = atom_coords[:, 2]
    re = atom_radii

    for nsfe1 in range(len(atom_coords)):
        if nsfe1 == ns:
            continue
        nv     = len(pts)
        intscr = intsph[:]
        pscr   = [p.copy() for p in pts]
        cccp   = [c.copy() for c in ccc]

        # Mark vertices buried inside sphere nsfe1
        ind = np.array([
            1 if np.linalg.norm(p - atom_coords[nsfe1]) < re[nsfe1] - _SMALL_TS
            else 0
            for p in pts
        ])
        if ind.sum() == nv:          # fully buried → discard
            return [], [], [], None, 0.0, False

        # Classify each edge
        ltyp   = np.zeros(nv, dtype=int)
        pointl = [None]*nv           # intermediate buried point (ltyp=3)

        for l in range(nv):
            iv1, iv2 = l, (l+1) % nv
            a, b = ind[iv1], ind[iv2]
            if   a == 1 and b == 1: ltyp[l] = 0
            elif a == 0 and b == 1: ltyp[l] = 1
            elif a == 1 and b == 0: ltyp[l] = 2
            else:
                ltyp[l] = 4
                # probe for mid-edge burial (doubly-cut edge, ltyp=3)
                rc = np.linalg.norm(ccc[l] - pts[l])
                for ii in range(1, 12):
                    pt = pts[iv1] + ii*(pts[iv2]-pts[iv1])/11.0 - ccc[l]
                    pt = pt * (rc / np.linalg.norm(pt)) + ccc[l]
                    if np.linalg.norm(pt - atom_coords[nsfe1]) - re[nsfe1] < _TOL_TS:
                        ltyp[l]   = 3
                        pointl[l] = pt.copy()
                        break

        # Discard fragmented tessera (more than one cut region)
        icut = sum(1 if ltyp[l] in (1,2) else 2 if ltyp[l]==3 else 0
                   for l in range(nv)) // 2
        if icut > 1:
            return [], [], [], None, 0.0, False

        # Recompute intersection-circle centre for arcs on the nsfe1 boundary
        de2_vec = atom_coords[nsfe1] - atom_coords[ns]
        de2     = np.dot(de2_vec, de2_vec)
        fac     = (re[ns]**2 - re[nsfe1]**2 + de2) / (2.0*de2)
        circ_c  = atom_coords[ns] + fac * de2_vec   # circle centre

        new_pts, new_ccc, new_intsph = [], [], []

        for l in range(nv):
            if ltyp[l] == 0:
                continue
            iv1, iv2 = l, (l+1) % nv

            if ltyp[l] == 1:           # exposed→buried: keep v1, add entry crossing
                new_pts.append(pscr[iv1].copy())
                new_ccc.append(cccp[iv1].copy())
                new_intsph.append(intscr[iv1])
                p4 = _inter_pcm(xe[nsfe1], ye[nsfe1], ze[nsfe1], re[nsfe1],
                                 pscr[iv1], pscr[iv2], cccp[iv1], 0)
                new_pts.append(p4);  new_ccc.append(circ_c.copy())
                new_intsph.append(nsfe1)

            elif ltyp[l] == 2:         # buried→exposed: add exit crossing only
                p4 = _inter_pcm(xe[nsfe1], ye[nsfe1], ze[nsfe1], re[nsfe1],
                                 pscr[iv1], pscr[iv2], cccp[iv1], 1)
                new_pts.append(p4)
                new_ccc.append(cccp[iv1].copy())
                new_intsph.append(intscr[iv1])

            elif ltyp[l] == 3:         # doubly cut: keep v1, entry, exit
                new_pts.append(pscr[iv1].copy())
                new_ccc.append(cccp[iv1].copy())
                new_intsph.append(intscr[iv1])
                p4a = _inter_pcm(xe[nsfe1], ye[nsfe1], ze[nsfe1], re[nsfe1],
                                  pscr[iv1], pointl[l], cccp[iv1], 0)
                new_pts.append(p4a); new_ccc.append(circ_c.copy())
                new_intsph.append(nsfe1)
                p4b = _inter_pcm(xe[nsfe1], ye[nsfe1], ze[nsfe1], re[nsfe1],
                                  pointl[l], pscr[iv2], cccp[iv1], 1)
                new_pts.append(p4b)
                new_ccc.append(cccp[iv1].copy())
                new_intsph.append(intscr[iv1])

            elif ltyp[l] == 4:         # fully exposed: keep v1 as-is
                new_pts.append(pscr[iv1].copy())
                new_ccc.append(cccp[iv1].copy())
                new_intsph.append(intscr[iv1])

            if len(new_pts) > 11:
                return [], [], [], None, 0.0, False

        pts, ccc, intsph = new_pts, new_ccc, new_intsph

    # Compute area and representative point via Gauss-Bonnet
    pp, area = _gaubon(ns, atom_coords, atom_radii, intsph, pts, ccc)
    if area <= 0.0:
        return [], [], [], None, 0.0, False

    return pts, ccc, intsph, pp, area, True


# ---------------------------------------------------------------------------
# gen_surface – top-level surface builder (replaces PySCF Lebedev version)
# Produces a surface dict with all keys expected by get_F_A, get_D_S,
# _get_vind, _get_v, _get_vmat (and the grad/hessian modules).
# ---------------------------------------------------------------------------

def gen_surface(mol, tsnum=None, tsare=0.4, pflag=1, rad=None, use_uatm=False):
    '''
    OpenMolcas-style PCM cavity: geodesic polyhedron tessellation per atom,
    clipped against all neighbouring spheres (polygen + tessera + gaubon).

    Reference: C.S. Pomelli, J. Tomasi, V. Barone,
               Theor. Chem. Acc. 105 (2001) 446.

    Parameters
    ----------
    mol      : pyscf.gto.Mole
    tsnum    : int or None
        Target tessera count per sphere (pflag=0). Ignored when pflag=1.
    tsare    : float
        Target tessera area per sphere in Å² (pflag=1).
        Default: 0.4 Å², matching rslpar(7) in pcmdef.F90 and the
        OpenMolcas output "Average area ... 0.4000 angstrom^2".
    pflag    : int
        0 = fix count to tsnum.
        1 = derive count from tsare per sphere (OpenMolcas default).
    rad      : array-like or None
        Optional explicit radii table (bohr) indexed by nuclear charge.
        When None (default), Pauling radii from prad table are used.
    use_uatm : bool
        Ignored. Kept for API compatibility only.

    Returns
    -------
    surface : dict  compatible with all downstream PySCF PCM routines
    '''
    from pyscf.data.elements import charge as charge_of_element

    atom_coords_bohr = mol.atom_coords(unit='B')
    natm        = mol.natm
    elem_z      = numpy.array([charge_of_element(e) for e in mol.elements])

    # ---- Pauling radii (fndsph.F90 case 2) ----
    if rad is None:
        R_J_bohr = pauling_radii(mol)   # bohr
    else:
        R_J_bohr = numpy.array([_rad_lookup(rad, int(z)) for z in elem_z])

    # -----------------------------------------------------------------------
    # fndtess.F90 lines 69-71: convert bohr → Å before all tessellation.
    # "PEDRA works with angstroms"
    # We mirror this exactly so all floating-point operations happen in Å,
    # matching OpenMolcas's rounding patterns.
    # -----------------------------------------------------------------------
    from pyscf.data import radii as _rad_data
    BOHR = _rad_data.BOHR                  # Å per bohr  (= angstrom constant)
    atom_coords = atom_coords_bohr * BOHR  # Å
    atom_radii  = R_J_bohr * BOHR         # Å

    all_coords   = []
    all_norm     = []
    all_area     = []
    all_rvdw     = []
    all_swf      = []
    all_xi       = []
    all_isph_list = []    # sphere index for each tessera (for too-close merging)
    gslice_by_atom = []
    p0 = 0

    for ia in range(natm):
        centre = atom_coords[ia]    # Å
        r_vdw  = atom_radii[ia]    # Å

        # Build geodesic polyhedron — tsare already in Å², radius in Å.
        # polygen.F90 line 161: tsnum = int(4π*ren²/tsare + 0.5)
        # This is the same formula we use; in Å units it's natural.
        if pflag == 1:
            _tsnum = max(4, int(4.0 * PI * r_vdw**2 / tsare + 0.5))
        else:
            _tsnum = tsnum if tsnum is not None else 400
        cv, jtr = _polygen(_tsnum, centre, r_vdw, pflag=0)   # pflag=0: use tsnum directly

        # Clip each triangle against all other spheres (all in Å)
        for tri in jtr:
            v0 = cv[tri[0]]; v1 = cv[tri[1]]; v2 = cv[tri[2]]

            pts_in    = [v0.copy(), v1.copy(), v2.copy()]
            ccc_in    = [centre.copy(), centre.copy(), centre.copy()]
            intsph_in = [ia, ia, ia]

            pts, ccc, intsph, pp, area_ang2, valid = _tessera(
                ia, atom_coords, atom_radii, pts_in, ccc_in, intsph_in)

            if not valid:
                continue

            # Outward unit normal: sphere centre → representative point (in Å)
            nvec = pp - centre
            nlen = numpy.linalg.norm(nvec)
            if nlen < 1e-30:
                continue
            nvec /= nlen

            all_coords.append(pp)          # Å
            all_norm.append(nvec)
            all_area.append(area_ang2)     # Å²
            all_rvdw.append(r_vdw)        # Å
            all_swf.append(1.0)
            all_xi.append((PI / area_ang2)**0.5)
            all_isph_list.append(ia)              # track which atom sphere

        p1 = len(all_coords)
        gslice_by_atom.append([p0, p1])
        p0 = p1

    if len(all_coords) == 0:
        raise RuntimeError('No valid tesserae generated. '
                           'Check atomic radii or tsnum.')

    # -----------------------------------------------------------------------
    # fndtess.F90 lines 252-279: discard the smaller of any two tesserae
    # from DIFFERENT spheres whose representative points are closer than
    # TEST = 0.02 Å.
    # At this point everything is in Å (matching OpenMolcas at this stage).
    # -----------------------------------------------------------------------
    _TEST_ANG  = 0.02
    _TEST2_ANG = _TEST_ANG ** 2

    all_coords_arr = numpy.array(all_coords)   # Å
    all_area_arr   = numpy.array(all_area)     # Å²
    all_isph       = numpy.array(all_isph_list)

    nts_raw = len(all_area_arr)

    for i in range(nts_raw - 1):
        if all_area_arr[i] == 0.0:
            continue
        ci = all_coords_arr[i]
        for j in range(i + 1, nts_raw):
            if all_area_arr[j] == 0.0:
                continue
            if all_isph[i] == all_isph[j]:     # same sphere: skip
                continue
            rij2 = numpy.dot(ci - all_coords_arr[j],
                             ci - all_coords_arr[j])
            if rij2 > _TEST2_ANG:
                continue
            # Too close: zero the smaller area
            if all_area_arr[i] < all_area_arr[j]:
                all_area_arr[i] = 0.0
                break
            else:
                all_area_arr[j] = 0.0

    # fndtess.F90 lines 288-306: remove tesserae with area < 1e-10 Å²
    _AREA_MIN_ANG2 = 1.0e-10   # Å²

    valid_mask = all_area_arr >= _AREA_MIN_ANG2

    all_coords_f = all_coords_arr[valid_mask]   # Å
    all_norm_f   = numpy.array(all_norm)[valid_mask]
    all_area_f   = all_area_arr[valid_mask]     # Å²
    all_rvdw_f   = numpy.array(all_rvdw)[valid_mask]   # Å
    all_swf_f    = numpy.array(all_swf)[valid_mask]

    # Rebuild gslice_by_atom after filtering
    gslice_by_atom_f = []
    cumcount = 0
    for ia in range(natm):
        old_p0, old_p1 = gslice_by_atom[ia]
        n_kept = int(valid_mask[old_p0:old_p1].sum())
        gslice_by_atom_f.append([cumcount, cumcount + n_kept])
        cumcount += n_kept

    if all_coords_f.shape[0] == 0:
        raise RuntimeError('No valid tesserae after filtering. '
                           'Check atomic radii or tsnum.')

    # -----------------------------------------------------------------------
    # fndtess.F90 lines 342-353: convert Å → bohr for all output quantities.
    # Rs /= Angstrom,  Xs,Ys,Zs /= Angstrom
    # At /= Angstrom²,  Xt,Yt,Zt /= Angstrom
    # Vert /= Angstrom,  Centr /= Angstrom
    # -----------------------------------------------------------------------
    grid_coords = all_coords_f / BOHR          # Å → bohr
    norm_vec    = all_norm_f                   # already unit vector, no unit
    area        = all_area_f / BOHR**2        # Å² → bohr²
    R_vdw       = all_rvdw_f / BOHR           # Å → bohr
    switch_fun  = all_swf_f
    charge_exp  = numpy.sqrt(PI / area)        # bohr^-1
    weights     = area / R_vdw**2             # dimensionless solid-angle weights

    surface = {
        'tsnum':           tsnum,
        'tsare':           tsare,
        'pflag':           pflag,
        'gslice_by_atom':  gslice_by_atom_f,
        'grid_coords':     grid_coords,
        'weights':         weights,
        'charge_exp':      charge_exp,
        'switch_fun':      switch_fun,
        'R_vdw':           R_vdw,
        'norm_vec':        norm_vec,
        'area':            area,
        'atom_coords':     atom_coords,
    }
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
    Build S and D matrices exactly as in OpenMolcas matpcm.F90.

    S matrix — plain Coulomb (NOT erf-screened), matpcm.F90 lines 41–54 / 79–87:
        S_ij = 1 / r_ij                              (i ≠ j)
        S_ii = diagscale * sqrt(4π / area_i)         diagscale = 1.0694

    D matrix — normal derivative of 1/r with OpenMolcas sign convention and
    solid-angle diagonal, matpcm.F90 lines 80–91:
        D_ij = −(rᵢ−rⱼ)·nᵢ / r_ij³                 (i ≠ j, normal at tessera i)
        D_ii  = −2π/aᵢ − Σ_{j≠i} D_ij·aⱼ/aᵢ        (solid-angle + conservation)

    Note on convention:  D_OM[i,j] = D_PySCF[j,i]   (OpenMolcas uses nᵢ with
    a sign flip versus the PySCF convention of nⱼ).  The IEF-PCM T and R
    matrices in build() are constructed directly from D_OM, so no conversion
    is needed.
    '''
    grid_coords = surface['grid_coords']
    norm_vec    = surface['norm_vec']    # outward unit normal nᵢ at each tessera
    area        = surface['area']        # tessera areas aᵢ in bohr²

    rij = scipy.spatial.distance.cdist(grid_coords, grid_coords)  # (N,N)
    numpy.fill_diagonal(rij, 1.0)   # avoid 1/0 on diagonal

    S = None
    if with_S:
        # Off-diagonal: 1/r (plain Coulomb)
        S = 1.0 / rij
        # Diagonal: diagscale * sqrt(4π/aᵢ)  — matpcm.F90 line 45 / 79
        numpy.fill_diagonal(S, _DIAGSCALE * numpy.sqrt(4.0 * PI / area))

    D = None
    if with_D:
        # rij_vec[i,j] = rᵢ − rⱼ   shape (N,N,3)
        rij_vec = numpy.expand_dims(grid_coords, 1) - grid_coords
        # dot_ni[i,j] = (rᵢ − rⱼ) · nᵢ
        dot_ni  = numpy.einsum('ijk,ik->ij', rij_vec, norm_vec)

        # Off-diagonal: D_ij = −dot_ni[i,j] / r_ij³   (matpcm.F90 line 89)
        D = -dot_ni / rij**3
        numpy.fill_diagonal(D, 0.0)

        # Diagonal by charge conservation (matpcm.F90 line 90):
        #   for each (its, jts≠its): D[jts,jts] −= D[its,jts] * a[its] / a[jts]
        # ⟹  D[j,j] = −Σ_{i≠j} D[i,j] * a[i] / a[j]
        diag_cons = -numpy.einsum('ij,i->j', D, area) / area

        # Solid-angle term (matpcm.F90 line 80):
        #   D[i,i] −= 2π / a[i]
        diag_sa = -2.0 * PI / area

        numpy.fill_diagonal(D, diag_cons + diag_sa)

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

    surface_discretization_method : str  (unused – kept for API compatibility only)
        Ignored. Sphere overlap is handled by exact geometric clipping
        (tessera.F90 algorithm) rather than a switching function.

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
        'mol', 'radii_table', 'tsnum', 'tsare',
        'eps', 'max_cycle', 'conv_tol', 'state_id', 'frozen',
        'equilibrium_solvation', 'e', 'v', 'v_grids_n',
        'conv_energy', 'es_method', 'refdm',
        'refidx', 'partition', 'rf_root'
    }

    #kernel = ddcosmo.DDCOSMO.kernel

    def __init__(self, mol):
        self.mol = mol
        self.stdout = mol.stdout
        self.verbose = mol.verbose
        self.max_memory = mol.max_memory
        self.method = 'C-PCM'

        self.vdw_scale = 1.2 # default value in qchem
        self.r_probe = 0.0
        self.radii_table = None
        # OpenMolcas defaults from pcmdef.F90 / fndsph.F90:
        #   islpar(9)  =  1 (UATM) → we use Pauling (islpar(9)=2) which is
        #                             the simplest reproducible model;
        #                             radii from prad[] in solvent_data.F90,
        #                             alpha=1.2 (rslpar(9)), H gets own sphere
        #   islpar(11) = -400       → pflag=1 (area-driven tessellation)
        #   rslpar(7)  =  0.4 Å²   → target tessera area
        self.tsnum = None     # ignored when tsare is used (pflag=1)
        self.tsare = 0.4      # Å² per tessera — matches rslpar(7)=0.4 Å²
        self.eps = 78.3553

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
        self.es_method = None
        self.refdm = None
        self.refidx = 0.0
        # Charge partition for nonequilibrium calculations:
        #   'pekar'  — Pekar partition (default, existing behaviour):
        #              q_in = q_total(GS) - q_dyn(GS)  using f(ε_opt)/f(ε) ratio
        #   'marcus' — Marcus partition (OpenMolcas / Partition I):
        #              Q̄^or = (ε-ε_opt)/(ε-1) · Q̄_total(GS)
        #              Q̄̄^el solved via D_{ε_opt}·Q̄̄^el = -b̄̄ - Ω·Q̄^or  (Ω = S)
        self.partition = 'pekar'
        self.rf_root = None


    def dump_flags(self, verbose=None):
        logger.info(self, '******** %s (In testing) ********', self.__class__)
        logger.warn(self, 'PCM is an experimental feature. It is '
                    'still in testing.\nFeatures and APIs may be changed '
                    'in the future.')
        logger.info(self, 'radii = Pauling (prad table, alpha=1.2, '
                    'fndsph.F90 case 2; H gets own sphere)')
        if self.tsnum is None:
            logger.info(self, 'tsare = %.4f Ang^2 (area-driven, '
                        'OpenMolcas default rslpar(7)=0.4)', self.tsare)
        else:
            logger.info(self, 'tsnum = %d (fixed tessera count per sphere)',
                        self.tsnum)
        logger.info(self, 'eps = %s'   , self.eps)
        logger.info(self, 'frozen = %s', self.frozen)
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
        mol = self.mol

        # Pauling radii (fndsph.F90 case 2): use radii_table only if
        # explicitly supplied by user, otherwise pauling_radii() is called
        # inside gen_surface via rad=None
        rad = self.radii_table   # None → Pauling; non-None → user override
        logger.debug2(self, 'radii_table = %s', rad)

        if self.tsnum is None:
            self.surface = gen_surface(mol, tsare=self.tsare, pflag=1,
                                       rad=rad)
        else:
            self.surface = gen_surface(mol, tsnum=self.tsnum, tsare=self.tsare,
                                       pflag=0, rad=rad)
        self._intermediates = {}
        _, S = get_D_S(self.surface, with_S=True, with_D=False)
        area = self.surface['area']           # tessera areas (bohr²)
        grid_coords = self.surface['grid_coords']
        atom_coords  = mol.atom_coords(unit='B')
        atom_charges = mol.atom_charges()

        epsilon = self.eps
        epsilon_opt = (self.refidx)**2

        print("self.state_id in build", self.state_id)
        if self.method.upper() in ['C-PCM', 'CPCM', 'COSMO']:
            if self.method.upper() == 'COSMO':
                f_epsilon = ((epsilon-1.)/(epsilon+0.5)
                             if epsilon != float('inf') else 1.0)
            else:
                f_epsilon = ((epsilon-1.)/epsilon
                             if epsilon != float('inf') else 1.0)

            K = S
            R = -f_epsilon * numpy.eye(K.shape[0])
            D = None

        elif self.method.upper() in ['IEF-PCM', 'IEFPCM', 'SS(V)PE']:
            _, D = get_D_S(self.surface, with_S=False, with_D=True)

            if self.method.upper() == 'SS(V)PE':
                fac = (epsilon + 1.0) / (epsilon - 1.0) if epsilon != float('inf') else 1.0
                SAD  = S @ (area[:, None] * D)
                SADT = S @ (area[None, :] * D.T)
                K = fac * S - (SAD + SADT) / (4.0 * PI)
            else:  # IEF-PCM
                fac = (epsilon + 1.0) / (epsilon - 1.0) if epsilon != float('inf') else 1.0
                SAD = S @ (area[:, None] * D)
                K = fac * S - SAD / (2.0 * PI)

            R = D.T * area / (2.0 * PI)
            numpy.fill_diagonal(R, numpy.diag(R) - 1.0)
            f_epsilon = (epsilon - 1.0) / (epsilon + 1.0) if epsilon != float('inf') else 1.0

        else:
            raise RuntimeError(f"Unknown implicit solvent model: {self.method}")

        intermediates = {
            'S': S,
            'D': D,
            'A': area,
            'K': K,
            'R': R,
            'f_epsilon': f_epsilon,
        }
        self._intermediates.update(intermediates)

        # ---------------------------------------------------------------
        # Nuclear electrostatic potential on surface tesserae
        # OpenMolcas: pure point charges  V_nuc(R_i) = Σ_A Z_A / |R_i − R_A|
        # (no Gaussian smearing — exact 1/r)
        # ---------------------------------------------------------------
        dists_nuc = scipy.spatial.distance.cdist(grid_coords, atom_coords)  # (ngrids, natm)
        self.v_grids_n = numpy.einsum('a,ia->i', atom_charges, 1.0 / dists_nuc)

    def _get_vind(self, dms):
        if not self._intermediates:
            self.build()

        nao = dms.shape[-1]
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

    def _get_K_opt_R_opt(self):
        '''
        Return K_opt and R_opt for the optical dielectric constant ε_opt = refidx².

        These matrices are NOT stored in build() because refidx is typically
        set after the HF/ground-state build (it is only needed for nonequilibrium
        excited-state calculations).  Instead they are computed on demand from
        the current self.refidx and cached in _intermediates under 'K_opt' and
        'R_opt'.  The cache is invalidated whenever refidx changes by checking
        the stored 'refidx_cached' value.

        For C-PCM:  K_opt = S,  R_opt = -f(ε_opt) · I
        For IEF-PCM: raises NotImplementedError (not yet needed).
        '''
        if not self._intermediates:
            self.build()

        refidx = self.refidx
        epsilon_opt = refidx ** 2

        # Use cached version if refidx hasn't changed
        if (self._intermediates.get('refidx_cached') == refidx
                and 'K_opt' in self._intermediates
                and 'R_opt' in self._intermediates):
            return self._intermediates['K_opt'], self._intermediates['R_opt']

        S = self._intermediates['S']

        if self.method.upper() in ['C-PCM', 'CPCM', 'COSMO']:
            if epsilon_opt <= 0.0:
                raise ValueError(
                    'refidx must be set before calling nonequilibrium methods. '
                    'Set mc.with_solvent.refidx = <refractive index> (e.g. 1.3328 for water).')
            f_epsilon_opt = (epsilon_opt - 1.0) / epsilon_opt
            K_opt = S
            R_opt = -f_epsilon_opt * numpy.eye(S.shape[0])
        elif self.method.upper() in ['IEF-PCM', 'IEFPCM', 'SS(V)PE']:
            raise NotImplementedError(
                'K_opt/R_opt for IEF-PCM nonequilibrium not yet implemented.')
        else:
            raise RuntimeError(f'Unknown method: {self.method}')

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

        b = numpy.dot(R_opt, v_grids.T)
        q = numpy.linalg.solve(K_opt, b).T

        vK_1 = numpy.linalg.solve(K_opt.T, v_grids.T)
        qt = numpy.dot(R_opt.T, vK_1).T
        q_sym = (q + qt)/2.0

        return q_sym, v_grids

    def _get_vind_marcus(self, dm_es, q_or):
        '''
        Marcus partition (Partition I / OpenMolcas) excited-state fast charges.

        Solves eq. 18 from the literature:
            D_{ε_opt} Q̄̄^el = -b̄̄ - Ω Q̄^or

        with Ω = S (plain Coulomb matrix, same S used for C-PCM).

        In C-PCM form:  K_opt · q_el = R_opt · V̄̄_mol - S · q_or
        where  R_opt = -f(ε_opt) · I  and  K_opt = S
        so:    S · q_el = -f(ε_opt) · V̄̄_mol - S · q_or
               q_el = -f(ε_opt) · S⁻¹ · V̄̄_mol  -  q_or

        Equivalently: add V_slow = S·q_or to V̄̄_mol, then solve with K_opt.

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

        # Molecular ESP from current ES density (no V_slow yet)
        v_grids_e = self._get_v(dms)
        v_grids_mol = self.v_grids_n - v_grids_e   # shape (1, ngrids)

        # V_slow = S · q_or  — potential of slow charges at tessera positions
        v_slow = S @ q_or                           # shape (ngrids,)

        # Total driving potential: V̄̄_mol + V_slow  (OpenMolcas line 167)
        v_grids_total = v_grids_mol.copy()
        v_grids_total[0] += v_slow

        # Solve D_{ε_opt} Q̄̄^el = -b̄̄ - Ω Q̄^or
        b = numpy.dot(R_opt, v_grids_total.T)       # -f(ε_opt)·(V_mol + V_slow)
        q = numpy.linalg.solve(K_opt, b).T

        vK_1 = numpy.linalg.solve(K_opt.T, v_grids_total.T)
        qt   = numpy.dot(R_opt.T, vK_1).T
        q_el = (q + qt) / 2.0                       # shape (1, ngrids)

        return q_el, v_grids_mol

    def _get_v(self, dms):
        '''
        Electrostatic potential of the electron density at each tessera centre.

        OpenMolcas uses exact 1/r molecular integrals.  We use PySCF's
        int1e_grids which computes <μ|1/|r−R_g||ν> exactly (no Gaussian
        smearing), matching the OpenMolcas point-charge model.

        Returns v_grids_e[set, ngrids]
        '''
        mol = self.mol
        grid_coords = self.surface['grid_coords']
        ngrids = grid_coords.shape[0]
        nset   = dms.shape[0]
        nao    = dms.shape[-1]
        v_grids_e = numpy.zeros([nset, ngrids])

        max_memory = self.max_memory - lib.current_memory()[0]
        blksize = int(max(max_memory * 0.9e6 / 8 / nao**2, 400))

        for p0, p1 in lib.prange(0, ngrids, blksize):
            # shape (p1-p0, nao, nao): exact <μ|1/|r−R_g||ν> at grid points g
            v_nj = mol.intor('int1e_grids', grids=grid_coords[p0:p1])
            for i in range(nset):
                v_grids_e[i, p0:p1] = numpy.einsum('gij,ij->g', v_nj, dms[i])

        return v_grids_e

    def _get_vmat(self, q):
        '''
        Fock matrix contribution from PCM surface charges.

        OpenMolcas adds the PCM operator as a sum of 1/r potentials from
        point charges located at tessera centres.  We use int1e_grids which
        computes <μ|1/|r−R_g||ν> exactly, matching the OpenMolcas integral.

        V_μν = −Σ_g q_g <μ|1/|r−R_g||ν>
        '''
        mol = self.mol
        nao = mol.nao
        grid_coords = self.surface['grid_coords']
        ngrids = grid_coords.shape[0]
        q = q.reshape([-1, ngrids])
        nset = q.shape[0]
        vmat = numpy.zeros([nset, nao, nao])

        max_memory = self.max_memory - lib.current_memory()[0]
        blksize = int(max(max_memory * 0.9e6 / 8 / nao**2, 400))

        for p0, p1 in lib.prange(0, ngrids, blksize):
            # shape (p1-p0, nao, nao)
            v_nj = mol.intor('int1e_grids', grids=grid_coords[p0:p1])
            for i in range(nset):
                vmat[i] -= numpy.einsum('gij,g->ij', v_nj, q[i, p0:p1])

        return vmat

    def ss_correction(self, dm, refdm=None):

        if self.equilibrium_solvation == False:
            '''_, _, _q_ref, _vgrid_ref = self._get_vind(refdm)    # reference state total charge
            _q_ref_dyn, _ = self._get_vind_dyn(refdm)           # reference state dynamic charge
            _q_ref_in = _q_ref - _q_ref_dyn                  # reference state inertial charge (fixed)

            epcms = []
            for i in range(len(dm)):
                _q_dyn, _vgrid = self._get_vind_dyn(dm[i])         # state-averaged dynamic charge (changing)

                # Calculate polarization free energy according to NEQ model
            
                epcm = 0.5 * numpy.dot(_q_dyn[0], _vgrid[0]) + numpy.dot(_q_ref_in[0], _vgrid[0])
                epcms.append(epcm)

            logger.info(self, 'state-specific corrected states E(pol) = %s', epcms)'''
            # GS reference charges (slow/inertial)
            _, _, _q_ref, _vgrid_ref = self._get_vind(refdm)
            _q_ref_dyn, _            = self._get_vind_pekar(refdm)
            _q_ref_in                = _q_ref - _q_ref_dyn    # inertial (fixed)
 
            # Fast (dynamic) charges from dm_fast
            #_q_dyn, _ = self._get_vind_pekar(dm_fast)
            #_q_neq    = _q_ref_in + _q_dyn
 
            # Fock matrix contribution (same for all states)
            #sol_pot   = self._get_vmat(_q_neq)[0]
 
            # State-independent term:  -½ q_in · V_GS
            const_gs_term = -0.5 * numpy.dot(_q_ref_in[0], _vgrid_ref[0])
 
            # Per-state energies:
            #   epcm_k = ½ q_dyn · V^k  +  q_in · V^k  +  drop_term
            epcms = []
            for i, d in enumerate(dm):
                '''if rf_root is not None:
                    _, _, _, _vgrid_k = self._get_vind(dm_fast)
                else:'''
                _q_dyn, _vgrid_k = self._get_vind_pekar(d)

                v_k    = _vgrid_k[0]
                epcm_k = (0.5 * numpy.dot(_q_dyn[0], v_k)
                              + numpy.dot(_q_ref_in[0], v_k)
                              + const_gs_term)
                epcms.append(epcm_k)
        
        if self.equilibrium_solvation:
            epcms = []
            for d in dm:
                _, _, _q, _vgrid = self._get_vind(d)
                #print("_vgrid", _vgrid) 
                epcm = 0.5 * numpy.dot(_q[0], _vgrid[0])
                #print("epcm", epcm)
                epcms.append(epcm)

            logger.info(self, 'state-specific corrected states E(pol) = %s', epcms)
        return epcms


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
            
            print("state id ", state_id)
            print("self.state_id", self.state_id)

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

                # --- Step 1: ground-state equilibrium total charges Q̄_total ---
                _, _, _q_ref_total, _vgrid_ref = self._get_vind(refdm)
                # _q_ref_total shape: (1, ngrids)

                # --- Step 2: orientational (slow) charges  Q̄^or (eq. 15) -----
                #   Q̄^or = (ε - ε_opt) / (ε - 1) · Q̄_total
                fact_or = (epsilon - epsilon_opt) / (epsilon - 1.0)
                q_or = fact_or * _q_ref_total[0]          # shape (ngrids,)

                # --- Step 3: GS fast charges Q̄^el  ----------------------------
                #   Q̄^el = Q̄_total - Q̄^or  (from eq. 13 for GS: Q̄ = Q̄^el + Q̄^or)
                q_el_gs = _q_ref_total[0] - q_or          # shape (ngrids,)

                # --- Step 4: ES fast charges Q̄̄^el (eq. 18) -------------------
                #   D_{ε_opt} Q̄̄^el = -b̄̄ - Ω Q̄^or   (Ω = S)
                q_el_es, v_grids_mol = self._get_vind_marcus(dm, q_or)
                # q_el_es shape: (1, ngrids);  v_grids_mol shape: (1, ngrids)

                # --- Step 5: total NEQ charges (eq. 13) and Fock contribution -
                #   Q̄̄^neq = Q̄̄^el + Q̄^or
                q_neq = q_el_es.copy()
                q_neq[0] += q_or
                sol_pot = self._get_vmat(q_neq)[0]

                # --- Step 6: polarization free energy (eqs 23-26) -------------
                S = self._intermediates['S']
                v_slow = S @ q_or                         # V_slow at tesserae

                # V̄̄_mol = molecular ESP of ES (from _get_vind_marcus)
                v_es = v_grids_mol[0]                     # shape (ngrids,)
                # V̄_mol = molecular ESP of GS (already computed)
                v_gs = _vgrid_ref[0]                      # shape (ngrids,)

                # G_P,el = ½ Σ_m V̄̄_m Q̄̄^el_m  (eq. 24)
                # Here V̄̄_m = V_nuc + V_el_ES = -v_grids_mol  wait—
                # note: v_grids = v_grids_n - v_grids_e,  i.e. V_mol = +v_grids
                G_el   = 0.5 * numpy.dot(q_el_es[0], v_es)

                # G_P,or = Σ_m (V̄̄_m - ½V̄_m) Q̄^or_m  (eq. 25)
                G_or   = numpy.dot(q_or, v_es - 0.5 * v_gs)

                # G_P,el-or = ½ Σ_{m,m'} (Q̄̄^el_{m'} - Q̄^el_{m'}) Q̄^or_m / r_{mm'}
                #           = ½ (Q̄̄^el - Q̄^el) · V_slow   (using Ω = S)
                delta_q_el = q_el_es[0] - q_el_gs        # shape (ngrids,)
                G_el_or = 0.5 * numpy.dot(delta_q_el, v_slow)

                epcm = G_el + G_or + G_el_or

                logger.info(self, 'Marcus partition:  ε = %.4f,  ε_opt = %.4f', epsilon, epsilon_opt)
                logger.info(self, '  fact_or = (ε-ε_opt)/(ε-1) = %.6f', fact_or)
                logger.info(self, '  G_P,el   = %.15g', G_el)
                logger.info(self, '  G_P,or   = %.15g', G_or)
                logger.info(self, '  G_P,el-or= %.15g', G_el_or)
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
        elif isinstance(state_id, str) and state_id.lower() == 'sa_casscf' and es_method in [
                'nonequilibrium', 'non-equilibrium', 'non_equilibrium',
                'non-eq', 'non_eq', 'NEQ', 'neq', 'NONEQ', 'VEM', 'vem']:
            # ==================================================================
            # SA-CASSCF nonequilibrium branch
            # ------------------------------------------------------------------
            # dm    : list of length N_states + 1
            #           dm[0..N_states-1]  — individual state density matrices
            #           dm[-1]             — SA-averaged density matrix
            #
            # refdm : GS equilibrium density for slow charges Q^or / Q^in.
            #
            # rf_root : None or int k (1-indexed, like OpenMolcas RFRoot)
            #   None  → fast charges computed from SA-averaged density dm[-1]
            #   k     → fast charges computed from state k's density dm[k-1]
            #
            # Returns
            # -------
            # epcms   : list of length N_states, one G_P per state
            # sol_pot : single Fock matrix contribution (same for all states),
            #           built from q_neq using dm_fast
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
 
            '''if partition == 'marcus':
                # ==============================================================
                # Marcus partition (Partition I / OpenMolcas) — eqs 13, 15, 18
                # ==============================================================
 
                # Step 1: GS equilibrium total charges Q̄_total
                _, _, _q_ref_total, _vgrid_ref = self._get_vind(refdm)
 
                # Step 2: orientational (slow) charges Q̄^or (eq. 15)
                #   Q̄^or = (ε - ε_opt)/(ε - 1) · Q̄_total
                fact_or = (epsilon - epsilon_opt) / (epsilon - 1.0)
                q_or    = fact_or * _q_ref_total[0]           # (ngrids,)
 
                # Step 3: GS fast charges Q̄^el = Q̄_total - Q̄^or
                q_el_gs = _q_ref_total[0] - q_or              # (ngrids,)
 
                # Step 4: ES fast charges Q̄̄^el from dm_fast (eq. 18)
                #   D_{ε_opt} Q̄̄^el = -b̄̄ - Ω Q̄^or   (Ω = S)
                q_el_es, _ = self._get_vind_marcus(dm_fast, q_or)
 
                # Step 5: total NEQ charges and Fock matrix (same for all states)
                q_neq      = q_el_es.copy()
                q_neq[0]  += q_or
                sol_pot    = self._get_vmat(q_neq)[0]
 
                # State-independent constants
                S       = self._intermediates['S']
                v_slow  = S @ q_or
                v_gs    = _vgrid_ref[0]
                G_el_or = 0.5 * numpy.dot(q_el_es[0] - q_el_gs, v_slow)  # eq. 26
                const   = -0.5 * numpy.dot(q_or, v_gs) + G_el_or
 
                # Per-state energies: epcm_k = (½ q_el_es + q_or)·V^k + const
                epcms = []
                for i, d in enumerate(state_dms):
                    _, _, _, _vgrid_k = self._get_vind(d)
                    v_k    = _vgrid_k[0]
                    G_el_k = 0.5 * numpy.dot(q_el_es[0], v_k)   # eq. 24 for state k
                    G_or_k = numpy.dot(q_or, v_k)                # eq. 25 (V^k part)
                    epcm_k = G_el_k + G_or_k + const
                    epcms.append(epcm_k)
                    logger.info(self, '  Marcus G_P state %d: el=%.10g  or=%.10g  '
                                'el-or=%.10g  total=%.10g',
                                i+1, G_el_k, G_or_k, G_el_or, epcm_k)
 
                logger.info(self, 'SA-CASSCF Marcus:  ε=%.4f  ε_opt=%.4f  '
                            'fact_or=%.6f  G_el_or=%.10g',
                            epsilon, epsilon_opt, fact_or, G_el_or)'''
            if partition == 'marcus':
                # ==============================================================
                # Marcus partition — exact OpenMolcas drvpcm.F90 energy assembly
                # ==============================================================
                # Mirrors the RepNuc line (255-256), h1 (278), TwoHam (299).
                #
                # Notation (all at tessera positions):
                #   V_n       = nuclear ESP  (v_grids_n)
                #   V_el^(k)  = electronic ESP from state k  (negative, = _vgrid_k - V_n)
                #   V_s       = S · Q^or  (V_slow)
                #   q_fn      = fast nuclear charges, driven by V_n + V_s  [q(1) in drvpcm]
                #   q_fe      = fast electronic charges, driven by V_el     [q(2) in drvpcm]
                #   q_or      = slow orientational charges  [Q_Slow]
                #   q_el_gs   = GS fast total charges  [QInf in equilibrium]
                #
                # RepNuc constants (line 255-256):
                #   ½ ENN          = ½ q_fn · V_n
                #   ½ W_or_nuc     = ½ q_or · V_n
                #   ½ W_or_InfNuc  = ½ q_fn · V_s
                #   −½ W_0_or_el   = −½ q_or · V_el_GS
                #   −½ W_0_or_Inf  = −½ q_el_gs · V_s
                #
                # h1 (per state k, line 278): (q_fn + q_or) · V_el^(k)
                # TwoHam (per state k, line 299): ½ q_fe · V_el^(k)
                # ==============================================================
 
                # --- GS quantities -------------------------------------------
                _, _, _q_ref_total, _vgrid_ref = self._get_vind(refdm)
                v_gs    = _vgrid_ref[0]              # V_n + V_el_GS  (PySCF total)
 
                # Q^or = (ε-ε_opt)/(ε-1) · Q_total_GS  (eq. 15)
                fact_or = (epsilon - epsilon_opt) / (epsilon - 1.0)
                q_or    = fact_or * _q_ref_total[0]  # (ngrids,)
 
                # Q^el_GS = Q_total_GS - Q^or  (GS fast total, = QInf in drvpcm)
                q_el_gs = _q_ref_total[0] - q_or     # (ngrids,)
 
                # V_slow = S · Q^or
                S, K_opt, R_opt = (self._intermediates['S'],
                                   *self._get_K_opt_R_opt())  # lazy — uses current refidx
                v_slow = S @ q_or                    # (ngrids,)
 
                # --- Fast charges from dm_fast --------------------------------
                # q_fn: driven by V_n + V_s  (OpenMolcas: PCM_Charge(1) after line 167)
                v_nuc_slow       = numpy.zeros((1, len(q_or)))
                v_nuc_slow[0]    = self.v_grids_n + v_slow
                b_fn             = numpy.dot(R_opt, v_nuc_slow.T)
                q_fn_raw         = numpy.linalg.solve(K_opt, b_fn).T
                vK_fn            = numpy.linalg.solve(K_opt.T, v_nuc_slow.T)
                q_fn_t           = numpy.dot(R_opt.T, vK_fn).T
                q_fast_nuc       = (q_fn_raw + q_fn_t) / 2.0   # (1, ngrids)
 
                # q_fast_total: driven by V_n + V_el + V_s  (eq. 18, _get_vind_marcus)
                q_fast_total, _ = self._get_vind_marcus(dm_fast, q_or)  # (1, ngrids)
 
                # q_fe: electronic only (linearity of K^{-1}R)
                q_fast_el = q_fast_total - q_fast_nuc               # (1, ngrids)
 
                # --- Fock matrix  (same for all states) ----------------------
                # q_neq = q_fn + q_fe + q_or  (OpenMolcas: h1←q_fn+q_or, TwoHam←q_fe)
                q_neq      = q_fast_total.copy()
                q_neq[0]  += q_or
                sol_pot    = self._get_vmat(q_neq)[0]
 
                # --- State-independent RepNuc constants (line 255-256) -------
                #   V_el_GS (OpenMolcas sign) = V_tot_GS - V_n = v_gs - v_grids_n
                half_ENN        = 0.5 * numpy.dot(q_fast_nuc[0], self.v_grids_n)
                half_W_or_nuc   = 0.5 * numpy.dot(q_or,          self.v_grids_n)
                half_W_InfNuc   = 0.5 * numpy.dot(q_fast_nuc[0], v_slow)
                W_0_or_el       = numpy.dot(q_or, v_gs - self.v_grids_n)  # q_or·V_el_GS
                W_0_or_Inf      = numpy.dot(q_el_gs, v_slow)
                const = (half_ENN + half_W_or_nuc + half_W_InfNuc
                         - 0.5 * W_0_or_el - 0.5 * W_0_or_Inf)
 
                logger.info(self, 'SA-CASSCF Marcus:  ε=%.4f  ε_opt=%.4f  fact_or=%.6f',
                            epsilon, epsilon_opt, fact_or)
                logger.info(self, '  ½ ENN        = %.10g', half_ENN)
                logger.info(self, '  ½ W_or_nuc   = %.10g', half_W_or_nuc)
                logger.info(self, '  ½ W_or_InfNuc= %.10g', half_W_InfNuc)
                logger.info(self, '  -½ W_0_or_el = %.10g', -0.5 * W_0_or_el)
                logger.info(self, '  -½ W_0_or_Inf= %.10g', -0.5 * W_0_or_Inf)
                logger.info(self, '  RepNuc const = %.10g', const)
 
                # --- Per-state energies ---------------------------------------
                # G_P^(k) = const
                #          + (q_fn + q_or) · V_el^(k)     [h1, line 278]
                #          + ½ q_fe · V_el^(k)             [TwoHam, line 299]
                # where V_el^(k) = _vgrid_k[0] - v_grids_n  (PySCF sign convention)
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
                # ==============================================================
                # Pekar partition — SA-CASSCF version
                # ==============================================================
 
                # GS reference charges (slow/inertial)
                _, _, _q_ref, _vgrid_ref = self._get_vind(refdm)
                _q_ref_dyn, _            = self._get_vind_pekar(refdm)
                _q_ref_in                = _q_ref - _q_ref_dyn    # inertial (fixed)
 
                # Fast (dynamic) charges from dm_fast
                _q_dyn, _ = self._get_vind_pekar(dm_fast)
                _q_neq    = _q_ref_in + _q_dyn
 
                # Fock matrix contribution (same for all states)
                sol_pot   = self._get_vmat(_q_neq)[0]
 
                # State-independent term:  -½ q_in · V_GS
                const_gs_term = -0.5 * numpy.dot(_q_ref_in[0], _vgrid_ref[0])
 
                # Per-state energies:
                #   epcm_k = ½ q_dyn · V^k  +  q_in · V^k  +  drop_term
                epcms = []
                for i, d in enumerate(state_dms):
                    if rf_root is not None:
                        _, _, _, _vgrid_k = self._get_vind(dm_fast)
                    else:
                        _, _, _, _vgrid_k = self._get_vind(d)

                    v_k    = _vgrid_k[0]
                    epcm_k = (0.5 * numpy.dot(_q_dyn[0], v_k)
                              + numpy.dot(_q_ref_in[0], v_k)
                              + const_gs_term)
                    epcms.append(epcm_k)
                    logger.info(self, '  Pekar G_P state %d = %.15g', i, epcm_k)
 
                logger.info(self, 'SA-CASSCF Pekar:  eps_opt=%.6f',
                            epsilon_opt)
 
            logger.info(self, 'SA-CASSCF NEQ G_P per state = %s', epcms)
            return epcms, sol_pot
        
        elif isinstance(state_id, str) and state_id.lower() == 'sa_casscf' and es_method is None and self.equilibrium_solvation:
            # ==================================================================
            # SA-CASSCF equilibrium branch
            # ------------------------------------------------------------------
            # dm    : list of length N_states + 1
            #           dm[0..N_states-1]  — individual state density matrices
            #           dm[-1]             — SA-averaged density matrix
            #
            # rf_root : None or int k (1-indexed)
            #   None  → charges from SA-averaged density dm[-1]  (default)
            #   k     → charges from state k's density dm[k-1]
            #           (analogous to OpenMolcas RFRoot=k for equilibrium)
            #
            # Returns
            # -------
            # epcms   : list of length N_states, one G_P per state
            # sol_pot : Fock matrix built from the equilibrium charges
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
 
            # Fock matrix is the same for all states
            sol_pot = self._get_vmat(_q)[0]
 
            # Per-state energies: G_P^(k) = ½ q · V^(k)
            # where V^(k) is the molecular ESP of state k.
            # If rf_root is set, V^(k) is still state k's potential —
            # only the charges come from the rf_root density.
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
