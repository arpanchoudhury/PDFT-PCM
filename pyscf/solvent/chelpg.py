# CHELPG charge calculation taken from gpu4pyscf, 
# with modifications to work on CPU. 


import numpy as np
import scipy
from pyscf.data import radii


modified_Bondi = radii.VDW.copy()
modified_Bondi[1] = 1.1 / radii.BOHR   # modified Bondi radius



def int1e_grids_cpu(mol, gridcoords, dm):
    """
    Compute electronic electrostatic potential on grid points
    using standard PySCF integrals.
    """

    # int1e_rinv evaluates <mu|1/r|nu>
    # with origin shifted to grid point
    nao = mol.nao_nr()
    v = np.zeros(len(gridcoords))

    for i, r0 in enumerate(gridcoords):
        with mol.with_rinv_origin(r0):
            rinv = mol.intor('int1e_rinv')
        v[i] = np.einsum("ij,ji->", rinv, dm)

    return v


def eval_chelpg_layer(
    mf,
    dm=None,
    deltaR=0.3,
    Rhead=2.8,
    ifqchem=True,
    Rvdw=modified_Bondi,
    verbose=None,
):
    """
    CPU version of CHELPG charge calculation
    """



    atomcoords = mf.mol.atom_coords(unit="B")
    if dm is None:
        dm = mf.make_rdm1()

    Roff = Rhead / radii.BOHR
    Deltar = 0.1


    def tau_f(R, Rcut, Roff):
        return (R - Rcut) ** 2 * (3 * Roff - Rcut - 2 * R) / (
            Roff - Rcut
        ) ** 3

    Rshort = np.array([Rvdw[i] for i in mf.mol._atm[:, 0]])

    atomtypes = np.array(mf.mol._atm[:, 0])

    xmin = atomcoords[:, 0].min() - Rhead / radii.BOHR - Rvdw[
        atomtypes[np.argmin(atomcoords[:, 0] - Rshort)]
    ]
    xmax = atomcoords[:, 0].max() + Rhead / radii.BOHR + Rvdw[
        atomtypes[np.argmax(atomcoords[:, 0] + Rshort)]
    ]

    ymin = atomcoords[:, 1].min() - Rhead / radii.BOHR - Rvdw[
        atomtypes[np.argmin(atomcoords[:, 1] - Rshort)]
    ]
    ymax = atomcoords[:, 1].max() + Rhead / radii.BOHR + Rvdw[
        atomtypes[np.argmax(atomcoords[:, 1] + Rshort)]
    ]

    zmin = atomcoords[:, 2].min() - Rhead / radii.BOHR - Rvdw[
        atomtypes[np.argmin(atomcoords[:, 2] - Rshort)]
    ]
    zmax = atomcoords[:, 2].max() + Rhead / radii.BOHR + Rvdw[
        atomtypes[np.argmax(atomcoords[:, 2] + Rshort)]
    ]

    x = np.arange(xmin, xmax, deltaR / radii.BOHR)
    y = np.arange(ymin, ymax, deltaR / radii.BOHR)
    z = np.arange(zmin, zmax, deltaR / radii.BOHR)

    gridcoords = np.meshgrid(x, y, z)
    gridcoords = np.vstack(list(map(np.ravel, gridcoords))).T


    r_pX = scipy.spatial.distance.cdist(atomcoords, gridcoords)

    Rkmin = (r_pX - np.expand_dims(Rshort, axis=1)).min(axis=0)

    Ron = Rshort + Deltar
    Rlong = Roff - Deltar

    AJk = np.ones(r_pX.shape)
    AJk[r_pX < np.expand_dims(Rshort, axis=1)] = 0

    if ifqchem:
        idx2 = (r_pX < np.expand_dims(Ron, 1)) & (
            r_pX >= np.expand_dims(Rshort, 1)
        )
        AJk[idx2] = tau_f(
            r_pX,
            np.expand_dims(Rshort, 1),
            np.expand_dims(Ron, 1),
        )[idx2]

        wLR = 1 - tau_f(Rkmin, Rlong, Roff)
        wLR[Rkmin < Rlong] = 1
        wLR[Rkmin > Roff] = 0
    else:
        wLR = np.ones(r_pX.shape[-1])
        wLR[Rkmin > Roff] = 0

    w = wLR * np.prod(AJk, axis=0)

    mask = w > 1e-14
    w = w[mask]
    r_pX = r_pX[:, mask]
    gridcoords = gridcoords[mask]


    r_pX_potential = 1.0 / r_pX

    potential_real = np.dot(
        mf.mol.atom_charges(), r_pX_potential
    )

    if dm.ndim == 3:  # UHF
        dm = dm[0] + dm[1]

    potential_real -= int1e_grids_cpu(
        mf.mol, gridcoords, dm
    )


    r_pX_potential_omega = r_pX_potential * w

    GXA = r_pX_potential_omega @ r_pX_potential.T
    eX = r_pX_potential_omega @ potential_real

    GXA_inv = np.linalg.inv(GXA)

    g = GXA_inv @ eX

    alpha = (g.sum() - mf.mol.charge) / GXA_inv.sum()

    q = g - alpha * GXA_inv @ np.ones(mf.mol.natm)


    return q
