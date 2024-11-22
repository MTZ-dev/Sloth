if __name__ == "__main__": #tylko proces gwny
    import slothpy as slt
    from slothpy._general_utilities._constants import B_AU_T
    import numpy as np
    from threadpoolctl import threadpool_limits
    import os
    os.remove('Nd_lumi.slt')
    Yb=slt.hamiltonian_from_orca('Yb_bpdo_casscf_nevpt2_dipmom_opt.out','Nd_lumi','test',True, True, False)
    # Yb = slt.slt_file("Nd_lumi.slt")
    # Yb["test"].s[:]
    #print(Yb)
    with threadpool_limits(4):
        from scipy.spatial.transform import Rotation
        magnetic_field = - np.asarray([0.01,0.01,0.01]) / B_AU_T
        M = Yb["test"].magnetic_dipole_momentum_matrices("xyz").eval()
        ham_zeeman = M[0] * magnetic_field[0] + M[1] * magnetic_field[1] + M[2] * magnetic_field[2]
        # magnetic_field[:,np.newaxis, np.newaxis] * M
        energies = Yb["test"].states_energies_au().eval()
        for i in range(ham_zeeman.shape[0]):
            ham_zeeman[i,i] += energies[i] #tu musi byc rzeczywiste, poprawic mnozenie l i s, -07=0
        energies2, eigenvectors = np.linalg.eigh(ham_zeeman)
        dipole_momenta = Yb["test"].electric_dipole_momentum_matrices("xyz").eval()
        # print(np.diagonal(dipole_momenta[0]))
        transformed_dip_mom = eigenvectors.conj().T[np.newaxis, :, :] @ dipole_momenta @ eigenvectors #rzeczywiste, czemu taka duza energia
        #To chyba działało tez bez tego newaxis ale trzeba sprawdzic

        #print(np.diag(transformed_dip_mom[0]))
        print(1e7/(energies*219474.6)) 
        print(1e7/(energies2*219474.6))
