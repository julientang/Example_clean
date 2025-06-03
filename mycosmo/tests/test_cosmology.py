import numpy as np
import numpy.testing as npt

from mycosmo.cosmology import critical_density, hubble


class TestCosmology:
    fid_cosmo = {
        "H0": 70,
        "omega_m_0": 0.3,
        "omega_k_0": 0.0,
        "omega_lambda_0": 0.7,
    }
    H_tolerance = 0.01
    z_range = np.array([0.0, 0.5, 1.0])
    H_expect = np.array([70, 91.60, 123.24])
    critical_density_tolerance = 0.01
    critical_density_expect = np.array([1.88e-26, 2.40e-26, 3.20e-26])  # in kg/m^3

    def test_hubble(self):
        H_vals = hubble(self.z_range, self.fid_cosmo)

        npt.assert_allclose(
            H_vals,
            self.H_expect,
            atol=self.H_tolerance,
            err_msg=(
                "The H(z) differs from expected values by more than "
                f"{self.H_tolerance} decimal places."
            ),
        )

    def test_critical_density(self):
        crit_dens_vals = critical_density(self.z_range, self.fid_cosmo)

        npt.assert_allclose(
            crit_dens_vals,
            self.critical_density_expect,
            atol=self.critical_density_tolerance,
            err_msg=(
                "The critical desity differs from expected values by more than "
                f"{self.critical_density_tolerance} decimal places."
            ),
        )
