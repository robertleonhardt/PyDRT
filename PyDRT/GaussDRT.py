from __future__ import annotations
import numpy as np
import numpy.typing as npt
from scipy.optimize import direct
from .DRT import DRT

class GaussDRT(DRT):
    """
    DRT with gauss (radial) basis functions
    
    @version:   AA-20240712(20240712)
    @code:      Robert Leonhardt - mail@robertleonhardt.de
    """
    
    # Default shape parameter; full with at half maximum, fwhm
    fwhm: float = 0.25
    
    def _get_basis_matrix(self, fwhm: float = None) -> npt.ArrayLike:
        """
        Gaussian distribution function as basis function
        
        Args:
            fwhm (float): FWHM shape parameter

        Returns:
            npt.ArrayLike: Basis matrix
        """
        
        if fwhm is None:
            fwhm = self.fwhm
        
        # Full with at half maximum can be expressed as FWHM = 2*sqrt(2 ln(2))*c
        # NOTE: ln(10) * fwhm(log)
        c = (np.log(10) * fwhm) / (2 * np.sqrt(2 * np.log(2)))
                
        # Populate distribution matrix
        # NOTE: https://numpy.org/doc/stable/user/basics.broadcasting.html
        # The Gauss function was taken from https://en.wikipedia.org/wiki/Gaussian_function 
        basis_matrix: npt.ArrayLike = np.exp(-(self.ln_tau_tau0 - self.ln_tau_tau0[:, np.newaxis])**2 / (2 * c**2)) / np.sqrt(2 * np.pi * c**2)
        
        # NOTE: Due to the broadcasting, the (quadratic) matrix was computed with one basis function per row,
        # so we need to transpose the matrix, so that we have - as required - one basis function per column.
        # 
        # Since this is a radial basis function, which leads to symmetrical basis matrices, ignoring this
        # step would not cause any harm, but we're doing this anyways.
        return basis_matrix.T
    
    @staticmethod
    def __parameter_optimization_cost(params: list[float], drt: GaussDRT) -> float:
        # Apply parameters and solve DRT
        drt.fwhm = params
        drt.solve()
        
        # Get goodness of fit measure 
        residual_cost = drt.relative_resnorm
        
        # Get number of "non-zero" elements in relation to the worst-case maximum (everything is nonzero)
        element_number_cost = drt.number_of_nonzero_w_hat_values / len(drt.weight_vector)
        
        return np.log10(residual_cost + element_number_cost)
    
    @staticmethod 
    def optimize_shape_parameters(drt: GaussDRT, fwhm_min: float = 1e-6, fwhm_max: float = 1e6) -> GaussDRT:
        # Use the rather slow global optimization
        optim = direct(GaussDRT.__parameter_optimization_cost, args = (drt,), bounds = [(fwhm_min, fwhm_max)])
        
        # Solve DRT once again to print some evaluation and store the results
        drt.fwhm, = optim.x
        drt.solve()
        
        print(f'Optimal DRT parameters: \t fwhm = {drt.fwhm:.4f}, \t res = {drt.relative_resnorm:.6f}, ng = {drt.get_number_of_w_hat_values(threshold_factor = 0.02)} ({drt.number_of_nonzero_w_hat_values})')
        
        return drt