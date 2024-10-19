from __future__ import annotations
import numpy as np
import numpy.typing as npt
from scipy.optimize import direct
from .DRT import DRT

class ColeColeDRT(DRT):
    """
    DRT with Cole-Cole (radial) basis functions
    
    @version:   AA-20240712(20240712)
    @code:      Robert Leonhardt - mail@robertleonhardt.de
    """
    
    # Default shape parameter; fractal exponent
    alpha: float = 0.95
    
    def _get_basis_matrix(self, alpha: float = None) -> npt.ArrayLike:
        """
        Cole-Cole distribution function as basis function
        
        Args:
            alpha (float): Shape parameter

        Returns:
            npt.ArrayLike: Basis matrix
        """
        
        if alpha is None:
            alpha = self.alpha
        
        # Populate distribution matrix and return
        # NOTE: https://numpy.org/doc/stable/user/basics.broadcasting.html
        # This equation was taken from Boukamp, B. A. and A. Rolle (2018). "Use of a distribution function of relaxation times (DFRT) in impedance analysis of SOFC electrodes." 
        # Solid State Ionics 314: 103-111. (https://doi.org/10.1016/j.ssi.2017.11.021)
        basis_matrix: npt.ArrayLike = (1 / (2 * np.pi)) * np.sin(np.pi * self.alpha) / (np.cosh(self.alpha * np.log(np.outer(self.tau_s, 1 / self.tau_s))) + np.cos(np.pi * self.alpha))
        
        # NOTE: Due to the broadcasting, the (quadratic) matrix was computed with one basis function per row,
        # so we need to transpose the matrix, so that we have - as required - one basis function per column.
        # 
        # Since this is a radial basis function, which leads to symmetrical basis matrices, ignoring this
        # step would not cause any harm, but we're doing this anyway. :)
        return basis_matrix.T
    
    @staticmethod
    def get_analytical_solution(tau_s: npt.ArrayLike, tau_c: float, alpha: float) -> npt.ArrayLike:
        # Return calculated vector
        return (1 / (2 * np.pi)) * np.sin(np.pi * alpha) / (np.cosh(alpha * np.log(np.array(tau_s) / tau_c)) + np.cos(np.pi * alpha))
    
    @staticmethod
    def __parameter_optimization_cost(params: list[float], drt: ColeColeDRT) -> float:
        # Apply parameters and solve DRT
        drt.alpha = params
        drt.solve()
        
        # Get goodness of fit measure 
        residual_cost = drt.relative_resnorm
        
        # Get number of "non-zero" elements in relation to the worst-case maximum (everything is nonzero)
        element_number_cost = drt.number_of_nonzero_w_hat_values / len(drt.weight_vector)
        
        return np.log10(residual_cost + element_number_cost)
    
    @staticmethod 
    def optimize_shape_parameters(drt: ColeColeDRT, alpha_min: float = 1e-3, alpha_max: float = 0.9999) -> ColeColeDRT:
        # Use the rather slow global optimization
        optim = direct(ColeColeDRT.__parameter_optimization_cost, args = (drt,), bounds = [(alpha_min, alpha_max)], locally_biased = False)
        
        # Solve DRT once again to print some evaluation and store the results
        drt.alpha, = optim.x
        drt.solve()
        
        print(f'Optimal DRT parameters: \t alpha = {drt.alpha:.4f}, \t res = {drt.relative_resnorm:.6f}, ng = {drt.get_number_of_w_hat_values(threshold_factor = 0.02)} ({drt.number_of_nonzero_w_hat_values})')
        
        return drt