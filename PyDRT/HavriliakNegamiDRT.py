from __future__ import annotations
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.optimize import minimize, direct, least_squares
from .DRT import DRT

class HavriliakNegamiDRT(DRT):
    """
    DRT with Havriliak-Negami basis functions
    
    @version:   AA-20240713(20240713)
    @code:      Robert Leonhardt - mail@robertleonhardt.de
    """
    
    # Default shape parameter; fractal exponents
    alpha:  float = 0.95
    beta:   float = 0.5
    
    def _get_basis_matrix(self, alpha: float = None, beta: float = None) -> npt.ArrayLike:
        """
        Cole-Cole distribution function as basis function
        
        The basic form of the corresponding impedance element is given by
        Z = R0 / (1 + (j * omega * tau)^(alpha))^(beta)
        
        Args:
            alpha (float): Shape parameter
            beta (float): Shape parameter

        Returns:
            npt.ArrayLike: Basis matrix
        """
        
        if alpha is None:
            alpha = self.alpha
            
        if beta is None:
            beta = self.beta
            
        # Setup basis matrix by broadcasting the tau vectors
        tau_c:          npt.ArrayLike = self.tau_s / self.tau_s[:, np.newaxis]
        
        # This equation was taken from Boukamp, B. A. and A. Rolle (2018). "Use of a distribution function of relaxation times (DFRT) in impedance analysis of SOFC electrodes." 
        # Solid State Ionics 314: 103-111. (https://doi.org/10.1016/j.ssi.2017.11.021)
        theta:          npt.ArrayLike = np.pi / 2 - np.arctan((tau_c**alpha + np.cos(np.pi * alpha)) / np.sin(np.pi * alpha))
        basis_matrix:   npt.ArrayLike = 1 / np.pi * tau_c**(alpha * beta) * np.sin(beta * theta) / (1 + 2 * np.cos(np.pi*alpha) * tau_c**alpha + tau_c**(2*alpha))**(beta / 2)
        
        # NOTE: Due to the broadcasting, the (quadratic) matrix was computed with one basis function per row,
        # so we need to transpose the matrix, so that we have - as required - one basis function per column.
        return basis_matrix.T
    
    @staticmethod
    def get_analytical_solution(tau_s: npt.ArrayLike, tau_c: float, alpha: float, beta: float) -> npt.ArrayLike:
        # Calculate theta
        theta:      npt.ArrayLike = np.pi / 2 - np.arctan(((tau_s / tau_c)**alpha + np.cos(np.pi * alpha)) / np.sin(np.pi * alpha))
        solution:   npt.ArrayLike = 1 / np.pi * (tau_s / tau_c)**(alpha * beta) * np.sin(beta * theta) / (1 + 2 * np.cos(np.pi*alpha) * (tau_s / tau_c)**alpha + (tau_s / tau_c)**(2*alpha))**(beta / 2)
        
        # Return calculated vector
        return solution
    
    @staticmethod
    def __parameter_optimization_cost(params: list[float], drt: HavriliakNegamiDRT) -> float:
        # Apply parameters and solve DRT
        drt.alpha, drt.beta = params
        drt.solve()
        
        # Get goodness of fit measure 
        residual_cost      = drt.relative_resnorm
        
        # Get number of "non-zero" elements in relation to the worst-case maximum (everything is nonzero)
        element_number_cost = drt.number_of_nonzero_w_hat_values / len(drt.weight_vector)
        
        return np.log10(residual_cost + element_number_cost)
    
    @staticmethod 
    def optimize_shape_parameters(drt: HavriliakNegamiDRT, alpha_min: float = 0.5, alpha_max: float = 0.999, beta_min: float = 1e-3, beta_max: float = 1) -> HavriliakNegamiDRT:
        # Use the rather slow global optimization
        optim = direct(HavriliakNegamiDRT.__parameter_optimization_cost, args = (drt,), bounds = [(alpha_min, alpha_max), (beta_min, beta_max)], locally_biased = False)
        
        # Solve DRT once again to print some evaluation and store the results
        drt.alpha, drt.beta = optim.x
        drt.solve()
        
        print(f'Optimal DRT parameters: \t alpha = {drt.alpha:.4f}, beta = {drt.beta:.4f}, \t res = {drt.relative_resnorm:.6f}, ng = {drt.get_number_of_w_hat_values(threshold_factor = 0.02)} ({drt.number_of_nonzero_w_hat_values})')
        
        return drt
