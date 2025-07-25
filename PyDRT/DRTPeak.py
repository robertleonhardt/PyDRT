from __future__ import annotations
import numpy as np
import numpy.typing as npt

class DRTPeak:
    """
    Simple data class for easier handling of separated DRT peaks

    @version:   AA-20240428(20240428)
    @author:    Robert Leonhardt <r.leonhardt@campus.tu-berlin.de>
    """
    
    def __init__(self, tau_s: float, R_Ohm: float, C_F: float, weight: float, gamma_hat_Ohm: npt.ArrayLike, R_offset_Ohm: float):
        # Store peak
        self._tau_s         = tau_s
        self._R_Ohm         = R_Ohm
        self._C_F           = C_F 
        self._weight        = weight
        self._gamma_hat_Ohm = gamma_hat_Ohm
        self._R_offset_Ohm  = R_offset_Ohm
    
    @property 
    def tau_s(self) -> float:
        return self._tau_s 
    
    @property
    def R_Ohm(self) -> float:
        return self._R_Ohm 
    
    @property
    def C_F(self) -> float:
        return self._C_F 
    
    @property
    def weight(self) -> float:
        return self._weight
    
    @property 
    def gamma_hat_Ohm(self) -> npt.ArrayLike:
        return np.array(self._gamma_hat_Ohm)
    
    @property
    def gamma(self) -> npt.ArrayLike:
        return self.gamma_hat_Ohm / self.R_Ohm
    
    @property 
    def R_offset_Ohm(self) -> float:
        return self._R_offset_Ohm