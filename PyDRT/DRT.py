from __future__ import annotations
import scipy
import numpy as np
import numpy.typing as npt
import warnings
from scipy.optimize import direct
from scipy.signal import find_peaks
from .DRTPeak import DRTPeak

class DRT:
    """
    Main DRT class for determining DRT functions from measured impedance spectra by discretization and approximation using basis functions.
    The present code is published as part of the following work:
        Leonhardt, et al. (2024). "Reconstructing the distribution of relaxation times with analytical basis functions" Journal of Y, Submitted, DOI: Y
    
    If this code helps you with your research, please consider citing the reference above - this would be very helpful. :)
    
    In case you want a more convenient DRT experience with a more user-friencly GUI, check out Polarographica: 
    https://github.com/Polarographica/Polarographica_program
    
    Notational remarks:
    - The regularization parameter is typically called "lambda". However, Python used the same name as its keyword for anonymous functions.
      Therefore, epsilon is used instead. So every epsilon here is a Tikhonov-lambda in its heart.
    - The code at hand uses "tau_s" to indicate that tau has the unit of seconds. Since the logarithm formally does not work with quantities
      with units, the time constants are typically centered (tau/tau_0). Since it is handy to use tau in seconds when it comes to plotting DRTs
      or reporting relevant time constants, however, tau_s is used throughout the script when it really should say "tau_tau0".
    
    @version:   BB-20241010(20230303)
    @code:      Robert Leonhardt - mail@robertleonhardt.de
    """
    
    ## Class constants (should not be Diddy'd with)
    # Which data shall be used for the DRT computation
    COMBINED, REAL, IMAG    = 0, 1, 2
    
    ## Default values (can be configured here)
    # Time constant array
    tau_range_min_s:        float = 1e-6
    tau_range_max_s:        float = 1e6
    tau_points_per_decade:  int = 30
    
    # Even though we computated the tau range as specified above, it can make sense to ignore some parts (diffusion, etc.)
    # This can be done by adapting the following values (should be done during initilization of the object)
    tau_min_s:              float = 1e-6
    tau_max_s:              float = 1e6
    
    # Regularization parameter (something between 0.01 - 0.001 is fine for most applications)
    # NOTE: If set to False, no regularization will take place
    epsilon:                float = False
    
    # Solving parameters (see constants above)
    solving_data:           int = COMBINED
    
    # Offset from rightmost (in the Nyquist plane) local minimum (where the diffusion usually starts) where the data is trimmed
    # NOTE: Negative values move this point to the left (towards the capacitive processes)
    diffusion_offset:       int = False
    
    
    def __init__(self, frequency_data_Hz: npt.ArrayLike, impedance_data_Ohm: npt.ArrayLike, solve: bool = True, **kwargs: Any) -> None:
        """
        Main initializer; this method prepares the input data (frequency and impedance) for the subsequent computation
        
        Args:
            frequency_data_Hz (npt.ArrayLike): Frequency array of the measured impedance
            impedance_data_Ohm (npt.ArrayLike): Impedance array of the measured impedance
            solve (bool): If true (default) DRT get calculated right now, otherwise, "drt.solve()" has to be called separately
        """
        
        # Store additionally parsed parameters
        # Useful for adding SOCs, etc.
        # NOTE: This can also used to redefine deault values on the fly
        self.__dict__.update(kwargs)
        
        # First of all, ensure that we have happy little numpy arrays
        frequency_Hz, impedance_Ohm = np.array(frequency_data_Hz), np.array(impedance_data_Ohm)
        
        # Sort data in ascending order
        if frequency_Hz[0] < frequency_Hz[1]:
            frequency_Hz    = frequency_Hz[::-1]
            impedance_Ohm   = impedance_Ohm[::-1]
        
        # Store original data as it can be handy to have this for later comparisons
        self.__frequency_data_orig_Hz   = frequency_Hz
        self.__impedance_data_orig_Ohm  = impedance_Ohm
        
        # Get Ohmic resistance (which will be excluded from the subsequent DRT computation)
        # NOTE: This step is performed using the impedance data sorted in ascending order 
        # (the author was unable to do this with a descending impedance vector reliably)
        ohmic_resistance_Ohm: float = np.interp(0, impedance_Ohm[::-1].imag, impedance_Ohm[::-1].real)
        impedance_Ohm -= ohmic_resistance_Ohm
        
        # Remove inductive parts
        frequency_Hz, impedance_Ohm = frequency_Hz[impedance_Ohm.imag <= 0], impedance_Ohm[impedance_Ohm.imag <= 0]
        
        # If configured: remove impedance right of the specified local minima
        # Behind that point, the impedance should be diffusive only, which is not helpful
        if self.diffusion_offset:
            # Find index of right-most local minimum
            # NOTE: That the author was once again incapable of doing this without flipping the impedance data
            impedance_imag_increase_indices_list = scipy.signal.argrelmin(-impedance_Ohm.imag)[0]
            
            # Shorten impedance data of local minimum is found
            if len(impedance_imag_increase_indices_list) > 0:
                impedance_imag_increase_index = impedance_imag_increase_indices_list[-1]
            
                # Set the index to the found index + an offset of some points to that we mitigate the diffusion-bases "pull-up" at the end of an arc
                diffusion_start_index = len(impedance_Ohm) - impedance_imag_increase_index + 1 + self.diffusion_offset
                
                # Trim diffusive parts
                frequency_Hz    = frequency_Hz[-diffusion_start_index:]
                impedance_Ohm   = impedance_Ohm[-diffusion_start_index:]
        
        # Store the data in the object for later use
        self.__frequency_data_Hz:       npt.arraylike = frequency_Hz
        self.__impedance_data_Ohm:      npt.arraylike = impedance_Ohm
        self.__ohmic_resistance_Ohm:    float = ohmic_resistance_Ohm
        
        # Setup time constants (with default values)
        self.setup_time_constants()
        
        # Directly solve the DRT
        if solve:
            self.solve()

    
    def solve(self, solving_data: int = None, epsilon: float = None, tau_lower_boundary_s: float = None, tau_upper_boundary_s: float = None, **kwargs: Any) -> DRT:
        """
        Main solving method that computes everything

        Args:
            solving_data (int, optional): Can be set to DRT.COMBINED, DRT.REAL, DRT.IMAG. Defaults to None (DRT:COMBINED).
            epsilon (float, optional): Tikhonov regularization parameter. Defaults to None (default is specific above).
            tau_lower_boundary_s (float, optional): Time constant below which everything will be set to zero. Defaults to None.
            tau_upper_boundary_s (float, optional): Time constant above which everything will be set to zero. Defaults to None.

        Returns:
            DRT: Chaining ...
        """
        
        # Store additionally parsed parameters
        # Is mainly used here to adapt shape parameters on the fly like "drt.solve(alpha = 0.92)"
        self.__dict__.update(kwargs)
        
        # Get info for which data the DRT shall be determined (i.e., real ony, imag only, or both)
        if solving_data is None:
            solving_data = self.solving_data
        
        # Get regularization information
        # NOTE: If nothing is passed, the last used value will be used
        if epsilon is None:
            epsilon = self.epsilon
            
        # Get default kernel matrix and apply the basis matrix
        rc_kernel_matrix:   npt.ArrayLike = self._get_rc_kernel_matrix()
        basis_matrix:       npt.ArrayLike = self._get_basis_matrix()
        kernel_matrix:      npt.ArrayLike = np.dot(rc_kernel_matrix, basis_matrix)
            
        # Init kernel and impedance data
        match solving_data:
            case DRT.REAL:
                stacked_kernel_matrix:  npt.ArrayLike = kernel_matrix.real
                impedance_vector:       npt.ArrayLike = self.__impedance_data_Ohm.real
                
            case DRT.IMAG:
                stacked_kernel_matrix:  npt.ArrayLike = kernel_matrix.imag
                impedance_vector:       npt.ArrayLike = self.__impedance_data_Ohm.imag
                
            case _:
                stacked_kernel_matrix:  npt.ArrayLike = np.vstack((kernel_matrix.real, kernel_matrix.imag))
                impedance_vector:       npt.ArrayLike = [*self.__impedance_data_Ohm.real, *self.__impedance_data_Ohm.imag]
        
        # Apply regularization, if necessary
        if epsilon:
            stacked_kernel_matrix   = np.vstack((stacked_kernel_matrix, epsilon * self._get_regularization_matrix()))
            impedance_vector        = [*impedance_vector, *np.zeros(self.tau_s.shape)]
            
            # # Report that regularization was done
            # print(f'DRT is regularized with lambda/epsilon = {epsilon:.4f}')

        # Compute DRT using nnls solver
        # NOTE: We determine w_hat = w * R_pol (the polarization resistance is fitted alongside the weight since determining it separately is more complicated). 
        # w_hat is, therefore, still a weight vector, even though its scaled. 
        # To get the DRT, linear combination with the basis is necessary.
        w_hat_vector, norm  = scipy.optimize.nnls(stacked_kernel_matrix, impedance_vector)
            
        # Refine weight vector (merge smeared weight-peaks, remove "noise", and boundary artifacts)
        w_hat_vector        = self.__refine_weight_vector(w_hat_vector, tau_lower_boundary_s, tau_upper_boundary_s)
        
        # Calculate gamma, R_pol, etc.
        gamma_hat_Ohm:  npt.ArrayLike = np.dot(self._get_basis_matrix(), w_hat_vector)
        R_pol_Ohm:      float = np.trapz(gamma_hat_Ohm, self.ln_tau_tau0)
        gamma_array:    npt.ArrayLike = gamma_hat_Ohm / R_pol_Ohm
        
        # Store data and back calculations etc.
        self.__weight_vector    = w_hat_vector / np.sum(w_hat_vector)
        self.__gamma_hat_Ohm    = gamma_hat_Ohm
        self.__gamma_array      = gamma_array
        self.__R_pol_Ohm        = R_pol_Ohm
        
        # Back-calc of impedance out of DRT data
        self.__impedance_back_Ohm: npt.ArrayLike = self.R_offset_Ohm + np.dot(kernel_matrix.real, w_hat_vector) + 1j * np.dot(kernel_matrix.imag, w_hat_vector)
        
        # Store additional information (e.g., g_hat, which contains the psotions of the peaks)
        self.__w_hat_vector      = w_hat_vector
        self.__norm             = norm
        
        return self
    
    def __refine_weight_vector(self, w_hat_vector: npt.ArrayLike, tau_lower_boundary_s: float = None, tau_upper_boundary_s: float = None) -> npt.ArrayLike:
        """
        Method to refine the weight vector by
        a) Trimming the weights to a specific time constant range to remove artifacts at the boundaries
        b) Remove very small, non-zero-elements which are probably artifacts and don't help with interpretation at all
        c) Merge neighboring peaks. It is probable that a found peak lies right between two discrete time constants (in the tau vector)
           This results in a single peak being distributed over two neighboring collocation points -> these peaks are merged here.

        Args:
            w_hat_vector (npt.ArrayLike): Weight vector
            tau_lower_boundary_s (float, optional): Time constant below which everything will be set to zero. Defaults to None.
            tau_upper_boundary_s (float, optional): Time constant above which everything will be set to zero. Defaults to None.

        Returns:
            npt.ArrayLike: Refined weight vector
        """
        
        # We can implement thresholds for tau here (given as function arguments), so we can ignore artifacts at the global tau boundaries
        # This basically will ignore all weight vector entries that are < tau_min_s and > tau_max_s.
        if tau_lower_boundary_s is None:
            tau_lower_boundary_s = self.tau_min_s
        if tau_upper_boundary_s is None:
            tau_upper_boundary_s = self.tau_max_s
            
        # Set gamma outside of the boundaries to zero
        # Also, set exceptionally small gammas to zero
        w_hat_vector[(self.tau_s < tau_lower_boundary_s) | (self.tau_s > tau_upper_boundary_s)] = 0
        w_hat_vector[w_hat_vector < 0.001 * np.max(w_hat_vector)] = 0
        
        # Merge neighboring peaks into the largest one
        # Step 1: We look though the weight vector for the maximum
        # Step 2: We look left and right from the maximum and add up all weight until they decay to zero
        #         The summed up value is the set to the position of the maximum value and the other (merged) weights will be set to zero
        #         E.g., [0, 0.1, 0, 0.1, 0.2, 0.3, 0.2, 0.1, 0, 0.1] -> [0, 0.1, 0, 0, 0, 0.9, 0, 0, 0, 0.1]
        # Step 3: Remove peak from consideration (so we don't merge peaks multiple times)
        # Step 3: Repeat step 1-3 until there are no peaks left.
        
        # Start by making copies of the weight vector
        w_hat_vector_stripped    = w_hat_vector.copy()
        w_hat_vector_merged      = np.zeros_like(w_hat_vector)
        
        while True:
            # Get maximum 
            index_max = np.argmax(w_hat_vector_stripped)
            
            # Get window of non-zero elements around maximum index
            index_lower, index_upper = index_max, index_max
            while (index_lower > 0) and (w_hat_vector_stripped[index_lower - 1] > 0):
                index_lower -= 1
            while (index_upper < len(w_hat_vector_stripped) - 1) and (w_hat_vector_stripped[index_upper + 1] > 0):
                index_upper += 1
            
            # Add a peak with the sum of the smeared peaks into the new weight vector
            w_hat_vector_merged[index_max] = np.sum(w_hat_vector_stripped[index_lower:index_upper + 1])
            
            # Remove the merged data from the "working"-vector
            w_hat_vector_stripped[index_lower:index_upper + 1] = 0
            
            # Break the cycle
            if index_max == 0: break
            
        # Set merged list as weight vector
        # This does not necessarly make sense for Debye DRTs, so the merging is skipped
        if not np.array_equal(self._get_basis_matrix(), np.identity(len(self.tau_s))):
            w_hat_vector = w_hat_vector_merged
        
        # Return the refined weight vector
        return w_hat_vector            
        
    def setup_time_constants(self, tau_range_min_s: float = None, tau_range_max_s: float = None, tau_points_per_decade: int = None) -> DRT:
        """
        Method to determine the time constants for the DRT computation

        Args:
            tau_range_min_s (float, optional): Minimum time constant. Defaults to (see above).
            tau_range_max_s (float, optional): Maxmium time constant. Defaults to (see above).
            tau_points_per_decade (int, optional): Points per decade. Defaults to (see above).

        Returns:
            DRT (chaining)
        """
        
        # Get values (or defaults)
        if tau_range_min_s is None:
            tau_range_min_s = self.tau_range_min_s 
            
        if tau_range_max_s is None: 
            tau_range_max_s = self.tau_range_max_s 
            
        if tau_points_per_decade is None:
            tau_points_per_decade = self.tau_points_per_decade
        
        # Determine number of points (called m in the corresponding chapter)
        m: int = np.floor(tau_points_per_decade * np.log10(tau_range_max_s / tau_range_min_s))
        
        # Calculate log10-spaced time constants
        # NOTE: This does more or less the same as np.logspace()
        self.__time_constant_array: npt.ArrayLike = np.array([tau_range_min_s * 10**(i/tau_points_per_decade) for i in np.arange(m)])
        
        return self
    
    
    def _get_rc_kernel_matrix(self) -> npt.ArrayLike:
        """
        Method to get the default (RC) kernel matrix

        Returns:
            npt.ArrayLike: Kernel matrix
        """
        
        # Calculate difference between two time constants (Δln𝜏)
        # This is put into the kernel for convenience 
        # NOTE: ln(a/b) = ln(a) - ln(b)
        delta_ln_tau_tau0: float = np.log(self.tau_s[1]) - np.log(self.tau_s[0])
        
        # Calculate kernel matrix
        kernel_matrix: npt.ArrayLike = delta_ln_tau_tau0 / (1 + 2j * np.pi * np.outer(self.__frequency_data_Hz.T, self.tau_s))
        
        return kernel_matrix
    
    def _get_regularization_matrix(self) -> npt.ArrayLike:
        """
        Method to setup the regularization matrix
        NOTE: This needs to eventually be multiplied with the regularization parameter

        Returns:
            npt.ArrayLike: Regularization matrix
        """
        return np.identity(len(self.tau_s))
    
    def _get_basis_matrix(self) -> npt.ArrayLike:
        raise NotImplementedError()
    
    def get_separated_peak_list(self, max_peak_number: int = None, min_tau_s: float = 1e-12, max_tau_s: float = 1e12, sort_by_tau: bool = True) -> list[DRTPeak]:
        """
        Method to get peaks from gamma (for further analysis)
        
        Args:
            max_peak_number (int, optional): Maximum number of maxima that shall be considered; only makes sense, if sorting is enabled. Defaults to None.
            min_tau_s (float, optional): Minimum tau that shall be considered. Defaults to 1e-12
            max_tau_s (float, optional): Maximum tau that shall be considered. Defaults to 1e12.
            sort_by_tau (bool, optional): If True, list is sorted by tau. Defaults to True
        
        Returns:
            list[DRTPeak]: List with peak objects
        """
        
        # Get data
        weight_vector    = self.weight_vector
        basis_matrix     = self._get_basis_matrix()
        
        # Setup lists with gamma traces
        peak_list = []
        
        # Iterate through gamma indices
        for index in np.where(weight_vector > 0)[0]:
            # Setup a sparse weight vector which we'll populate only for the tau that we're interested in
            sparse_weight_vector = np.zeros_like(weight_vector)
            
            # Get tau
            tau_s = self.tau_s[index]
            
            # Filter for valid taus
            if ~(min_tau_s < tau_s < max_tau_s):
                continue
            
            # Populate sparse weight vector
            sparse_weight_vector[index] = weight_vector[index]
            
            # Calculate DRT for the peak
            gamma_peak_Ohm = self.R_pol_Ohm * np.dot(basis_matrix, sparse_weight_vector)
            
            # Calculate R and C
            R_Ohm = np.trapz(gamma_peak_Ohm, self.ln_tau_tau0)
            C_F   = tau_s / R_Ohm
            
            # Filter for valid resistances (below 0.1 mOhm is hardly measureable for common batteries)
            if R_Ohm < 1e-4:
                continue
            
            peak_list.append(DRTPeak(tau_s, R_Ohm, C_F, weight_vector[index], gamma_peak_Ohm))
        
        # Sort list by R and keep highest n peaks
        if max_peak_number:
            peak_list.sort(key = lambda peak: peak.R_Ohm, reverse = 1)
            peak_list = peak_list[:max_peak_number]
        
        # Sort again by tau if requested
        if sort_by_tau:
            peak_list.sort(key = lambda peak: peak.tau_s, reverse = 0)
            
        return peak_list
    
    @staticmethod 
    def optimize_regularization_parameters(drt: DRT, epsilon_min: float = 1e-8, epsilon_max: float = 1e3, relative_residual_threshold: float = 0.01, tol = 1e-6) -> DRT:
        # Reset threshold if necessary
        drt.solve(epsilon = False)
        if drt.relative_resnorm > relative_residual_threshold:
            print(f'Even overfitted, the threshold {relative_residual_threshold} is unreachable. Threshold will be set to 1.5 times the relative resnorm of the unregularized solution (hence, {1.5 * drt.relative_resnorm:.4f}).')
            relative_residual_threshold = 1.5 * drt.relative_resnorm
            
        
        # Use binary search to find the optimum parameter
        low, high = epsilon_min, epsilon_max
        
        while high - low > tol:
            mid = (low + high) / 2.0
            
            # Calculate epsilon
            epsilon_mid = mid
            
            # Calculate DRT
            drt.solve(epsilon = epsilon_mid)
            
            if drt.relative_resnorm < relative_residual_threshold:
                # Try larger values
                low = mid 
            else:
                # Try smaller values
                high = mid
        
        drt.epsilon = low
        drt.solve()
        
        print(f'Optimal DRT regularization: \t epsilon = {drt.epsilon:.4f}, \t res = {drt.relative_resnorm:.6f}')
        
        return drt
    
    # Number of non-zero (with a threshold) elements in w_hat
    def get_number_of_w_hat_values(self, threshold_factor: float = 0.02) -> int:
        return np.sum(self.weight_hat_Ohm > self.weight_hat_Ohm.max() * threshold_factor)
        
    ## Class properties
    # Time constants
    @property 
    def tau_s(self) -> npt.ArrayLike:
        return self.__time_constant_array
    
    @property 
    def ln_tau_tau0(self) -> npt.ArrayLike:
        return np.log(self.__time_constant_array / 1) # The /1 is only for the show here
    
    # Weights and gamma
    @property
    def weight_vector(self) -> npt.ArrayLike:
        return self.__weight_vector
    
    @property 
    def weight_hat_Ohm(self) -> npt.ArrayLike:
        return self.__w_hat_vector
    
    @property
    def gamma(self) -> npt.ArrayLike:
        return self.__gamma_array
    
    @property 
    def gamma_hat_Ohm(self) -> npt.ArrayLike:
        return self.__gamma_hat_Ohm
    
    @property
    def basis_matrix(self) -> npt.ArrayLike:
        return self._get_basis_matrix()
    
    
    # Resistances, impedances and frequency data
    @property 
    def R_offset_Ohm(self) -> float:
        return self.__ohmic_resistance_Ohm
    
    @property
    def R_pol_Ohm(self) -> float:
        return self.__R_pol_Ohm
    
    @property
    def frequency_data_orig_Hz(self) -> npt.ArrayLike:
        return self.__frequency_data_orig_Hz
    
    @property 
    def z_data_orig_Ohm(self) -> npt.ArrayLike:
        return self.__impedance_data_orig_Ohm
    
    @property 
    def z_data_Ohm(self) -> npt.ArrayLike:
        return self.__impedance_data_Ohm + self.__ohmic_resistance_Ohm
    
    @property 
    def z_back_Ohm(self) -> npt.ArrayLike:
        return self.__impedance_back_Ohm   
        
    # Some more attributes than can be used to assess the goodness of the DRT computation
    @property 
    def z_back_stacked_Ohm(self) -> npt.ArrayLike:
        return np.array([*self.z_back_Ohm.real, *self.z_back_Ohm.imag])
    
    @property
    def z_data_stacked_Ohm(self) -> npt.ArrayLike:
        return np.array([*self.z_data_Ohm.real, *self.z_data_Ohm.imag])
    
    @property 
    def z_diff_stacked_Ohm(self) -> npt.ArrayLike:
        return self.z_back_stacked_Ohm - self.z_data_stacked_Ohm
        
    @property
    def resnorm_nnls_Ohm(self) -> float:
        return self.__norm
    
    @property 
    def resnorm_Ohm(self) -> float:
        # The same as the second return value of the nnls method
        # NOTE: Small values are better
        # return np.linalg.norm(self.z_diff_stacked_Ohm)
        return self.resnorm_nnls_Ohm
    
    @property 
    def relative_resnorm(self) -> float:
        # NOTE: Small value (-> 0) are great, large values (-> 1) are bad
        return self.resnorm_Ohm / np.linalg.norm(self.z_data_stacked_Ohm)
    
    @property 
    def r_squared(self) -> float:
        # NOTE: The closer to 1 the better
        return 1 - (np.sum(self.z_diff_stacked_Ohm**2) / np.sum((self.z_data_stacked_Ohm - np.mean(self.z_data_stacked_Ohm))**2))
        
    @property
    def rmse_Ohm(self) -> float:
        # NOTE: The smaller the better
        return np.sqrt(np.mean(self.z_diff_stacked_Ohm**2))
    
    @property
    def number_of_nonzero_w_hat_values(self) -> int:
        return self.get_number_of_w_hat_values(threshold_factor = 0)
    