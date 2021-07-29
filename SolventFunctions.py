#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 17:30:34 2020

@author: jbeckwith
"""
import numpy as np

class SolventFunc():
    def __init__(self):
        self = self
    
    @staticmethod
    def waterrefractiveindex(wavelength, T, rho):
        # This function takes a wavelength (in nanometers), temperature (in
        # Kelvin), and density (in g/mL) and returns the refractive index of
        # water as calculated using the equation in IAPWS "Release on the
        # Refractive Index of Ordinary Water Substance as a Function of Wavelength,
        # Temperature, and Pressure" (1997).

        #  The recommended ranges for this function are:
        # -12 Celsius <= T <= 500 Celsius
        # 0 kg*m^-3 <= rho <= 1060 kg*m^-3
        # 200 nm <= lambda <= 1100 nm
        np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
        T_star = 273.15
        rho_star = 1000
        wavelength_star = 0.589

        T_bar = np.divide(T, T_star)
        rho_bar = np.divide(np.multiply(rho,1000), rho_star)
        wavelength_bar = np.divide(np.divide(wavelength, 1000), wavelength_star)
        
        a0 = 0.244257733
        a1 = 9.74634476e-3
        a2 = -3.73234996e-3
        a3 = 2.68678472e-4
        a4 = 1.58920570e-3
        a5 = 2.45934259e-3
        a6 = 0.900704920
        a7 = -1.66626219e-2
        wavelength_UV = 0.2292020
        wavelength_IR = 5.432937
        
        a = np.reciprocal(rho_bar)
        
        wl_square = np.square(wavelength_bar)
        
        t1 = a0
        t2 = np.multiply(a1, rho_bar)
        t3 = np.multiply(a2, T_bar)
        t4 = np.multiply(np.multiply(a3, wl_square), T_bar)
        t5 = np.divide(a4, wl_square)
        t6 = np.divide(a5, np.subtract(wl_square, np.square(wavelength_UV)))
        t7 = np.divide(a6, np.subtract(wl_square, np.square(wavelength_IR)))
        t8 = np.multiply(a7, np.square(rho_bar))
        
        b = np.sum([t1, t2, t3, t4, t5, t6, t7, t8])

        n = np.sqrt(np.divide(np.add(np.multiply(2, b), a), np.subtract(a, b)))
        return n
    
    @staticmethod
    def glycerolrefractiveindex(wavelength):
        from scipy.constants import h, c, e
        #This function takes the data from Birkhoff, R. D. et al. J. Chem. Phys.
        # 69, 4185 (1978) and calulates, using linear interpolation, the refractive
        # index of glycerol for a given wavelength (in nanometers) over the range
        # 51.2 nm - 619.9 nm
        #
        # Above 619.9 nm, the function from Rheims, J. et al. Meas.
        # Sci. Technol. 8, 601 (1997) is used instead, valid over the range >619.9
        # nm - 1050 nm.

        # Light energies in eV as given by Birkhoff et al.
        np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
        energies = np.hstack([np.arange(2,25), 24.2])
        n_list = np.array([1.47, 1.47, 1.48, 1.50, 1.56, 1.65, 1.76, 1.87, 1.98, 2.01, 1.94, 1.83, 1.70, 1.57, 1.44, 1.32, 1.22, 1.14, 1.07, 1.01, 0.968, 0.944, 0.939, 0.937, 0.937])[::-1]
 #       k_list = np.array([0, 0, 0, 0, 0, 0, 0, 0.121, 0.335, 0.515, 0.646, 0.737, 0.793, 0.809, 0.800, 0.774, 0.736, 0.690, 0.635, 0.576, 0.517, 0.469, 0.437, 0.433])
        wavelengths = np.hstack([1050, np.multiply(np.divide(np.multiply(h, c), np.multiply(energies, e)), 1e9)])[::-1]
        if np.all(wavelength <= 619.9):
            #convert to wavelength in nm
            n_real_B = np.interp(wavelength, wavelengths, n_list)
            return n_real_B
        else:
            n_real_B =  np.interp(wavelength[np.where(wavelength <= 619.9)], wavelengths, n_list)
            n_real_619 = np.interp(619.9, wavelengths, n_list)
           
            A = 1.45797
            B = 0.00598e-6
            C = -0.00036e-12
            
            # Note that the Rheims curve is adjusted so as to avoid a 0.8%
            # discontinuity with the Birkhoff data at lambda = 619.9 nm
            t1 = A
            t2 = np.divide(B, np.square(wavelength[np.where(wavelength > 619.9)]))
            t3 = np.divide(C, np.power(wavelength[np.where(wavelength > 619.9)], 4))
            t4 = np.subtract(n_real_619, np.add(np.add(A, np.divide(B, np.square(619.9))), np.divide(C, np.power(619.9, 4))))
            
            n_real_R = np.sum([t1, t2, t3, t4])
            n_real_R = np.hstack([n_real_B, n_real_R])
            return n_real_R

    
    @staticmethod
    def waterdensity(T):
        # Returns density (in g/mL) of pure, air-saturated water for a given temperature (in Celsius) using
        # equation from Jones et al. J. Res. Natl. Inst. Stand. Technol. 97, 335
        # (1992). Valid over the range 5 Celsius <= T <= 40 Celsius.
        t1 = 999.84847
        t2 = np.multiply(6.337563e-2, T)
        t3 = np.multiply(-8.523829e-3, np.square(T))
        t4 = np.multiply(6.943248e-5, np.power(T, 3))
        t5 = np.multiply(-3.821216e-7, np.power(T,4))
        
        rho = np.divide(np.sum([t1, t2, t3, t4, t5]), 1000)        
        return rho
    
    @staticmethod
    def glyceroldensity(T):
        # Returns, using linear interpolation, density (in g/mL) of pure glycerol for a given temperature (in Celsius) using
        # data from "Physical Properties of Glycerine and its Solutions" Glycerine Producers' Association: New
        # York, 1963. Valid over the range 0 Celsius <= T <= 290 Celsius.

        # Temperatures
        T_list = np.array([0, 10, 15, 20, 30, 40, 54, 75.5, 99.5, 110, 120, 130, 140, 160, 180, 200, 220, 240, 260, 280, 290])
        rho_list = np.array([1.27269, 1.26699, 1.26443, 1.26134, 1.25512, 1.24896, 1.2397, 1.2256, 1.2097, 1.20178, 1.19446, 1.18721, 1.17951, 1.16440, 1.14864, 1.13178, 1.11493, 1.09857, 1.08268, 1.06725, 1.05969])
        
        rho = np.interp(T, T_list, rho_list)
        return rho
    
    @staticmethod
    def wglycerolfromvisc(eta_0):
         # The function used to calculate the %w glycerol from the viscosity was
         # derived from the equations found in the following reference and with T = 20 C:
         # Cheng, N. Ind. Eng. Chem. Res. 47, 3285 (2008).
         t1 = 92.27204742509134
         t2 = np.multiply(8.847150502255046, np.log(eta_0))
         t3 = np.multiply(-1.7845903636318703, np.sqrt(np.multiply(np.add(2616.684662088474, np.log(eta_0)), np.add(504.4410011392123, np.multiply(24.577049457486897, np.log(eta_0))))))
         
         wgly = np.sum([t1, t2, t3])
         return wgly

        
