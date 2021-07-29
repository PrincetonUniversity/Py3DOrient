#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 17:40:08 2020

@author: jbeckwith
"""
import numpy as np
from scipy.constants import constants
from scipy.constants import physical_constants
heV =  physical_constants['Planck constant in eV/Hz'][0]
from SolventFunctions import SolventFunc
solvf = SolventFunc()

class Permittivities():
    def __init__(self, element):
        # Initialises Class
        # ------ INPUTS ------ #
        # element (string)
        # ------ OUTPUT ------ #
        # sets up class with self.element
        if isinstance(element, str):
            test_list = ['Au', 'Ag', 'Cu', 'Al'] 
            res = any(ele in element for ele in test_list)
            if res:
                self.element = element
            else:
                print("Error. Element not Recognised.")
                return
        else:
            print("Error. Not a string.")
            
    def metal_properties_yu(self):
        # Relative Permittivity properties for particle (chosen by defining variable 'element' above).
        # Metal Property equations come from paper of Yu, R.; Liz-Marzán, L. M.; Abajo, F. J. G. de.
        # Universal Analytical Modeling of Plasmonic Nanoparticles.
        # Chem. Soc. Rev. 2017, 46 (22), 6710–6724. https://doi.org/10.1039/C6CS00919K.
        if self.element == 'Au':
            hbartau = 0.071
            hbartau1 = 0.0716
            hbaromega1 = 2.43
            hbaromega2 = 1.52
            A = 0.132
            B = -1.755
            C = 20.43
            omega_p = 9.06
        elif self.element == 'Ag':
            hbartau = 0.021
            hbartau1 = 0.0760
            hbaromega1 = 4.02
            hbaromega2 = 18.5
            A = -9.71
            B = -1.111
            C = 13.77
            omega_p = 9.17
        elif self.element == 'Cu':
            hbartau = 0.103
            hbartau1 = 0.0528
            hbaromega1 = 2.12
            hbaromega2 = 5.43
            A = -4.36
            B = -1.655
            C = 12.31
            omega_p = 8.88
        return hbartau, hbartau1, hbaromega1, hbaromega2, A, B, C, omega_p

    def metal_properties_radic_ld(self):
        # Relative Permittivity properties for particle (chosen by defining variable 'element' above).
        # Metal Property equations come from Rakić, A. D.; Djurišić, A. B.; Elazar, J. M.; Majewski, M. L.
        # Appl. Opt. 1998, 37 (22), 5271–5283.
        # these parameters for Lorentz-Drude model (equation 3)
        if self.element == 'Au':
            omegap = 9.03
            f0 = 0.760
            Omegap = np.multiply(np.sqrt(f0), omegap)
            Gamma0 = 0.053
            fj = np.array([0.024, 0.010, 0.071, 0.601, 4.384])
            Gammaj = np.array([0.241, 0.345, 0.870, 2.494, 2.214])
            omegaj = np.array([0.415, 0.830, 2.969, 4.304, 13.32])
        elif self.element == 'Ag':
            omegap = 9.01
            f0 = 0.845
            Omegap = np.multiply(np.sqrt(f0), omegap)
            Gamma0 = 0.048
            fj = np.array([0.065, 0.124, 0.011, 0.840, 5.646])
            Gammaj = np.array([3.886, 0.452, 0.065, 0.916, 2.419])
            omegaj = np.array([0.816, 4.481, 8.185, 9.083, 20.29])           
        elif self.element == 'Cu':
            omegap = 10.83
            f0 = 0.575
            Omegap = np.multiply(np.sqrt(f0), omegap)
            Gamma0 = 0.030
            fj = np.array([0.061, 0.104, 0.723, 0.638])
            Gammaj = np.array([0.378, 1.056, 3.213, 4.305])
            omegaj = np.array([0.291, 2.957, 5.300, 11.18])
        elif self.element == 'Al':
            omegap = 14.98
            f0 = 0.523
            Omegap = np.multiply(np.sqrt(f0), omegap)
            Gamma0 = 0.047
            fj = np.array([0.227, 0.050, 0.166, 0.030])
            Gammaj = np.array([0.333, 0.312, 1.351, 3.382])
            omegaj = np.array([0.162, 1.544, 1.808, 3.473])
        return omegap, f0, Omegap, Gamma0, fj, Gammaj, omegaj

    def metal_properties_radic_bb(self):
        # Relative Permittivity properties for particle (chosen by defining variable 'element' above).
        # Metal Property equations come from Rakić, A. D.; Djurišić, A. B.; Elazar, J. M.; Majewski, M. L.
        # Appl. Opt. 1998, 37 (22), 5271–5283.
        # these parameters for Brendel–Bormann model (equation 11)
        if self.element == 'Au':
            omegap = 9.03
            f0 = 0.77
            Omegap = np.multiply(np.sqrt(f0), omegap)
            Gamma0 = 0.05
            fj = np.array([0.054, 0.05, 0.312, 0.719, 1.648])
            Gammaj = np.array([0.074, 0.035, 0.083, 0.125, 0.179])
            omegaj = np.array([0.218, 2.885, 4.069, 6.137, 27.97])  
            sigmaj = np.array([0.742, 0.349, 0.830, 1.246, 1.795])
        elif self.element == 'Ag':
            omegap = 9.01
            f0 = 0.821
            Omegap = np.multiply(np.sqrt(f0), omegap)
            Gamma0 = 0.049
            fj = np.array([0.050, 0.133, 0.051, 0.467, 4])
            Gammaj = np.array([0.189, 0.067, 0.019, 0.117, 0.052])
            omegaj = np.array([2.2025, 5.185, 4.343, 9.809, 18.56])  
            sigmaj = np.array([1.894, 0.665, 0.189, 1.170, 0.516])
        elif self.element == 'Cu':
            omegap = 10.83
            f0 = 0.562
            Omegap = np.multiply(np.sqrt(f0), omegap)
            Gamma0 = 0.030
            fj = np.array([0.076, 0.081, 0.324, 0.726])
            Gammaj = np.array([0.056, 0.047, 0.113, 0.172])
            omegaj = np.array([0.416, 2.849, 4.819, 8.136])
            sigmaj = np.array([0.562, 0.469, 1.131, 1.719])
        elif self.element == 'Al':
            omegap = 14.98
            f0 = 0.526
            Omegap = np.multiply(np.sqrt(f0), omegap)
            Gamma0 = 0.047
            fj = np.array([0.213, 0.060, 0.182, 0.014])
            Gammaj = np.array([0.312, 0.315, 1.587, 2.145])
            omegaj = np.array([0.163, 1.561, 1.827, 4.495])
            sigmaj = np.array([0.013, 0.042, 0.256, 1.735])
        return omegap, f0, Omegap, Gamma0, fj, Gammaj, omegaj, sigmaj
    
    def geometry_factors(self, shape, R):
        # Outputs particle properties based on shape and R (L/W)
        # ------ INPUTS ------ #
        # shape (string) - one of 'Rod', 'Triangle', 'Cage', 'Ellipsoid', 'Bicone', 'Disk', 'Ring' 'Bipyramid', 'Squared Rod', 'Cylinder'
        # R = Length/Width of Shape
        # ------ OUTPUTS ------ #
        # eps_long epsilon of longitudinal mode
        # V1 volume of longitudinal mode
        # a12_long a12 parameter for longitudinal mode
        # a14_long a14 parameter for longitudinal mode
        # Vparticle volume of particle in units of L^3
        # eps_trans1 epsilon of transverse mode 1
        # eps_trans2 epsilon of transverse mode 2
        # eps_trans3 epsilon of transverse mode 3
        # Vtrans1 volume of transverse mode 1
        # Vtrans2 volume of transverse mode 2
        # Vtrans3 volume of transverse mode 3
        # a12_trans1 a12 parameter for transverse mode 1
        # a12_trans2 a12 parameter for transverse mode 2
        # a12_trans3 a12 parameter for transverse mode 3
        # a14_trans1 a14 parameter for transverse mode 1
        # a14_trans2 a14 parameter for transverse mode 2
        # a14_trans3 a14 parameter for transverse mode 3   
        accepted = ['Rod', 'Triangle', 'Cage', 'Ellipsoid', 'Bicone', 'Disk', 'Ring', 'Bipyramid', 'Squared Rod', 'Cylinder']
        if shape not in accepted:
            print('Shape not Recognised.')
            return
        if shape == 'Rod':
            # longitudinal mode for Rod
            eps_long = np.subtract((np.multiply(-1.73, np.power(R, 1.45))), 0.296)
            V1 = 0.896
            a12_long = np.divide(6.92, np.subtract(1., eps_long))
            a14_long = np.subtract(np.divide(-11, np.power(R, 2.49)), 0.0868)
            
            # transverse modes for Rod
            eps_trans1 = np.add(-1.75, np.divide(3.19, np.power(R, 6.14)))
            eps_trans2 = np.subtract(-0.978, np.divide(0.661, np.power(R, 1.1)))
            eps_trans3 = np.add(-1.57, np.multiply(0.0446, R))
            Vtrans1 = np.add(0.0679, np.divide(1.83, np.power(R, 2.1)))
            Vtrans2 = np.subtract(0.891, np.divide(2.28, np.power(R, 2.53)))
            Vtrans3 = np.add(-0.0346, np.multiply(0.0111, R))
            a12_trans1 = np.add(0.0148, np.divide(3.69, np.power(R, 2.86)))
            a12_trans2 = np.add(-21.7, np.divide(22.7, np.power(R, 0.0232)))
            a12_trans3 = np.add(-0.0117, np.divide(0.773, np.power(R, 1.46)))
            a14_trans1 = np.subtract(0.0142, np.divide(16.9, np.power(R, 3.58)))
            a14_trans2 = np.subtract(1.48, np.divide(3.67, np.power(R, 0.48)))
            a14_trans3 = np.add(-0.256, np.multiply(0.0554, np.power(R, 0.758)))
            
            Vparticle = lambda R, L: np.multiply(np.divide((np.multiply(np.pi, np.subtract(np.multiply(3., R), 1.)))\
                                      , np.multiply(12., np.power(R, 3.))), np.power(L, 3.))
        elif shape == 'Triangle':
            # mode for Triangle
            eps_long = np.subtract(np.multiply(-0.87, np.power(R, 1.12)), 4.33)
            V1 = np.add(np.multiply(-0.645, np.power(R, -1.24)), 0.678)
            a12_long = np.divide(5.57, np.subtract(1, eps_long))
            a14_long = np.divide(-6.83, np.subtract(1, eps_long))
            
            Vparticle = lambda R, L: np.multiply(np.add(np.divide(-0.00544, np.square(R)),\
                                     np.divide(0.433, R)), np.power(L, 3))
        elif shape == 'Cage':
            # mode for Cage
            eps_long = np.subtract(np.multiply(-0.0678, np.power(R, 2.02)), 3.42)
            V1 = np.add(np.add(np.multiply(-0.008, np.square(R)), np.multiply(1.103, R)), 0.316)
            a12_long = np.add(np.multiply(-0.00405, np.power(R, 2.59)), 2.21)
            a14_long = -13.9
            
            Vparticle = lambda R, L: np.multiply(np.subtract(np.add(np.subtract(np.divide(8.04, np.power(R, 3.)),\
                                     np.divide(12., np.square(R))), np.divide(6., R)), 0.00138), np.power(L, 3.))                
        elif shape == 'Ellipsoid':
            # mode for Ellipsoid
            eps_long = np.subtract(-0.871, np.multiply(1.35, np.power(R, 1.54)))
            V1 = 0.994
            a12_long = np.divide(1.34, np.subtract(1., eps_long))
            a14_long = np.divide(-1.04, np.subtract(1., eps_long))
            
            Vparticle = lambda R, L: np.multiply(np.divide(np.pi, np.multiply(6., np.square(R))), np.power(L, 3.))
        elif shape == 'Bicone':
            # mode for Bicone
            eps_long = np.subtract(-0.687, np.multiply(2.54, np.power(R, 1.5)))
            V1 = np.subtract(0.648, np.divide(0.441, np.power(R, 0.687)))
            a12_long = np.divide(1.34, np.subtract(1., eps_long))
            a14_long = np.divide(-1.04, np.subtract(1., eps_long))
            
            Vparticle = lambda R, L: np.multiply(np.divide(0.262, np.square(R)), np.power(L, 3.))
        elif shape == 'Disk':
            # mode for Disk
            eps_long = np.subtract(-0.479, np.multiply(1.36, np.power(R, 0.98)))
            V1 = 0.944
            a12_long = np.divide(7.05, np.subtract(1., eps_long))
            a14_long = np.divide(-10.9, np.power(R, 0.98))
            
            Vparticle = lambda R, L: np.multiply(np.divide(np.multiply(np.pi, np.add(4.,\
                                     np.multiply(3., np.multiply(np.subtract(R, 1),\
                                     np.add(np.multiply(2., R), np.subtract(np.pi, 2)))))), \
                                     np.multiply(24., np.power(R, 3.))), np.power(L, 3.))
        elif shape == 'Ring':
            # mode for Ring
            eps_long = np.subtract(1.39, np.multiply(1.31, np.power(R, 1.73)))
            V1 = np.add(0.514, np.divide(2.07, np.power(2.67)))
            a12_long = np.divide(7.24, np.subtract(1., eps_long))
            a14_long = np.divide(-19.1, np.subtract(1., eps_long))
            Vparticle = lambda R, L: np.multiply(np.divide(np.multiply(np.square(np.pi), np.subtract(R, 1.)),\
                                     np.multiply(4., np.power(R, 3.))), np.power(L, 3.))
        elif shape == 'Bipyramid':
            # mode for Bipyramid
            eps_long = np.subtract(1.43, np.multiply(4.52, np.power(R, 1.12)))
            V1 = np.subtract(1.96, np.divide(1.73, np.power(R, 0.207)))
            a12_long = np.divide(2.89, np.subtract(1, eps_long))
            a14_long = np.divide(-1.79, np.subtract(1, eps_long))
            Vparticle = lambda R, L: np.multiply(np.divide(0.219, np.square(R)), np.power(L, 3.))
        elif shape == 'Squared Rod':
            # mode for Squared Rod
            eps_long = np.subtract(-2.28, np.multiply(1.47, np.power(R, 1.49)))
            V1 = np.subtract(0.904, np.divide(0.411, np.power(R, 2.26)))
            a12_long = np.add(-0.573, np.divide(3.31, np.power(R, 0.747)))
            a14_long = np.subtract(0.213, np.divide(13.1, np.power(R, 1.97)))
            Vparticle = lambda R, L: np.multiply(np.reciprocal(np.square(R)), np.power(L, 3.))
        elif shape == 'Cylinder':
            # mode for Cylinder
            eps_long = np.subtract(-1.59, np.multiply(1.96, np.power(R, 1.4)))
            V1 = np.subtract(0.883, np.divide(0.149, np.power(R, 3.97)))
            a12_long = np.add(-1.05, np.divide(3.02, np.power(R, 0.494)))
            a14_long = np.subtract(0.0796, np.divide(9.08, np.power(R, 2.08)))
            Vparticle = lambda R, L: np.multiply(np.divide(np.pi, np.multiply(4., np.square(R))), np.power(L, 3.))
            
        if shape != 'Rod':
            # transverse modes for non-rod
            eps_trans1 = 0
            eps_trans2 = 0
            eps_trans3 = 0
            Vtrans1 = 0
            Vtrans2 = 0
            Vtrans3 = 0
            a12_trans1 = 0
            a12_trans2 = 0
            a12_trans3 = 0
            a14_trans1 = 0
            a14_trans2 = 0
            a14_trans3 = 0
        return eps_long, V1, a12_long, a14_long, Vparticle, \
                eps_trans1, eps_trans2, eps_trans3, Vtrans1, \
                Vtrans2, Vtrans3, a12_trans1, a12_trans2, a12_trans3, \
                a14_trans1, a14_trans2, a14_trans3
    
    def lorentz_drude_rakic_dielectric(self, wavelength): # takes wavelength in nm
        # Outputs epsilon of particle according to Lorentz-Drude model parameterised as in
        # Rakić, A. D.; Djurišić, A. B.; Elazar, J. M.; Majewski, M. L.
        # Appl. Opt. 1998, 37 (22), 5271–5283
        # ------ INPUTS ------ #
        # wavelength (in nm)
        # ------ OUTPUTS ------ #
        # eps_p epsilon of metal at specified wavelengths
        energy = np.divide(np.multiply(heV, constants.c), np.multiply(wavelength, 1e-9))
        omegap, f0, Omegap, Gamma0, fj, Gammaj, omegaj = self.metal_properties_radic_ld()
        eps_r_f = np.subtract(1, np.divide(np.square(Omegap), np.multiply(energy, np.add(energy, np.multiply(1j, Gamma0)))))
        if isinstance(wavelength, float):
            eps_r_b = np.sum(np.divide(np.multiply(fj, np.square(omegap)),\
                    np.subtract(np.subtract(np.square(omegaj),\
                    np.square(energy[np.newaxis].T)),\
                    np.multiply(1j, (np.multiply(Gammaj , energy[np.newaxis].T))))))
        else:
            eps_r_b = np.sum(np.divide(np.multiply(fj, np.square(omegap)),\
            np.subtract(np.subtract(np.square(omegaj),\
            np.square(energy[np.newaxis].T)),\
            np.multiply(1j, (np.multiply(Gammaj , energy[np.newaxis].T))))), axis=1)
        eps_p = np.add(eps_r_f, eps_r_b)
        return eps_p

    def brendel_bormann_rakic_dielectric(self, wavelength): # takes wavelength in nm
        # Outputs epsilon of particle according to Brendel–Bormann model parameterised as in
        # Rakić, A. D.; Djurišić, A. B.; Elazar, J. M.; Majewski, M. L.
        # Appl. Opt. 1998, 37 (22), 5271–5283
        # ------ INPUTS ------ #
        # wavelength (in nm)
        # ------ OUTPUTS ------ #
        # eps_p epsilon of metal at specified wavelengths
        energy = np.divide(np.multiply(heV, constants.c), np.multiply(wavelength, 1e-9))
        omegap, f0, Omegap, Gamma0, fj, Gammaj, omegaj, sigmaj = self.metal_properties_radic_bb()
        eps_r_f = np.subtract(1, np.divide(np.square(Omegap), np.multiply(energy, np.add(energy, np.multiply(1j, Gamma0)))))
        ajdash = energy[np.newaxis].T/np.sqrt(2) * np.sqrt((np.sqrt(1 + (Gammaj/energy[np.newaxis].T))) + 1)
        ajdash2 = energy[np.newaxis].T/np.sqrt(2) * np.sqrt((np.sqrt(1 + (Gammaj/energy[np.newaxis].T))) - 1)
        aj = np.add(ajdash, np.multiply(1j, ajdash2))
        z1 = (np.nan_to_num(-np.square(np.divide(np.subtract(aj, omegaj), np.multiply(np.sqrt(2), sigmaj)))))
        z2 = (np.nan_to_num(-np.square(np.divide(np.add(aj, omegaj), np.multiply(np.sqrt(2), sigmaj)))))
        from scipy.special import erfc
        if isinstance(wavelength, float):            
            erfterm1 = np.nan_to_num(erfc(np.sqrt(z1)))
            erfterm2 = np.nan_to_num(erfc(np.sqrt(z2)))
            Unegterm = np.nan_to_num(np.multiply(np.multiply(np.sqrt(np.pi), np.exp(z1)), erfterm1))
            Uposterm = np.nan_to_num(np.multiply(np.multiply(np.sqrt(np.pi), np.exp(z2)), erfterm2))
            chi_j = np.sum(np.multiply(np.divide(np.multiply(np.multiply(1j, fj), np.square(omegap)),\
                    np.multiply(2., np.multiply(np.multiply(np.sqrt(2), aj), sigmaj))),\
                    np.add(Unegterm, Uposterm)))
        else:
            erfterm1 = np.nan_to_num(erfc(np.sqrt(z1)))
            erfterm2 = np.nan_to_num(erfc(np.sqrt(z2)))
            Unegterm = np.nan_to_num(np.multiply(np.multiply(np.sqrt(np.pi), np.exp(z1)), erfterm1))
            Uposterm = np.nan_to_num(np.multiply(np.multiply(np.sqrt(np.pi), np.exp(z2)), erfterm2))
            chi_j = np.sum(np.multiply(np.divide(np.multiply(np.multiply(1j, fj), np.square(omegap)),\
                    np.multiply(2., np.multiply(np.multiply(np.sqrt(2), aj), sigmaj))),\
                    np.add(Unegterm, Uposterm)), axis=1)
        eps_p = np.add(eps_r_f, chi_j)
        return eps_p
    
    def yu_dielectric(self, wavelength):
        # Outputs epsilon of particle according to Lorentz-Drude model parameterised as in
        # Yu, R.; Liz-Marzán, L. M.; Abajo, F. J. G. de.
        # Universal Analytical Modeling of Plasmonic Nanoparticles.
        # Chem. Soc. Rev. 2017, 46 (22), 6710–6724.
        # ------ INPUTS ------ #
        # wavelength (in nm)
        # ------ OUTPUTS ------ #
        # eps_p epsilon of metal at specified wavelengths
        energy = np.divide(np.multiply(heV, constants.c), np.multiply(wavelength, 1e-9))
        hbartau, hbartau1, hbaromega1, hbaromega2, A, B, C, omega_p = self.metal_properties_yu()
        term2 = np.multiply(B, np.log(np.divide(np.subtract(np.subtract(hbaromega1, energy), np.multiply(1j, hbartau1)),\
                              np.add(np.add(hbaromega1, energy), np.multiply(1j, hbartau1)))))
        eps_b = np.add(np.add(A, term2), np.multiply(C, np.exp(-np.divide(energy, hbaromega2))))
                           
        eps_p = eps_b - np.divide(np.square(omega_p), np.multiply(energy, np.add(energy, np.multiply(1j,hbartau)))) # permittivity of the particle medium
        return eps_p
    
    def eps_m(self, wavelength, T, cv): # wavelength (in nanometers), temperature (in Kelvin), and density (in g/mL)
        rhow = solvf.waterdensity(np.subtract(T, 273.15))
        n0w = solvf.waterrefractiveindex(wavelength, T, rhow)
        n0g = solvf.glycerolrefractiveindex(wavelength)
        n0g[np.isnan(n0g)] = 0
        n_m = np.sqrt(np.add(np.multiply(cv, np.square(n0g)), np.multiply(np.subtract(1., cv), np.square(n0w))))
        return np.square(n_m)
    
