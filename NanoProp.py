#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 10:45:29 2020

@author: jbeckwith
"""

import numpy as np
from RelPerm import Permittivities

class NanoProperties():
    def __init__(self):
        self = self
        return
    
    @staticmethod
    def Yualphaj(s, L, R, a12, a14, V, eps_j, eps_p, eps_m):
        Aj = np.add(np.add(np.multiply(a12, np.square(s)), np.multiply(np.divide(np.multiply(np.multiply(4, np.square(np.pi)), \
                np.multiply(1j, V)), np.multiply(3., np.power(L, 3.))), np.power(s, 3.))), np.multiply(a14, np.power(s, 4.)))
        alpha_j = np.multiply(np.multiply(np.divide(eps_m, np.multiply(4., np.pi)), V),\
                  np.reciprocal(np.subtract(np.subtract(np.reciprocal(np.subtract(np.divide(eps_p, eps_m), 1)),\
                  np.reciprocal(np.subtract(eps_j, 1))), Aj)))
        return alpha_j
    
    def Yualpha3(self, s, L, R, a12_long, a14_long, V, eps_long, eps_p, eps_m):
        alpha3 = self.Yualphaj(s, L, R, a12_long, a14_long, V, eps_long, eps_p, eps_m)
        return alpha3
    
    def Yualpha1(self, s, L, R, a12_trans1, a14_trans1, a12_trans2, a14_trans2, a12_trans3, a14_trans3,\
                 Vtrans1, Vtrans2, Vtrans3, eps_trans1, eps_trans2, eps_trans3, eps_p, eps_m, AHFactor):
        alpha1_1 = self.Yualphaj(s, L, R, a12_trans1, a14_trans1, Vtrans1, eps_trans1, eps_p, eps_m)
        alpha1_2 = self.Yualphaj(s, L, R, a12_trans2, a14_trans2, Vtrans2, eps_trans2, eps_p, eps_m)
        alpha1_3 = self.Yualphaj(s, L, R, a12_trans3, a14_trans3, Vtrans3, eps_trans3, eps_p, eps_m)
        alpha1 = np.multiply(AHFactor, np.sum([alpha1_1, alpha1_2, alpha1_3], axis=0))
        return alpha1
    
    def alphaellipsoid(self, element, wavelength, xi, T, cv, AHFactor, model='Yu'):
        e = np.sqrt(np.subtract(1., np.power(np.add(1., xi), -2))) # eccentricity of the elipse
        L3 = np.multiply(np.divide(np.subtract(1., np.square(e)), np.square(e)), np.subtract(np.multiply(np.divide(1, np.multiply(2., e)), np.log(np.divide(np.add(1., e), np.subtract(1., e)))), 1.))
        L12 = np.divide(np.subtract(1., L3), 2.)
        epsgen = Permittivities(element)
        eps_m = epsgen.eps_m(wavelength, T, cv)
        if model == 'LDR' or element == 'Al':
            eps_p = epsgen.lorentz_drude_rakic_dielectric(wavelength) 
        else:
            eps_p = epsgen.yu_dielectric(wavelength)
        a3_E = np.divide(np.subtract(eps_p, eps_m), np.add(eps_m, np.multiply(L3, np.subtract(eps_p, eps_m))))
        a2_E = AHFactor*np.divide(np.subtract(eps_p, eps_m), np.add(eps_m, np.multiply(L12, np.subtract(eps_p, eps_m))))
        a1_E = AHFactor*np.divide(np.subtract(eps_p, eps_m), np.add(eps_m, np.multiply(L12, np.subtract(eps_p, eps_m))))
        return a3_E, a2_E, a1_E
    
    def depolarisationFactors(self, xi):
        e = np.sqrt(np.subtract(1., np.power(np.add(1., xi), -2))) # eccentricity of the elipse
        L3 = np.multiply(np.divide(np.subtract(1., np.square(e)), np.square(e)), np.subtract(np.multiply(np.divide(1, np.multiply(2., e)), np.log(np.divide(np.add(1., e), np.subtract(1., e)))), 1.))
        L12 = np.divide(np.subtract(1., L3), 2.)
        return L3, L12

    def alpharod(self, element, wavelength, T, cv, model='Yu'):
        #e = np.sqrt(np.subtract(1, np.power(np.add(1, xi), -2))) # eccentricity of the elipse
        epsgen = Permittivities(element)
        eps_m = epsgen.eps_m(wavelength, T, cv)
        if model == 'LDR' or element == 'Al':
            eps_p = epsgen.lorentz_drude_rakic_dielectric(wavelength) 
        else:
            eps_p = epsgen.yu_dielectric(wavelength)
        a3R_R = np.divide(np.subtract(eps_p, eps_m), eps_m)
        return a3R_R

    def alphasphere(self, element, wavelength, T, cv, model='Yu'):
        epsgen = Permittivities(element)
        eps_m = epsgen.eps_m(wavelength, T, cv)
        if model == 'LDR' or element == 'Al':
            eps_p = epsgen.lorentz_drude_rakic_dielectric(wavelength) 
        else:
            eps_p = epsgen.yu_dielectric(wavelength)
        alpha_s = np.divide(np.subtract(eps_p, eps_m), np.divide(np.add(eps_p, np.multiply(eps_m, 2)), 3))
        return alpha_s
    
    def Yua3a1(self, wavelength, T, cv, L, R, element, AHFactor, shape, model='Yu'):
        # Outputs alpha3, alpha1 of particle according to model parameterised in
        # Yu, R.; Liz-Marzán, L. M.; Abajo, F. J. G. de.
        # Universal Analytical Modeling of Plasmonic Nanoparticles.
        # Chem. Soc. Rev. 2017, 46 (22), 6710–6724.
        # ------ INPUTS ------ #
        # wavelength (in nm)
        # T temperature (in K)
        # cv volume fraction of glycerol
        # L length of particle
        # R aspect ratio of particle
        # element - element from 'Au', 'Ag', 'Cu', 'Al'
        # AHFactor - ad-hoc factor that increases intensity of transverse modes
        # shape - shape (string) - one of 'Rod', 'Triangle', 'Cage', 'Ellipsoid', 'Bicone', 'Disk', 'Ring' 'Bipyramid', 'Squared Rod', 'Cylinder'
        # ------ OUTPUTS ------ #
        # alpha3 alpha_3 tensor element
        # alpha1 alpha_1 tensor element
        L = float(L)
        R = float(R)
        epsgen = Permittivities(element)
        eps_m = epsgen.eps_m(wavelength, T, cv)
        if model == 'LDR' or element == 'Al':
            eps_p = epsgen.lorentz_drude_rakic_dielectric(wavelength) 
        else:
            eps_p = epsgen.yu_dielectric(wavelength)
            
        eps_long, V1, a12_long, a14_long, Vparticle, \
                eps_trans1, eps_trans2, eps_trans3, Vtrans1, \
                Vtrans2, Vtrans3, a12_trans1, a12_trans2, a12_trans3, \
                a14_trans1, a14_trans2, a14_trans3 = epsgen.geometry_factors(shape, R)
                
        s = np.divide(np.multiply(np.sqrt(eps_m), L), wavelength)
        alpha3 = self.Yualpha3(s, L, R, a12_long, a14_long, np.multiply(Vparticle(R, L), V1), eps_long, eps_p, eps_m)
        if shape == 'Rod':
            alpha1 = self.Yualpha1(s, L, R, a12_trans1, a14_trans1, a12_trans2, a14_trans2, a12_trans3, a14_trans3,\
                 np.multiply(Vparticle(R, L), Vtrans1), np.multiply(Vparticle(R, L), Vtrans2),\
                 np.multiply(Vparticle(R, L), Vtrans3), eps_trans1, eps_trans2, eps_trans3, eps_p, eps_m, AHFactor)
        return alpha3, alpha1
    
    def YuSpectra(self, wavelength, T, cv, L, R, element, AHFactor, shape, model='Yu'):
        # Outputs spectra of particle according to model parameterised in
        # Yu, R.; Liz-Marzán, L. M.; Abajo, F. J. G. de.
        # Universal Analytical Modeling of Plasmonic Nanoparticles.
        # Chem. Soc. Rev. 2017, 46 (22), 6710–6724.
        # ------ INPUTS ------ #
        # wavelength (in nm)
        # T temperature (in K)
        # cv volume fraction of glycerol
        # L length of particle
        # R aspect ratio of particle
        # element - element from 'Au', 'Ag', 'Cu', 'Al'
        # AHFactor - ad-hoc factor that increases intensity of transverse modes
        # shape - shape (string) - one of 'Rod', 'Triangle', 'Cage', 'Ellipsoid', 'Bicone', 'Disk', 'Ring' 'Bipyramid', 'Squared Rod', 'Cylinder'
        # ------ OUTPUTS ------ #
        # alpha3 alpha_3 tensor element
        # alpha1 alpha_1 tensor element
        L = float(L)
        R = float(R)
        epsgen = Permittivities(element)
        eps_m = epsgen.eps_m(wavelength, T, cv)
        alpha3, alpha1 = self.Yua3a1(wavelength, T, cv, L, R, element, AHFactor, shape, model)
        Vparticle = epsgen.geometry_factors(shape, R)[4]
        extinctionprefactor = np.divide(np.multiply(8., np.square(np.pi)), np.multiply(np.sqrt(eps_m), wavelength))
        scatteringprefactor = np.divide(np.multiply(128., np.power(np.pi, 5.)), np.multiply(3., np.power(wavelength, 4.)))
        
        totalplasmons = np.sum([alpha3, alpha1], 0)
        totaltransverse = np.multiply(alpha1, 1.)
        
        exttransvers = np.divide(np.multiply(extinctionprefactor, np.imag(totaltransverse)), Vparticle(R, L))
        extlongitudinal = np.divide(np.multiply(extinctionprefactor, np.imag(alpha3)), Vparticle(R, L))
        exttotal = np.divide(np.multiply(extinctionprefactor, np.imag(totalplasmons)), Vparticle(R, L))
        
        scattertransvers = np.divide(np.multiply(scatteringprefactor, np.square(np.abs(totaltransverse))), Vparticle(R, L))
        scatterlongitudinal = np.divide(np.multiply(scatteringprefactor, np.square(np.abs(alpha3))), Vparticle(R, L))
        scattertotal = np.divide(np.multiply(scatteringprefactor, np.square(np.abs(totalplasmons))), Vparticle(R, L))
    
        # Absorption cross sections are sigma_(ext) - sigma_(sca)
        absorptiontransvers = np.subtract(exttransvers, scattertransvers)
        absorptiontotal = np.subtract(exttotal, scattertotal)
        absorptionlongitudinal = np.subtract(extlongitudinal, scatterlongitudinal)
        return exttotal, extlongitudinal, exttransvers, scattertotal, scatterlongitudinal, scattertransvers\
                , absorptiontotal, absorptionlongitudinal, absorptiontransvers
                
    def Yua11a13a33(self, wavelength, T, cv, L, R, element, AHFactor, shape, IW, model='Yu'):
        # Outputs a11, a13, a33 according to model parameterised in
        # Yu, R.; Liz-Marzán, L. M.; Abajo, F. J. G. de.
        # Universal Analytical Modeling of Plasmonic Nanoparticles.
        # Chem. Soc. Rev. 2017, 46 (22), 6710–6724.
        # ------ INPUTS ------ #
        # wavelength (in nm)
        # T temperature (in K)
        # cv volume fraction of glycerol
        # L length of particle
        # R aspect ratio of particle
        # element - element from 'Au', 'Ag', 'Cu', 'Al'
        # AHFactor - ad-hoc factor that increases intensity of transverse modes
        # shape - shape (string) - one of 'Rod', 'Triangle', 'Cage', 'Ellipsoid', 'Bicone', 'Disk', 'Ring' 'Bipyramid', 'Squared Rod', 'Cylinder'
        # IW intensity weights at wavelengths
        # ------ OUTPUTS ------ #
        # a11 a11 tensor element * IW
        # a13 a13 tensor element * IW
        # a33 a33 tensor element * IW
        L = float(L)
        R = float(R)
        alpha3, alpha1 = self.Yua3a1(wavelength, T, cv, L, R, element, AHFactor, shape, model)
        
        a11 = np.multiply(IW, np.square(np.abs(alpha1)))
        a13 = np.real(np.multiply(IW, np.add(np.multiply(alpha1, np.conj(alpha3)), np.multiply(np.conj(alpha1), alpha3))))
        a33 = np.multiply(IW, np.square(np.abs(alpha3)))
        return a11, a13, a33
