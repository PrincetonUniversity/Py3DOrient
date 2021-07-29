#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 10:08:11 2021

@author: jbeckwith
"""

import numpy as np
from scipy.interpolate import interp1d
from copy import deepcopy

class CRFs:
    # Cramer Rao lower bound functions for using fluorescence and/or scattering
    # to observe orientation of single particle
    def __init__(self):
        self = self
        
    @staticmethod
    def InvertMatrix(Matrix):
        try:
            IM = np.linalg.inv(Matrix)
        except:
            IM = np.zeros(Matrix.shape)+np.Inf
        return IM
    
    @staticmethod
    def DefineLimits(CRLBT, CRLBP):
        CRLBT[np.isnan(CRLBT)] = np.pi/2
        CRLBP[np.isnan(CRLBP)] = np.pi
        CRLBT[CRLBT > np.pi/2] = np.pi/2
        CRLBP[CRLBP > np.pi] = np.pi
        return CRLBT, CRLBP
    
    @staticmethod
    def Itest(I, A, n_detectors=4):
        if not hasattr(I, "__len__"):
            if hasattr(A, "__len__"):
                Ir = np.full([n_detectors, len(A)], np.divide(I, len(A)))
            else:
                Ir = np.full([n_detectors, 1], I)
        return Ir
    
    @staticmethod
    def Betatest(Beta, A, n_detectors=4):
        if not hasattr(Beta, "__len__"):
            if hasattr(A, "__len__"):
                Beta = np.full([n_detectors, len(A)], Beta)
            else:
                Beta = np.full([n_detectors, 1], Beta)
        return Beta
    
    @staticmethod
    def Irecast(I, n_detectors):
        I1 = I[0, :]
        I2 = I[1, :]
        I3 = I[2, :]
        if n_detectors == 4:
            I4 = I[3, :]
            return I1, I2, I3, I4
        else:
            return I1, I2, I3
    
    @staticmethod
    def Betarecast(Beta, n_detectors):
        Beta1 = Beta[0, :]
        Beta2 = Beta[1, :]
        Beta3 = Beta[2, :]
        if n_detectors == 4:
            Beta4 = Beta[3, :]
            return Beta1, Beta2, Beta3, Beta4
        else:
            return Beta1, Beta2, Beta3

    @staticmethod
    def cosinedistance(theta1, phi1):
        # Calculates Cosine Error relative to 0, 0
        # ------ INPUTS ------ #
        # theta1 theta value (in radians) of vector
        # phi1 phi value (in radians) of vector
        # ------ OUTPUT ------ #
        # cdist cosine distance between (0,0) and input vector (in radians)
        sinterm1 = np.sin(theta1, np.sin(0))
        costerm1 = np.cos(phi1)
        costerm2 = np.multiply(np.cos(theta1), np.cos(0))
        cdist = np.arccos(np.add(np.multiply(sinterm1, costerm1), costerm2))
        return cdist

    @staticmethod    
    def n_m(wavelength, T, cv):
        # Calculates refractive index of glycerol-water mixture
        # ------ INPUTS ------ #
        # wavelength is wavelength in nm
        # T is temperature in Kelvin
        # cv is glycerol volume fraction
        # ------ OUTPUT ------ #
        # n0 is refractive index
        from SolventFunctions import SolventFunc
        rhow = SolventFunc.waterdensity(T-273.15)
        n0w= SolventFunc.waterrefractiveindex(wavelength, T, rhow)
        n0g = SolventFunc.glycerolrefractiveindex(wavelength)
        if type(n0g) == np.array:
            n0g[np.isnan(n0g)] = 0
        n0 = np.sqrt(np.add(np.multiply(cv, np.square(n0g)), np.multiply(np.subtract(1, cv), np.square(n0w))))
        return n0

    @staticmethod    
    def InstrResp(NACond, NAObj, n0):
        # Calculates Instrument factors from Fourkas and Yang
        # Fourkas, J. T. Opt. Lett. 2001, 26 (4), 211–213. (equation 5)
        # Yang, H. J. Phys. Chem. A 2007, 111 (23), 4987–4997. (equation 10)
        # ------ INPUTS ------ #
        # NACond is numerical aperture of condenser
        # NAObj is numerical aperture of objective
        # n0 is refractive index (wavelength dependent)
        # ------ OUTPUTS ------ #
        # Theta is condenser excitation angle
        # Delta is objective collection angle
        # A is A factor
        # B is B factor
        # C is C factor
        # H is H factor
        Theta = np.arcsin(np.divide(NACond, n0))
        Delta = np.arcsin(np.divide(NAObj, n0))
        cosdelta = np.cos(Delta)
        cosdeltasqr = np.square(cosdelta)
        cosdeltacubed = np.power(cosdelta, 3)
        A = np.add(np.subtract(np.divide(1, 6), np.multiply(0.25, cosdelta)), np.multiply(np.divide(1, 12), cosdeltacubed))
        B = np.subtract(np.multiply(0.125, cosdelta), np.multiply(0.125, cosdeltacubed))
        C = np.subtract(np.subtract(np.subtract(np.divide(7, 48), np.multiply(0.0625, cosdelta)), np.multiply(0.0625, cosdeltasqr)), np.multiply(np.divide(1, 48), cosdeltacubed))
        H = np.cos(np.multiply(2, Theta))
        return Theta, Delta, A, B, C, H
    
    def Jthetathetaf(self, theta, phi, time, Beta, I, A, B, C, n_detectors=4):
        # Calculates Fisher Information Matrix element theta, theta for
        # fluorescence case
        # ------ INPUTS ------ #
        # theta is angle in radians
        # phi is angle in radians
        # time is time taken to observe photons
        # Beta is SBR = (Intensity + Background) / Background, should be n_detector x wavelength matrix OR scalar
        # I is Intensity/(1-Beta^{-1}) should be n_detector x wavelength matrix OR scalar
        # A, B, C are instrument factors (wavelength dependent)
        # n_detectors is how many detectors used
        # ------ OUTPUT ------ #
        # Jval Fisher Information Matrix element theta, theta
        I = self.Itest(I, A, n_detectors)
        Beta = self.Betatest(Beta, A, n_detectors)
        if n_detectors == 3:
            I1, I2, I3 = self.Irecast(I, n_detectors)
            Beta1, Beta2, Beta3 = self.Betarecast(Beta, n_detectors)
        else:
            I1, I2, I3, I4 = self.Irecast(I, n_detectors)
            Beta1, Beta2, Beta3, Beta4 = self.Betarecast(Beta, n_detectors)
            
        Asqr = np.square(A); Csqr = np.square(C); AC = np.multiply(Asqr, Csqr)
        costheta2 = np.square(np.cos(theta))
        sintheta2 = np.square(np.sin(theta))
        cos2phi = np.cos(np.multiply(2., phi))
        sin2phi = np.sin(np.multiply(2., phi))
        sin2theta = np.sin(np.multiply(2., theta))
        
        term12num = lambda I, Beta: np.multiply(np.multiply(np.multiply(np.multiply(\
                np.multiply(AC, np.square(np.subtract(Beta, 1))), I) , costheta2)\
                , np.square(cos2phi)), sintheta2)
        term12denom = lambda I, Beta, sgn: np.multiply(np.multiply(Beta, np.power(np.add(A, np.multiply(B, sintheta2)), 3)), \
                np.add(np.multiply(A, np.add(3, Beta)), np.multiply(np.add(np.multiply(B, np.add(3, Beta)), \
                np.multiply(sgn, np.multiply(np.multiply(C, np.subtract(Beta, 1)), cos2phi))), sintheta2)))    
    
        term12 = lambda I, Beta, sgn: np.divide(term12num(I, Beta), term12denom(I, Beta, sgn))
        
        term34num = lambda I, Beta: np.multiply(np.multiply(np.multiply(\
                np.multiply(AC, np.square(np.subtract(Beta, 1))), I) , np.square(sin2phi))\
                , np.square(sin2theta))
        
        term34denom = lambda I, Beta, sgn: np.multiply(np.multiply(np.multiply(4, Beta),\
                np.power(np.add(A, np.multiply(B, sintheta2)), 3)), \
                np.add(np.multiply(A, np.add(3, Beta)), np.multiply(np.add(np.multiply(B, np.add(3, Beta)), \
                np.multiply(sgn, np.multiply(np.multiply(C, np.subtract(Beta, 1)), sin2phi))), sintheta2)))    

        term34 = lambda I, Beta, sgn: np.divide(term34num(I, Beta), term34denom(I, Beta, sgn))
    
        Term1 = term12(I1, Beta1, -1)
        Term2 = term12(I2, Beta2, 1)
        Term3 = term34(I3, Beta3, -1)
        if n_detectors == 4:
            Term4 = term34(I4, Beta4, 1)
            Jval = np.sum(np.multiply(time, [Term1, Term2, Term3, Term4]))
        else:
            Jval = np.sum(np.multiply(time, [Term1, Term2, Term3]))
        return Jval
    
    def JthetathetafNB(self, theta, phi, time, I, A, B, C, n_detectors=4):
        # Calculates Fisher Information Matrix element theta, theta for
        # fluorescence case (no background)
        # ------ INPUTS ------ #
        # theta is angle in radians
        # phi is angle in radians
        # time is time taken to observe photons
        # I is Intensity should be n_detector x wavelength matrix OR scalar
        # A, B, C are instrument factors (wavelength dependent)
        # n_detectors is how many detectors used
        # ------ OUTPUT ------ #
        # Jval Fisher Information Matrix element theta, theta
        I = self.Itest(I, A, n_detectors)
        if n_detectors == 3:
            I1, I2, I3 = self.Irecast(I, n_detectors)
        else:
            I1, I2, I3, I4 = self.Irecast(I, n_detectors)
            
        Asqr = np.square(A); Csqr = np.square(C); AC = np.multiply(Asqr, Csqr)
        costheta2 = np.square(np.cos(theta))
        sintheta2 = np.square(np.sin(theta))
        cos2phi = np.cos(np.multiply(2., phi))
        sin2phi = np.sin(np.multiply(2., phi))
        sin2theta = np.sin(np.multiply(2., theta))
        asin2 = np.square(np.reciprocal(np.sin(theta)))
        
        prefac = np.multiply(np.reciprocal(np.multiply(4., np.power(np.add(A, np.multiply(B, sintheta2)), 3.))), AC)
        prefac13 = np.multiply(np.multiply(4, costheta2), np.square(cos2phi))
        prefac24 = np.multiply(np.square(sin2theta), np.square(sin2phi))
        
        term13 = lambda I, sgn: np.divide(I, np.add(np.add(B,\
                 np.multiply(np.multiply(sgn, C), cos2phi)), np.multiply(A, asin2)))
        term24 = lambda I, sgn: np.divide(I, np.add(A, \
                 np.multiply(sintheta2, np.add(B, np.multiply(np.multiply(sgn, C), sin2phi)))))
        
        if n_detectors == 3:
            Jval = np.multiply(time, np.multiply(prefac, \
            np.add(np.multiply(prefac13, np.add(term13(I1, -1), term13(I3, 1))), \
            np.multiply(prefac24, term24(I2, 1)))))
        else:
            Jval = np.multiply(time, np.multiply(prefac, \
            np.add(np.multiply(prefac13, np.add(term13(I1, -1), term13(I3, 1))), \
            np.multiply(prefac24, np.add(term24(I2, 1), term24(I4, -1.))))))
        return Jval
    
    def Jthetaphif(self, theta, phi, time, Beta, I, A, B, C, n_detectors=4):
        # Calculates Fisher Information Matrix element theta, phi for
        # fluorescence case
        # ------ INPUTS ------ #
        # theta is angle in radians
        # phi is angle in radians
        # time is time taken to observe photons
        # Beta is SBR = (Intensity + Background) / Background, should be n_detector x wavelength matrix OR scalar
        # I is Intensity/(1-Beta^{-1}) should be n_detector x wavelength matrix OR scalar
        # A, B, C are instrument factors (wavelength dependent)
        # n_detectors is how many detectors used
        # ------ OUTPUT ------ #
        # Jval Fisher Information Matrix element theta, phi
        I = self.Itest(I, A, n_detectors)
        Beta = self.Betatest(Beta, A, n_detectors)
        if n_detectors == 3:
            I1, I2, I3 = self.Irecast(I, n_detectors)
            Beta1, Beta2, Beta3 = self.Betarecast(Beta, n_detectors)
        else:
            I1, I2, I3, I4 = self.Irecast(I, n_detectors)
            Beta1, Beta2, Beta3, Beta4 = self.Betarecast(Beta, n_detectors)
            
        Csqr = np.square(C); AC = np.multiply(A, Csqr)
        cos2phi = np.cos(np.multiply(2., phi))
        sin2phi = np.sin(np.multiply(2., phi))
        sin4phi = np.sin(np.multiply(4., phi))
        csc2 = np.square(np.reciprocal(np.sin(theta)))
        cot = np.reciprocal(np.tan(theta))

        term1234num = lambda I, Beta, sgn: np.multiply(sgn, np.multiply(np.multiply(np.multiply(np.multiply(\
                np.multiply(AC, np.square(np.subtract(Beta, 1))), I) , cot)\
                , csc2), sin4phi))
                
        term12denom = lambda I, Beta, sgn: np.multiply(np.multiply(2., Beta),\
                    np.multiply(np.square(np.add(B, np.multiply(A, csc2))),\
                    np.add(np.multiply(np.multiply(sgn, C), np.multiply(np.subtract(Beta, 1), cos2phi)),\
                    np.multiply(np.add(3, Beta), np.add(B, np.multiply(A, csc2)))))) 
    
        term12 = lambda I, Beta, sgn1, sgn2: np.divide(term1234num(I, Beta, sgn1), term12denom(I, Beta, sgn2))
                
        term34denom = lambda I, Beta, sgn: np.multiply(np.multiply(2., Beta),\
                    np.multiply(np.square(np.add(B, np.multiply(A, csc2))),\
                    np.add(np.multiply(np.multiply(sgn, C), np.multiply(np.subtract(Beta, 1), sin2phi)),\
                    np.multiply(np.add(3, Beta), np.add(B, np.multiply(A, csc2)))))) 

        term34 = lambda I, Beta, sgn1, sgn2: np.divide(term1234num(I, Beta, sgn1), term34denom(I, Beta, sgn2))
    
        Term1 = term12(I1, Beta1, -1, -1)
        Term2 = term12(I2, Beta2, -1, 1)
        Term3 = term34(I3, Beta3, 1, -1)
        if n_detectors == 4:
            Term4 = term34(I4, Beta4, 1, 1)
            Jval = np.sum(np.multiply(time, [Term1, Term2, Term3, Term4]))
        else:
            Jval = np.sum(np.multiply(time, [Term1, Term2, Term3]))
        return Jval
    
    def JthetaphifNB(self, theta, phi, time, I, A, B, C, n_detectors=4):
        # Calculates Fisher Information Matrix element theta, phi for
        # fluorescence case (no background)
        # ------ INPUTS ------ #
        # theta is angle in radians
        # phi is angle in radians
        # time is time taken to observe photons
        # I is Intensity/(1-Beta^{-1}) should be n_detector x wavelength matrix OR scalar
        # A, B, C are instrument factors (wavelength dependent)
        # n_detectors is how many detectors used
        # ------ OUTPUT ------ #
        # Jval Fisher Information Matrix element theta, phi
        I = self.Itest(I, A, n_detectors)
        if n_detectors == 3:
            I1, I2, I3 = self.Irecast(I, n_detectors)
        else:
            I1, I2, I3, I4 = self.Irecast(I, n_detectors)
            
        Csqr = np.square(C); AC = np.multiply(A, Csqr)
        cos2phi = np.cos(np.multiply(2., phi))
        sin2phi = np.sin(np.multiply(2., phi))
        sin4phi = np.sin(np.multiply(4., phi))
        csc2 = np.square(np.reciprocal(np.sin(theta)))
        cot = np.reciprocal(np.tan(theta))
        
        prefactor = np.multiply(np.reciprocal(np.multiply(2., np.square(np.add(B, np.multiply(A, csc2))))),\
                    np.multiply(np.multiply(np.multiply(AC, cot), csc2), sin4phi))

        term13 = lambda I, sgn1, sgn2: np.divide(np.multiply(sgn1, I),\
                 np.add(np.add(B, np.multiply(np.multiply(sgn2, C), cos2phi)), np.multiply(A, csc2)))
        
        term24 = lambda I, sgn1, sgn2: np.divide(np.multiply(sgn1, I),\
                 np.add(np.add(B, np.multiply(np.multiply(sgn2, C), sin2phi)), np.multiply(A, csc2)))
                
        
        Term1 = term13(I1, -1, 1)
        Term2 = term24(I2, 1, 1)
        Term3 = term13(I3, -1, -1)
        if n_detectors == 4:
            Term4 = term24(I4, 1, -1)
            Jval = np.sum(np.multiply(time, np.multiply(prefactor, np.sum([Term1, Term2, Term3, Term4]))))
        else:
            Jval = np.sum(np.multiply(time, np.multiply(prefactor, np.sum([Term1, Term2, Term3]))))
        return Jval
   
    def Jphiphif(self, theta, phi, time, Beta, I, A, B, C, n_detectors=4):
        # Calculates Fisher Information Matrix element phi, phi for
        # fluorescence case
        # ------ INPUTS ------ #
        # theta is angle in radians
        # phi is angle in radians
        # time is time taken to observe photons
        # Beta is SBR = (Intensity + Background) / Background, should be n_detector x wavelength matrix OR scalar
        # I is Intensity/(1-Beta^{-1}) should be n_detector x wavelength matrix OR scalar
        # A, B, C are instrument factors (wavelength dependent)
        # n_detectors is how many detectors used
        # ------ OUTPUT ------ #
        # Jval Fisher Information Matrix element phi, phi
        I = self.Itest(I, A, n_detectors)
        Beta = self.Betatest(Beta, A, n_detectors)
        if n_detectors == 3:
            I1, I2, I3 = self.Irecast(I, n_detectors)
            Beta1, Beta2, Beta3 = self.Betarecast(Beta, n_detectors)
        else:
            I1, I2, I3, I4 = self.Irecast(I, n_detectors)
            Beta1, Beta2, Beta3, Beta4 = self.Betarecast(Beta, n_detectors)
            
        Csqr = np.square(C)
        cos2phi = np.cos(np.multiply(2., phi))
        sin2phi = np.sin(np.multiply(2., phi))
        csc2 = np.square(np.reciprocal(np.sin(theta)))

        term12num = lambda I, Beta: np.multiply(np.multiply(np.multiply(Csqr, np.square(np.subtract(Beta, 1))),\
                                    I), np.square(sin2phi))
                
        term12denom = lambda I, Beta, sgn: np.multiply(np.multiply(Beta, np.add(B, np.multiply(A, csc2))),\
                      np.add(np.multiply(np.multiply(np.multiply(sgn, C), np.subtract(Beta, 1.)), cos2phi),\
                      np.multiply(np.add(3., Beta), np.add(B, np.multiply(A, csc2)))))
    
        term12 = lambda I, Beta, sgn: np.divide(term12num(I, Beta), term12denom(I, Beta, sgn))
        
        term34num = lambda I, Beta: np.multiply(np.multiply(np.multiply(Csqr, np.square(np.subtract(Beta, 1))),\
                                    I), np.square(cos2phi))
        
        term34denom = lambda I, Beta, sgn: np.multiply(np.multiply(Beta, np.add(B, np.multiply(A, csc2))),\
                      np.add(np.multiply(np.multiply(np.multiply(sgn, C), np.subtract(Beta, 1.)), sin2phi),\
                      np.multiply(np.add(3., Beta), np.add(B, np.multiply(A, csc2)))))

        term34 = lambda I, Beta, sgn: np.divide(term34num(I, Beta), term34denom(I, Beta, sgn))
    
        Term1 = term12(I1, Beta1, -1,)
        Term2 = term12(I2, Beta2, 1)
        Term3 = term34(I3, Beta3, -1)
        if n_detectors == 4:
            Term4 = term34(I4, Beta4, 1)
            Jval = np.sum(np.multiply(time, [Term1, Term2, Term3, Term4]))
        else:
            Jval = np.sum(np.multiply(time, [Term1, Term2, Term3]))
        return Jval
    
    def JphiphifNB(self, theta, phi, time, I, A, B, C, n_detectors=4):
        # Calculates Fisher Information Matrix element phi, phi for
        # fluorescence case (no background)
        # ------ INPUTS ------ #
        # theta is angle in radians
        # phi is angle in radians
        # time is time taken to observe photons
        # I is Intensity/(1-Beta^{-1}) should be n_detector x wavelength matrix OR scalar
        # A, B, C are instrument factors (wavelength dependent)
        # n_detectors is how many detectors used
        # ------ OUTPUT ------ #
        # Jval Fisher Information Matrix element phi, phi
        I = self.Itest(I, A, n_detectors)
        if n_detectors == 3:
            I1, I2, I3 = self.Irecast(I, n_detectors)
        else:
            I1, I2, I3, I4 = self.Irecast(I, n_detectors)
            
        Csqr = np.square(C)
        cos2phi = np.cos(np.multiply(2., phi))
        sin2phi = np.sin(np.multiply(2., phi))
        csc2 = np.square(np.reciprocal(np.sin(theta)))
        
        prefactor = np.multiply(np.reciprocal(np.add(B, np.multiply(A, csc2))), Csqr)
        

        term13 = lambda I, sgn: np.multiply(np.square(sin2phi), np.divide(I,\
                 np.add(np.add(B, np.multiply(np.multiply(sgn, C), cos2phi)), np.multiply(A, csc2))))
        
        term24 = lambda I, sgn: np.multiply(np.square(cos2phi), np.multiply(np.square(cos2phi), np.divide(I,\
                 np.add(np.add(B, np.multiply(np.multiply(sgn, C), sin2phi)), np.multiply(A, csc2)))))
                
        
        Term1 = term13(I1, 1)
        Term2 = term24(I2, 1)
        Term3 = term13(I3, -1)
        if n_detectors == 4:
            Term4 = term24(I4, -1)
            Jval = np.sum(np.multiply(time, np.multiply(prefactor, np.sum([Term1, Term2, Term3, Term4]))))
        else:
            Jval = np.sum(np.multiply(time, np.multiply(prefactor, np.sum([Term1, Term2, Term3]))))
        return Jval

    @staticmethod        
    def NormFactorScatter(theta, a11, a13, a33, A, B, H):
        # Calculates normalisation factor for
        # scattering equations
        # ------ INPUTS ------ #
        # theta is angle in radians
        # a11, a13 and a33 are polarizability tensor elements * illumination
        # A, B, C, H are instrument factors (wavelength dependent)
        # ------ OUTPUT ------ #
        # NF normalisation factor for angle theta
        if hasattr(theta, "__len__"):
            if hasattr(A, "__len__"):
                theta = np.tile(theta, (len(A), 1)).T
        NF = (2*(A + B)*(3. + H) - (A + 3*A*H + 4*B*(1 + H))*(np.sin(theta)**2) + B*(1 + 3*H)\
                *(np.sin(theta)**4))*a11 + B*(1 + 3*H)*(np.cos(theta)**2)*(np.sin(theta)**2)*a13 \
                - 0.25*(2*A + B - B*np.cos(2*theta))*(-5 + H + (1 + 3*H)*np.cos(2*theta))*a33
        return NF
        
    def thetadepgetscatter(self, a11, a13, a33, A, B, H):
        # Calculates theta dependence factor for
        # scattering equations
        # ------ INPUTS ------ #
        # a11, a13 and a33 are polarizability tensor elements * illumination
        # A, B, H are instrument factors (wavelength dependent)
        # ------ OUTPUT ------ #
        # thetadepf is lambda function for theta dependence of intensity
        x = np.linspace(0, np.pi/2, 10000)
        thetadepmean = np.mean(self.NormFactorScatter(x, a11, a13, a33, A, B, H), 1)
        sinx = np.sin(x)
        m0 = np.trapz(y=np.multiply(thetadepmean, sinx), x=sinx)
        m1 = np.trapz(y=np.divide(np.multiply(np.multiply(thetadepmean, sinx), x), m0), x=sinx)
        thetadep = np.divide(thetadepmean, np.mean(self.NormFactorScatter(m1, a11, a13, a33, A, B, H)))
        thetfun = interp1d(x, thetadep, 'cubic', bounds_error=False, fill_value=thetadep[-1])
        thetadepf = lambda theta: thetfun(theta)
        return thetadepf
    
    @staticmethod    
    def NormFactorFluo(theta, A, B):
        # Calculates normalisation factor for
        # fluorescence equations
        # ------ INPUTS ------ #
        # theta is angle in radians
        # A, B are instrument factors (wavelength dependent)
        # ------ OUTPUT ------ #
        # NF normalisation factor for angle theta
        if hasattr(theta, "__len__"):
            if hasattr(A, "__len__"):
                theta = np.tile(theta, (len(A), 1)).T
        NF = np.multiply(4, np.add(A, np.multiply(B, np.square(np.sin(theta)))))
        return NF
    
    def thetadepgetfluo(self, A, B):
        # Calculates theta dependence factor for
        # fluorescecnce equations
        # ------ INPUTS ------ #
        # A, B are instrument factors (wavelength dependent)
        # ------ OUTPUT ------ #
        # thetadepf is lambda function for theta dependence of intensity
        x = np.linspace(0, np.pi/2, 10000)
        thetadepmean = np.mean(self.NormFactorFluo(x, A, B), 1)
        sinx = np.sin(x)
        m0 = np.trapz(y=np.multiply(thetadepmean, sinx), x=sinx)
        m1 = np.trapz(y=np.divide(np.multiply(np.multiply(thetadepmean, sinx), x), m0), x=sinx)
        thetadep = np.divide(thetadepmean, np.mean(self.NormFactorFluo(m1, A, B)))
        thetfun = interp1d(x, thetadep, 'cubic', bounds_error=False, fill_value=thetadep[-1])
        thetadepf = lambda theta: thetfun(theta)
        return thetadepf
    
    @staticmethod    
    def Qthetatheta1(theta, phi, Beta1, I1, A, B, C, H, a11, a13, a33):
        # Calculates Fisher Information Matrix element Q(theta, theta)1 for
        # scattering case
        # ------ INPUTS ------ #
        # theta is angle in radians
        # phi is angle in radians
        # Beta1 is SBR for detector 1 should be wavelength matrix OR scalar
        # I1 is Intensity1/(1-Beta1^{-1}) should be wavelength matrix OR scalar
        # A, B, C, H are instrument factors (wavelength dependent)
        # a11, a13 and a33 are polarizability tensor elements * illumination
        # ------ OUTPUT ------ #
        # Qval Q function for theta, theta 1
        if np.any(Beta1): # if we're taking background into account
            Betafac = (((1./Beta1) - 1)**2) # calculate background factor
        else: # if not
            Betafac = 1 # ignore it
        Qval = (C**2)*(np.cos(2*phi)**2)*((-2*(16*B*((3 + H)**2) + A*(147 + H*(114 + 43*H)) + \
                4*(1 + 3*H)*(4*B*(3 + H) + A*(11 + H))*np.cos(2*theta) + A*((1 + 3*H)**2)*np.cos(4*theta))\
                *np.sin(2*theta)*(a11**2) + 2*A*np.sin(2*theta)*a33*(-(1 + 3*H)*(4*(-5 + H)*np.cos(2*theta) +\
                (1 + 3*H)*(3 + np.cos(4*theta)))*a13 + 2*((-5 + H + (1 + 3*H)*np.cos(2*theta))**2)*a33) + \
                a11*((1 + 3*H)*(5*A*(1 + 3*H)*np.sin(2*theta) + 4*(4*B*(3 + H) + A*(11 + H))*np.sin(4*theta) +\
                A*(1 + 3*H)*np.sin(6*theta))*a13 + 16*(2*(3 + H)*(2*A*(1 + H) + B*(3 + H))*np.sin(2*theta) \
                - (1 + 3*H)*(4*A + B*(3 + H))*np.sin(4*theta))*a33))**2)*Betafac*I1
        return Qval
       
    @staticmethod    
    def Mthetatheta1(theta, phi, Beta1, A, B, C, H, NF, a11, a13, a33):
        # Calculates Fisher Information Matrix element M(theta, theta)1 for
        # scattering case (no background)
        # ------ INPUTS ------ #
        # theta is angle in radians
        # phi is angle in radians
        # Beta1 is SBR for detector 1 should be wavelength matrix OR scalar
        # A, B, C, H are instrument factors (wavelength dependent)
        # a11, a13 and a33 are polarizability tensor elements * illumination
        # ------ OUTPUT ------ #
        # Mval M function for theta, theta 1
        if np.any(Beta1):
            Betafac1 = (1 - (1./Beta1))
            Betafac2 = (1./Beta1)
        else:
            Betafac1 = 1
            Betafac2 = 0
        Mval = 4096*(NF**4)*((1./(16*NF))*(4*(2*(A + B)*(3 + H) - (A + 3*A*H + 4*B*(1 + H) + \
                4*(1 + H)*C*np.cos(2*phi))*(np.sin(theta)**2) + (1 + 3*H)*(B + C*np.cos(2*phi))*(np.sin(theta)**4))*a11\
                + (1 + 3*H)*(B + C*np.cos(2*phi))*(np.sin(2*theta)**2)*a13 + 4*(2 - 2*H + (1 + 3*H)*(np.sin(theta)**2))\
                *(A + (B + C*np.cos(2*phi))*(np.sin(theta)**2))*a33)*Betafac1 + Betafac2)
        return Mval
    
    @staticmethod    
    def Qthetatheta2(theta, phi, Beta2, I2, A, B, C, H, a11, a13, a33):
        # Calculates Fisher Information Matrix element Q(theta, theta)2 for
        # scattering case
        # ------ INPUTS ------ #
        # theta is angle in radians
        # phi is angle in radians
        # Beta2 is SBR for detector 1 should be wavelength matrix OR scalar
        # I2 is Intensity1/(1-Beta1^{-1}) should be wavelength matrix OR scalar
        # A, B, C, H are instrument factors (wavelength dependent)
        # a11, a13 and a33 are polarizability tensor elements * illumination
        # ------ OUTPUT ------ #
        # Qval Q function for theta, theta 1
        if np.any(Beta2): # if we're taking background into account
            Betafac = (((1./Beta2) - 1)**2) # calculate background factor
        else: # if not
            Betafac = 1 # ignore it
        Qval = (C**2)*(np.sin(2*theta)**2)*(np.sin(2*phi)**2)*((-(16*B*((3 + H)**2) + A*(147 + H*(114 + 43*H))\
                + 4*(1 + 3*H)*(4*B*(3 + H) + A*(11 + H))*np.cos(2*theta) + A*((1 + 3*H)**2)*np.cos(4*theta))\
                *(a11**2) + A*a33*(-(1 + 3*H)*(4*(-5 + H)*np.cos(2*theta) + (1 + 3*H)*(3 + np.cos(4*theta)))*a13\
                + 2*((-5 + H + (1 + 3*H)*np.cos(2*theta))**2)*a33) + a11*((1 + 3*H)*(4*(4*B*(3 + H) + A*(11 + H))\
                *np.cos(2*theta) + A*(1 + 3*H)*(3 + np.cos(4*theta)))*a13 + 16*((3 + H)*(2*A*(1 + H) + B*(3 + H))\
                - (1 + 3*H)*(4*A + B*(3 + H))*np.cos(2*theta))*a33))**2)*Betafac*I2
        return Qval
    
    @staticmethod    
    def Mthetatheta2(theta, phi, Beta2, A, B, C, H, NF, a11, a13, a33):
        # Calculates Fisher Information Matrix element M(theta, theta)2 for
        # scattering case (no background)
        # ------ INPUTS ------ #
        # theta is angle in radians
        # phi is angle in radians
        # Beta2 is SBR for detector 1 should be wavelength matrix OR scalar
        # A, B, C, H are instrument factors (wavelength dependent)
        # a11, a13 and a33 are polarizability tensor elements * illumination
        # ------ OUTPUT ------ #
        # Mval M function for theta, theta 2
        if np.any(Beta2):
            Betafac1 = (1 - (1./Beta2))
            Betafac2 = (1./Beta2)
        else:
            Betafac1 = 1
            Betafac2 = 0
        Mval = 1024*(NF**4)*((1./(16*NF))*(4*(2*(A + B)*(3 + H) + (1 + 3*H)*(np.sin(theta)**4)*\
                (B + C*np.sin(2*phi)) - (np.sin(theta)**2)*(A + 3*A*H + 4*B*(1 + H) + 4*(1 + H)*C*np.sin(2*phi)))*a11\
                + (1 + 3*H)*(np.sin(2*theta)**2)*(B + C*np.sin(2*phi))*a13 + 4*(2 - 2*H + (1 + 3*H)*(np.sin(theta)**2))\
                *(A + (np.sin(theta)**2)*(B + C*np.sin(2*phi)))*a33)*Betafac1 + Betafac2)
        return Mval
    
    @staticmethod    
    def Qthetatheta3(theta, phi, Beta3, I3, A, B, C, H, a11, a13, a33):
        # Calculates Fisher Information Matrix element Q(theta, theta)3 for
        # scattering case
        # ------ INPUTS ------ #
        # theta is angle in radians
        # phi is angle in radians
        # Beta3 is SBR for detector 1 should be wavelength matrix OR scalar
        # I3 is Intensity1/(1-Beta1^{-1}) should be wavelength matrix OR scalar
        # A, B, C, H are instrument factors (wavelength dependent)
        # a11, a13 and a33 are polarizability tensor elements * illumination
        # ------ OUTPUT ------ #
        # Qval Q function for theta, theta 3
        if np.any(Beta3): # if we're taking background into account
            Betafac = (((1./Beta3) - 1)**2) # calculate background factor
        else: # if not
            Betafac = 1 # ignore it
        Qval = (C**2)*(np.cos(2*phi)**2)*((2*(16*B*((3 + H)**2) + A*(147 + H*(114 + 43*H)) + \
                4*(1 + 3*H)*(4*B*(3 + H) + A*(11 + H))*np.cos(2*theta) + A*((1 + 3*H)**2)*np.cos(4*theta))\
                *np.sin(2*theta)*(a11**2) + 2*A*np.sin(2*theta)*a33*((1 + 3*H)*(4*(-5 + H)*np.cos(2*theta) +\
                (1 + 3*H)*(3 + np.cos(4*theta)))*a13 - 2*((-5 + H + (1 + 3*H)*np.cos(2*theta))**2)*a33) + \
                a11*(-2*(1 + 3*H)*(4*(4*B*(3 + H) + A*(11 + H))*np.cos(2*theta) + A*(1 + 3*H)\
                *(3 + np.cos(4*theta)))*np.sin(2*theta)*a13 + 16*(-2*(3 + H)*(2*A*(1 + H) + B*(3 + H))\
                *np.sin(2*theta) + (1 + 3*H)*(4*A + B*(3 + H))*np.sin(4*theta))*a33))**2)*Betafac*I3
        return Qval

    @staticmethod    
    def Mthetatheta3(theta, phi, Beta3, A, B, C, H, NF, a11, a13, a33):
        # Calculates Fisher Information Matrix element M(theta, theta)3 for
        # scattering case
        # ------ INPUTS ------ #
        # theta is angle in radians
        # phi is angle in radians
        # Beta3 is SBR for detector 1 should be wavelength matrix OR scalar
        # A, B, C, H are instrument factors (wavelength dependent)
        # a11, a13 and a33 are polarizability tensor elements * illumination
        # ------ OUTPUT ------ #
        # Mval M function for theta, theta 3
        if np.any(Beta3):
            Betafac1 = (1 - (1./Beta3))
            Betafac2 = (1./Beta3)
        else:
            Betafac1 = 1
            Betafac2 = 0
        Mval = 4096*(NF**4)*((1./(16*NF))*(4*(2*(A + B)*(3 + H) - (A + 3*A*H + 4*B*(1 + H) - \
                4*(1 + H)*C*np.cos(2*phi))*(np.sin(theta)**2) + (1 + 3*H)*(B - C*np.cos(2*phi))*(np.sin(theta)**4))*a11\
                + (1 + 3*H)*(B - C*np.cos(2*phi))*(np.sin(2*theta)**2)*a13 + 4*(2 - 2*H + (1 + 3*H)*(np.sin(theta)**2))\
                *(A + (B - C*np.cos(2*phi))*(np.sin(theta)**2))*a33)*Betafac1 + Betafac2)
        return Mval

    @staticmethod    
    def Qthetatheta4(theta, phi, Beta4, I4, A, B, C, H, a11, a13, a33):
        # Calculates Fisher Information Matrix element Q(theta, theta)4 for
        # scattering case
        # ------ INPUTS ------ #
        # theta is angle in radians
        # phi is angle in radians
        # Beta4 is SBR for detector 1 should be wavelength matrix OR scalar
        # I4 is Intensity1/(1-Beta1^{-1}) should be wavelength matrix OR scalar
        # A, B, C, H are instrument factors (wavelength dependent)
        # a11, a13 and a33 are polarizability tensor elements * illumination
        # ------ OUTPUT ------ #
        # Qval Q function for theta, theta 4
        if np.any(Beta4): # if we're taking background into account
            Betafac = (((1./Beta4) - 1)**2) # calculate background factor
        else: # if not
            Betafac = 1 # ignore it
        Qval = (C**2)*(np.sin(2*theta)**2)*(np.sin(2*phi)**2)*(((16*B*((3 + H)**2) + A*(147 + H*(114 + 43*H))\
                + (1 + 3*H)*(4*(4*B*(3 + H) + A*(11 + H))*np.cos(2*theta) + A*(1 + 3*H)*np.cos(4*theta)))\
                *(a11**2) + A*a33*((1 + 3*H)*(4*(-5 + H)*np.cos(2*theta) + (1 + 3*H)*(3 + np.cos(4*theta)))*a13\
                - 2*((-5 + H + (1 + 3*H)*np.cos(2*theta))**2)*a33) + a11*(-(1 + 3*H)*(4*(4*B*(3 + H) + A*(11 + H))\
                *np.cos(2*theta) + A*(1 + 3*H)*(3 + np.cos(4*theta)))*a13 + 16*(-(3 + H)*(2*A*(1 + H) + B*(3 + H))\
                + (1 + 3*H)*(4*A + B*(3 + H))*np.cos(2*theta))*a33))**2)*Betafac*I4
        return Qval

    @staticmethod    
    def Mthetatheta4(theta, phi, Beta4, A, B, C, H, NF, a11, a13, a33):
        # Calculates Fisher Information Matrix element M(theta, theta)4 for
        # scattering case
        # ------ INPUTS ------ #
        # theta is angle in radians
        # phi is angle in radians
        # Beta4 is SBR for detector 1 should be wavelength matrix OR scalar
        # A, B, C, H are instrument factors (wavelength dependent)
        # a11, a13 and a33 are polarizability tensor elements * illumination
        # ------ OUTPUT ------ #
        # Mval M function for theta, theta 4
        if np.any(Beta4):
            Betafac1 = (1 - (1./Beta4))
            Betafac2 = (1./Beta4)
        else:
            Betafac1 = 1
            Betafac2 = 0
        Mval = 1024*(NF**4)*((1./(16*NF))*(4*(2*(A + B)*(3 + H) + (1 + 3*H)*(np.sin(theta)**4)*\
                (B - C*np.sin(2*phi)) - (np.sin(theta)**2)*(A + 3*A*H + 4*B*(1 + H) - 4*(1 + H)*C*np.sin(2*phi)))*a11\
                + (1 + 3*H)*(np.sin(2*theta)**2)*(B - C*np.sin(2*phi))*a13 + 4*(2 - 2*H + (1 + 3*H)*(np.sin(theta)**2))\
                *(A + (np.sin(theta)**2)*(B - C*np.sin(2*phi)))*a33)*Betafac1 + Betafac2)
        return Mval

    @staticmethod    
    def Qthetaphi1(theta, phi, Beta1, I1, A, B, C, H, a11, a13, a33):
        # Calculates Fisher Information Matrix element Q(theta, phi)1 for
        # scattering case
        # ------ INPUTS ------ #
        # theta is angle in radians
        # phi is angle in radians
        # Beta1 is SBR for detector 1 should be wavelength matrix OR scalar
        # I1 is Intensity1/(1-Beta1^{-1}) should be wavelength matrix OR scalar
        # A, B, C, H are instrument factors (wavelength dependent)
        # a11, a13 and a33 are polarizability tensor elements * illumination
        # ------ OUTPUT ------ #
        # Qval Q function for theta, phi 1
        if np.any(Beta1): # if we're taking background into account
            Betafac = (((1./Beta1) - 1)**2) # calculate background factor
        else: # if not
            Betafac = 1 # ignore it
        Qval = (C**2)*np.cos(2*phi)*(np.sin(theta)**2)*np.sin(2*phi)*((7 + 5*H + (1 + 3*H)*np.cos(2*theta))*a11\
            - 2*(1 + 3*H)*(np.cos(theta)**2)*a13 + (-5 + H + (1 + 3*H)*np.cos(2*theta))*a33)*\
            (-2*(16*B*((3 + H)**2) + A*(147 + H*(114 + 43*H)) + 4*(1 + 3*H)*(4*B*(3 + H) + \
            A*(11 + H))*np.cos(2*theta) + A*((1 + 3*H)**2)*np.cos(4*theta))*np.sin(2*theta)*(a11**2)\
            + 2*A*np.sin(2*theta)*a33*(-(1 + 3*H)*(4*(-5 + H)*np.cos(2*theta) + (1 + 3*H)\
            *(3 + np.cos(4*theta)))*a13 + 2*((-5 + H + (1 + 3*H)*np.cos(2*theta))**2)*a33) + \
            a11*((1 + 3*H)*(5*A*(1 + 3*H)*np.sin(2*theta) + 4*(4*B*(3 + H) + A*(11 + H))*\
            np.sin(4*theta) + A*(1 + 3*H)*np.sin(6*theta))*a13 + \
            16*(2*(3 + H)*(2*A*(1 + H) + B*(3 + H))*np.sin(2*theta) - (1 + 3*H)*(4*A + B*(3 + H))\
            *np.sin(4*theta))*a33))*Betafac*I1
        return Qval
    
    @staticmethod    
    def Mthetaphi1(theta, phi, Beta1, A, B, C, H, NF, a11, a13, a33):
        # Calculates Fisher Information Matrix element M(theta, phi)1 for
        # scattering case
        # ------ INPUTS ------ #
        # theta is angle in radians
        # phi is angle in radians
        # Beta1 is SBR for detector 1 should be wavelength matrix OR scalar
        # A, B, C, H are instrument factors (wavelength dependent)
        # a11, a13 and a33 are polarizability tensor elements * illumination
        # ------ OUTPUT ------ #
        # Mval M function for theta, phi 1
        if np.any(Beta1):
            Betafac1 = (1 - (1./Beta1))
            Betafac2 = ((16*NF)/Beta1)
        else:
            Betafac1 = 1
            Betafac2 = 0
        Mval = 16*(NF**2)*((4*(2*(A + B)*(3 + H) - (A + 3*A*H + 4*B*(1 + H) + \
                4*(1 + H)*C*np.cos(2*phi))*(np.sin(theta)**2) + (1 + 3*H)*(B + C*np.cos(2*phi))*(np.sin(theta)**4))*a11\
                + (1 + 3*H)*(B + C*np.cos(2*phi))*(np.sin(2*theta)**2)*a13 + \
                4*(2 - 2*H + (1 + 3*H)*(np.sin(theta)**2))*(A + (B + C*np.cos(2*phi))*(np.sin(theta)**2))*a33)\
                *Betafac1 + Betafac2)
        return Mval
    
    @staticmethod    
    def Qthetaphi2(theta, phi, Beta2, I2, A, B, C, H, a11, a13, a33):
        # Calculates Fisher Information Matrix element Q(theta, phi)2 for
        # scattering case
        # ------ INPUTS ------ #
        # theta is angle in radians
        # phi is angle in radians
        # Beta2 is SBR for detector 1 should be wavelength matrix OR scalar
        # I2 is Intensity1/(1-Beta1^{-1}) should be wavelength matrix OR scalar
        # A, B, C, H are instrument factors (wavelength dependent)
        # a11, a13 and a33 are polarizability tensor elements * illumination
        # ------ OUTPUT ------ #
        # Qval Q function for theta, phi 2
        if np.any(Beta2): # if we're taking background into account
            Betafac = (((1./Beta2) - 1)**2) # calculate background factor
        else: # if not
            Betafac = 1 # ignore it
        Qval = (C**2)*np.cos(2*phi)*(np.sin(theta)**2)*np.sin(2*theta)*np.sin(2*phi)*((7 + 5*H + (1 + 3*H)\
            *np.cos(2*theta))*a11 - 2*(1 + 3*H)*(np.cos(theta)**2)*a13 + (-5 + H + (1 + 3*H)*np.cos(2*theta))\
            *a33)*(-(16*B*((3 + H)**2) + A*(147 + H*(114 + 43*H)) + 4*(1 + 3*H)*(4*B*(3 + H)\
            + A*(11 + H))*np.cos(2*theta) + A*((1 + 3*H)**2)*np.cos(4*theta))*(a11**2) + A*a33*\
            (-(1 + 3*H)*(4*(-5 + H)*np.cos(2*theta) + (1 + 3*H)*(3 + np.cos(4*theta)))*a13 + \
            2*((-5 + H + (1 + 3*H)*np.cos(2*theta))**2)*a33) + \
            a11*((1 + 3*H)*(4*(4*B*(3 + H) + A*(11 + H))*np.cos(2*theta) + A*(1 + 3*H)*\
            (3 + np.cos(4*theta)))*a13 + 16*((3 + H)*(2*A*(1 + H) + B*(3 + H)) \
            - (1 + 3*H)*(4*A + B*(3 + H))*np.cos(2*theta))*a33))*Betafac*I2
        return Qval
    
    @staticmethod    
    def Mthetaphi2(theta, phi, Beta2, A, B, C, H, NF, a11, a13, a33):
        # Calculates Fisher Information Matrix element M(theta, phi)2 for
        # scattering case
        # ------ INPUTS ------ #
        # theta is angle in radians
        # phi is angle in radians
        # Beta2 is SBR for detector 1 should be wavelength matrix OR scalar
        # A, B, C, H are instrument factors (wavelength dependent)
        # a11, a13 and a33 are polarizability tensor elements * illumination
        # ------ OUTPUT ------ #
        # Mval M function for theta, phi 2
        if np.any(Beta2):
            Betafac1 = (1 - (1./Beta2))
            Betafac2 = ((16*NF)/Beta2)
        else:
            Betafac1 = 1
            Betafac2 = 0
        Mval = 8*(NF**2)*((4*(2*(A + B)*(3 + H) + (1 + 3*H)*(np.sin(theta)**4)*(B + C*np.sin(2*phi))\
            - (np.sin(theta)**2)*(A + 3*A*H + 4*B*(1 + H) + 4*(1 + H)*C*np.sin(2*phi)))*a11 \
            + (1 + 3*H)*(np.sin(2*theta)**2)*(B + C*np.sin(2*phi))*a13 + 4*(2 - 2*H + (1 + 3*H)\
            *(np.sin(theta)**2))*(A + (np.sin(theta)**2)*(B + C*np.sin(2*phi)))*a33)*Betafac1 + \
            Betafac2)
        return Mval
    
    @staticmethod    
    def Qthetaphi3(theta, phi, Beta3, I3, A, B, C, H, a11, a13, a33):
        # Calculates Fisher Information Matrix element Q(theta, phi)3 for
        # scattering case
        # ------ INPUTS ------ #
        # theta is angle in radians
        # phi is angle in radians
        # Beta3 is SBR for detector 1 should be wavelength matrix OR scalar
        # I3 is Intensity1/(1-Beta1^{-1}) should be wavelength matrix OR scalar
        # A, B, C, H are instrument factors (wavelength dependent)
        # a11, a13 and a33 are polarizability tensor elements * illumination
        # ------ OUTPUT ------ #
        # Qval Q function for theta, phi 3
        if np.any(Beta3): # if we're taking background into account
            Betafac = (((1./Beta3) - 1)**2) # calculate background factor
        else: # if not
            Betafac = 1 # ignore it
        Qval = (C**2)*np.cos(2*phi)*(np.sin(theta)**2)*np.sin(2*phi)*((7 + 5*H + (1 + 3*H)*np.cos(2*theta))*a11\
            - 2*(1 + 3*H)*(np.cos(theta)**2)*a13 + (-5 + H + (1 + 3*H)*np.cos(2*theta))*a33)*\
            (2*(16*B*((3 + H)**2) + A*(147 + H*(114 + 43*H)) + 4*(1 + 3*H)*(4*B*(3 + H) + \
            A*(11 + H))*np.cos(2*theta) + A*((1 + 3*H)**2)*np.cos(4*theta))*np.sin(2*theta)*(a11**2)\
            + 2*A*np.sin(2*theta)*a33*((1 + 3*H)*(4*(-5 + H)*np.cos(2*theta) + (1 + 3*H)\
            *(3 + np.cos(4*theta)))*a13 - 2*((-5 + H + (1 + 3*H)*np.cos(2*theta))**2)*a33) + \
            a11*(-2*(1 + 3*H)*(4*(4*B*(3 + H) + A*(11 + H))*np.cos(2*theta) + A*(1 + 3*H)\
            *(3 + np.cos(4*theta)))*np.sin(2*theta)*a13 + 16*(-2*(3 + H)*(2*A*(1 + H)\
            + B*(3 + H))*np.sin(2*theta) + (1 + 3*H)*(4*A + B*(3 + H))*np.sin(4*theta))*a33))\
            *Betafac*I3
        return Qval
    
    @staticmethod    
    def Mthetaphi3(theta, phi, Beta3, A, B, C, H, NF, a11, a13, a33):
        # Calculates Fisher Information Matrix element M(theta, phi)3 for
        # scattering case
        # ------ INPUTS ------ #
        # theta is angle in radians
        # phi is angle in radians
        # Beta3 is SBR for detector 1 should be wavelength matrix OR scalar
        # A, B, C, H are instrument factors (wavelength dependent)
        # a11, a13 and a33 are polarizability tensor elements * illumination
        # ------ OUTPUT ------ #
        # Mval M function for theta, phi 3
        if np.any(Beta3):
            Betafac1 = (1 - (1./Beta3))
            Betafac2 = ((16*NF)/Beta3)
        else:
            Betafac1 = 1
            Betafac2 = 0
        Mval = 16*(NF**2)*((4*(2*(A + B)*(3 + H) - (A + 3*A*H + 4*B*(1 + H) - \
            4*(1 + H)*C*np.cos(2*phi))*(np.sin(theta)**2) + (1 + 3*H)*(B - C*np.cos(2*phi))*(np.sin(theta)**4))*a11\
            + (1 + 3*H)*(B - C*np.cos(2*phi))*(np.sin(2*theta)**2)*a13 + \
            4*(2 - 2*H + (1 + 3*H)*(np.sin(theta)**2))*(A + (B - C*np.cos(2*phi))*(np.sin(theta)**2))*a33)\
            *Betafac1 + Betafac2)
        return Mval
    
    @staticmethod    
    def Qthetaphi4(theta, phi, Beta4, I4, A, B, C, H, a11, a13, a33):
        # Calculates Fisher Information Matrix element Q(theta, phi)4 for
        # scattering case
        # ------ INPUTS ------ #
        # theta is angle in radians
        # phi is angle in radians
        # Beta4 is SBR for detector 1 should be wavelength matrix OR scalar
        # I4 is Intensity1/(1-Beta1^{-1}) should be wavelength matrix OR scalar
        # A, B, C, H are instrument factors (wavelength dependent)
        # a11, a13 and a33 are polarizability tensor elements * illumination
        # ------ OUTPUT ------ #
        # Qval Q function for theta, phi 4
        if np.any(Beta4): # if we're taking background into account
            Betafac = (((1./Beta4) - 1)**2) # calculate background factor
        else: # if not
            Betafac = 1 # ignore it
        Qval = (C**2)*np.cos(2*phi)*(np.sin(theta)**2)*np.sin(2*theta)*np.sin(2*phi)*((7 + 5*H + (1 + 3*H)\
            *np.cos(2*theta))*a11 - 2*(1 + 3*H)*(np.cos(theta)**2)*a13 + (-5 + H + (1 + 3*H)*np.cos(2*theta))\
            *a33)*((16*B*((3 + H)**2) + A*(147 + H*(114 + 43*H)) + (1 + 3*H)*(4*(4*B*(3 + H)\
            + A*(11 + H))*np.cos(2*theta) + A*(1 + 3*H)*np.cos(4*theta)))*(a11**2) + A*a33*\
            ((1 + 3*H)*(4*(-5 + H)*np.cos(2*theta) + (1 + 3*H)*(3 + np.cos(4*theta)))*a13 - \
            2*((-5 + H + (1 + 3*H)*np.cos(2*theta))**2)*a33) + \
            a11*(-(1 + 3*H)*(4*(4*B*(3 + H) + A*(11 + H))*np.cos(2*theta) + A*(1 + 3*H)*\
            (3 + np.cos(4*theta)))*a13 + 16*(-(3 + H)*(2*A*(1 + H) + B*(3 + H)) \
            + (1 + 3*H)*(4*A + B*(3 + H))*np.cos(2*theta))*a33))*Betafac*I4
        return Qval
    
    @staticmethod    
    def Mthetaphi4(theta, phi, Beta4, A, B, C, H, NF, a11, a13, a33):
        # Calculates Fisher Information Matrix element M(theta, phi)4 for
        # scattering case
        # ------ INPUTS ------ #
        # theta is angle in radians
        # phi is angle in radians
        # Beta4 is SBR for detector 1 should be wavelength matrix OR scalar
        # A, B, C, H are instrument factors (wavelength dependent)
        # a11, a13 and a33 are polarizability tensor elements * illumination
        # ------ OUTPUT ------ #
        # Mval M function for theta, phi 4
        if np.any(Beta4):
            Betafac1 = (1 - (1./Beta4))
            Betafac2 = ((16*NF)/Beta4)
        else:
            Betafac1 = 1
            Betafac2 = 0
        Mval = 8*(NF**2)*((4*(2*(A + B)*(3 + H) + (1 + 3*H)*(np.sin(theta)**4)*(B - C*np.sin(2*phi))\
            - (np.sin(theta)**2)*(A + 3*A*H + 4*B*(1 + H) - 4*(1 + H)*C*np.sin(2*phi)))*a11 \
            + (1 + 3*H)*(np.sin(2*theta)**2)*(B - C*np.sin(2*phi))*a13 + 4*(2 - 2*H + (1 + 3*H)\
            *(np.sin(theta)**2))*(A + (np.sin(theta)**2)*(B - C*np.sin(2*phi)))*a33)*Betafac1 + \
            Betafac2)
        return Mval
    
    @staticmethod    
    def Qphiphi1(theta, phi, Beta1, I1, A, B, C, H, a11, a13, a33):
        # Calculates Fisher Information Matrix element Q(phi, phi)1 for
        # scattering case
        # ------ INPUTS ------ #
        # theta is angle in radians
        # phi is angle in radians
        # Beta1 is SBR for detector 1 should be wavelength matrix OR scalar
        # I1 is Intensity1/(1-Beta1^{-1}) should be wavelength matrix OR scalar
        # A, B, C, H are instrument factors (wavelength dependent)
        # a11, a13 and a33 are polarizability tensor elements * illumination
        # ------ OUTPUT ------ #
        # Qval Q function for phi, phi 1
        if np.any(Beta1): # if we're taking background into account
            Betafac = (((1./Beta1) - 1)**2) # calculate background factor
        else: # if not
            Betafac = 1 # ignore it
        Qval = (C**2)*(np.sin(theta)**4)*(np.sin(2*phi)**2)*(((7 + 5*H + (1 + 3*H)*np.cos(2*theta))*a11\
            - 2*(1 + 3*H)*(np.cos(theta)**2)*a13 + (-5 + H + (1 + 3*H)*np.cos(2*theta))*a33)**2)\
            *Betafac*I1
        return Qval

    @staticmethod    
    def Mphiphi1(theta, phi, Beta1, A, B, C, H, NF, a11, a13, a33):
        # Calculates Fisher Information Matrix element M(phi, phi)1 for
        # scattering case
        # ------ INPUTS ------ #
        # theta is angle in radians
        # phi is angle in radians
        # Beta1 is SBR for detector 1 should be wavelength matrix OR scalar
        # A, B, C, H are instrument factors (wavelength dependent)
        # a11, a13 and a33 are polarizability tensor elements * illumination
        # ------ OUTPUT ------ #
        # Mval M function for phi, phi 1
        if np.any(Beta1):
            Betafac1 = (1 - (1./Beta1))
            Betafac2 = ((16*NF)/Beta1)
        else:
            Betafac1 = 1
            Betafac2 = 0
        Mval = NF*((4*(2*(A + B)*(3 + H) - (A + 3*A*H + 4*B*(1 + H) + \
            4*(1 + H)*C*np.cos(2*phi))*(np.sin(theta)**2)\
            + (1 + 3*H)*(B + C*np.cos(2*phi))*(np.sin(theta)**4))*a11 + (1 + 3*H)*(B + C*np.cos(2*phi))\
            *(np.sin(2*theta)**2)*a13 + 4*(2 - 2*H + (1 + 3*H)*(np.sin(theta)**2))*(A + (B + C*np.cos(2*phi))\
            *(np.sin(theta)**2))*a33)*Betafac1 + Betafac2)
        return Mval

    @staticmethod    
    def Qphiphi2(theta, phi, Beta2, I2, A, B, C, H, a11, a13, a33):
        # Calculates Fisher Information Matrix element Q(phi, phi)2 for
        # scattering case
        # ------ INPUTS ------ #
        # theta is angle in radians
        # phi is angle in radians
        # Beta2 is SBR for detector 1 should be wavelength matrix OR scalar
        # I2 is Intensity1/(1-Beta1^{-1}) should be wavelength matrix OR scalar
        # A, B, C, H are instrument factors (wavelength dependent)
        # a11, a13 and a33 are polarizability tensor elements * illumination
        # ------ OUTPUT ------ #
        # Qval Q function for phi, phi 2
        if np.any(Beta2): # if we're taking background into account
            Betafac = (((1./Beta2) - 1)**2) # calculate background factor
        else: # if not
            Betafac = 1 # ignore it
        Qval = (C**2)*(np.sin(theta)**4)*(np.cos(2*phi)**2)*(((7 + 5*H + (1 + 3*H)*np.cos(2*theta))*a11\
            - 2*(1 + 3*H)*(np.cos(theta)**2)*a13 + (-5 + H + (1 + 3*H)*np.cos(2*theta))*a33)**2)\
            *Betafac*I2
        return Qval

    @staticmethod    
    def Mphiphi2(theta, phi, Beta2, A, B, C, H, NF, a11, a13, a33):
        # Calculates Fisher Information Matrix element M(phi, phi)2 for
        # scattering case
        # ------ INPUTS ------ #
        # theta is angle in radians
        # phi is angle in radians
        # Beta2 is SBR for detector 1 should be wavelength matrix OR scalar
        # A, B, C, H are instrument factors (wavelength dependent)
        # a11, a13 and a33 are polarizability tensor elements * illumination
        # ------ OUTPUT ------ #
        # Mval M function for phi, phi 2
        if np.any(Beta2):
            Betafac1 = (1 - (1./Beta2))
            Betafac2 = ((16*NF)/Beta2)
        else:
            Betafac1 = 1
            Betafac2 = 0
        Mval = NF*((4*(2*(A + B)*(3 + H) + (1 + 3*H)*(np.sin(theta)**4)*(B + C*np.sin(2*phi)) \
            - (np.sin(theta)**2)*(A + 3*A*H + 4*B*(1 + H) + 4*(1 + H)*C*np.sin(2*phi)))*a11 + \
            (1 + 3*H)*(np.sin(2*theta)**2)*(B + C*np.sin(2*phi))*a13 + 4*(2 - 2*H + (1 + 3*H)*(np.sin(theta)**2))\
            *(A + (np.sin(theta)**2)*(B + C*np.sin(2*phi)))*a33)*Betafac1 + Betafac2)
        return Mval

    @staticmethod    
    def Qphiphi3(theta, phi, Beta3, I3, A, B, C, H, a11, a13, a33):
        # Calculates Fisher Information Matrix element Q(phi, phi)3 for
        # scattering case
        # ------ INPUTS ------ #
        # theta is angle in radians
        # phi is angle in radians
        # Beta3 is SBR for detector 1 should be wavelength matrix OR scalar
        # I3 is Intensity1/(1-Beta1^{-1}) should be wavelength matrix OR scalar
        # A, B, C, H are instrument factors (wavelength dependent)
        # a11, a13 and a33 are polarizability tensor elements * illumination
        # ------ OUTPUT ------ #
        # Qval Q function for phi, phi 3
        if np.any(Beta3): # if we're taking background into account
            Betafac = (((1./Beta3) - 1)**2) # calculate background factor
        else: # if not
            Betafac = 1 # ignore it
        Qval = (C**2)*(np.sin(theta)**4)*(np.sin(2*phi)**2)*(((7 + 5*H + (1 + 3*H)*np.cos(2*theta))*a11\
            - 2*(1 + 3*H)*(np.cos(theta)**2)*a13 + (-5 + H + (1 + 3*H)*np.cos(2*theta))*a33)**2)\
            *Betafac*I3
        return Qval

    @staticmethod    
    def Mphiphi3(theta, phi, Beta3, A, B, C, H, NF, a11, a13, a33):
        # Calculates Fisher Information Matrix element M(phi, phi)3 for
        # scattering case
        # ------ INPUTS ------ #
        # theta is angle in radians
        # phi is angle in radians
        # Beta3 is SBR for detector 1 should be wavelength matrix OR scalar
        # A, B, C, H are instrument factors (wavelength dependent)
        # a11, a13 and a33 are polarizability tensor elements * illumination
        # ------ OUTPUT ------ #
        # Mval M function for phi, phi 3
        if np.any(Beta3):
            Betafac1 = (1 - (1./Beta3))
            Betafac2 = ((16*NF)/Beta3)
        else:
            Betafac1 = 1
            Betafac2 = 0
        Mval = NF*((4*(2*(A + B)*(3 + H) - (A + 3*A*H + 4*B*(1 + H) - \
            4*(1 + H)*C*np.cos(2*phi))*(np.sin(theta)**2)\
            + (1 + 3*H)*(B - C*np.cos(2*phi))*(np.sin(theta)**4))*a11 + (1 + 3*H)*(B - C*np.cos(2*phi))\
            *(np.sin(2*theta)**2)*a13 + 4*(2 - 2*H + (1 + 3*H)*(np.sin(theta)**2))*(A + (B - C*np.cos(2*phi))\
            *(np.sin(theta)**2))*a33)*Betafac1 + Betafac2)
        return Mval

    @staticmethod    
    def Qphiphi4(theta, phi, Beta4, I4, A, B, C, H, a11, a13, a33):
        # Calculates Fisher Information Matrix element Q(phi, phi)4 for
        # scattering case
        # ------ INPUTS ------ #
        # theta is angle in radians
        # phi is angle in radians
        # Beta3 is SBR for detector 1 should be wavelength matrix OR scalar
        # I3 is Intensity1/(1-Beta1^{-1}) should be wavelength matrix OR scalar
        # A, B, C, H are instrument factors (wavelength dependent)
        # n_detectors is how many detectors used
        # ------ OUTPUT ------ #
        # Qval Q function for phi, phi 1
        if np.any(Beta4): # if we're taking background into account
            Betafac = (((1./Beta4) - 1)**2) # calculate background factor
        else: # if not
            Betafac = 1 # ignore it
        Qval = (C**2)*(np.sin(theta)**4)*(np.cos(2*phi)**2)*(((7 + 5*H + (1 + 3*H)*np.cos(2*theta))*a11\
            - 2*(1 + 3*H)*(np.cos(theta)**2)*a13 + (-5 + H + (1 + 3*H)*np.cos(2*theta))*a33)**2)\
            *Betafac*I4
        return Qval

    @staticmethod    
    def Mphiphi4(theta, phi, Beta4, A, B, C, H, NF, a11, a13, a33):
        # Calculates Fisher Information Matrix element M(phi, phi)4 for
        # scattering case
        # ------ INPUTS ------ #
        # theta is angle in radians
        # phi is angle in radians
        # Beta3 is SBR for detector 1 should be wavelength matrix OR scalar
        # A, B, C, H are instrument factors (wavelength dependent)
        # n_detectors is how many detectors used
        # ------ OUTPUT ------ #
        # Mval M function for phi, phi 4
        if np.any(Beta4):
            Betafac1 = (1 - (1./Beta4))
            Betafac2 = ((16*NF)/Beta4)
        else:
            Betafac1 = 1
            Betafac2 = 0
        Mval = NF*((4*(2*(A + B)*(3 + H) + (1 + 3*H)*(np.sin(theta)**4)*(B - C*np.sin(2*phi)) \
            - (np.sin(theta)**2)*(A + 3*A*H + 4*B*(1 + H) - 4*(1 + H)*C*np.sin(2*phi)))*a11 + \
            (1 + 3*H)*(np.sin(2*theta)**2)*(B - C*np.sin(2*phi))*a13 + 4*(2 - 2*H + (1 + 3*H)*(np.sin(theta)**2))\
            *(A + (np.sin(theta)**2)*(B - C*np.sin(2*phi)))*a33)*Betafac1 + Betafac2)
        return Mval

    def FisherMatrixScatter(self, time, theta, phi, I, Beta, A, B, C, H, NF, a11, a13, a33, n_detectors=4):
        # Calculates Fisher Information Matrix element for
        # scattering case
        # ------ INPUTS ------ #
        # time is time taken to observe photons
        # theta is angle in radians
        # phi is angle in radians
        # I is Intensity1/(1-Beta1^{-1}) should be n_detector*wavelength matrix OR scalar
        # Beta is SBR for detector 1 should be n_detector*wavelength matrix OR scalar
        # A, B, C, H are instrument factors (wavelength dependent)
        # a11, a13 and a33 are polarizability tensor elements * illumination
        # n_detectors is how many detectors used
        # ------ OUTPUT ------ #
        # FisherInfoM fisher information matrix for parameters
        I = self.Itest(I, A, n_detectors)
        Beta = self.Betatest(Beta, A, n_detectors)
        if n_detectors == 3:
            I1, I2, I3 = self.Irecast(I, n_detectors)
            Beta1, Beta2, Beta3 = self.Betarecast(Beta, n_detectors)
        else:
            I1, I2, I3, I4 = self.Irecast(I, n_detectors)
            Beta1, Beta2, Beta3, Beta4 = self.Betarecast(Beta, n_detectors)
        FisherInfoM = np.zeros([2, 2])
        Qtheta1 = self.Qthetatheta1(theta, phi, Beta1, I1, A, B, C, H, a11, a13, a33)
        Mtheta1 = self.Mthetatheta1(theta, phi, Beta1, A, B, C, H, NF, a11, a13, a33)
        QMtt1 = np.sum(np.divide(Qtheta1, Mtheta1))
        
        Qtheta2 = self.Qthetatheta2(theta, phi, Beta2, I2, A, B, C, H, a11, a13, a33)
        Mtheta2 = self.Mthetatheta2(theta, phi, Beta2, A, B, C, H, NF, a11, a13, a33)
        QMtt2 = np.sum(np.divide(Qtheta2, Mtheta2))
        
        Qtheta3 = self.Qthetatheta3(theta, phi, Beta3, I3, A, B, C, H, a11, a13, a33)
        Mtheta3 = self.Mthetatheta3(theta, phi, Beta3, A, B, C, H, NF, a11, a13, a33)
        QMtt3 = np.sum(np.divide(Qtheta3, Mtheta3))
        
        if n_detectors == 4:
            Qtheta4 = self.Qthetatheta4(theta, phi, Beta4, I4, A, B, C, H, a11, a13, a33)
            Mtheta4 = self.Mthetatheta4(theta, phi, Beta4, A, B, C, H, NF, a11, a13, a33)
            QMtt4 = np.sum(np.divide(Qtheta4, Mtheta4))
            
        if n_detectors == 4:
            FisherInfoM[0, 0] = np.multiply(time, np.sum([QMtt1, QMtt2, QMtt3, QMtt4]))
        else:
            FisherInfoM[0, 0] = np.multiply(time, np.sum([QMtt1, QMtt2, QMtt3]))
        
        Qthetaphi1 = self.Qthetaphi1(theta, phi, Beta1, I1, A, B, C, H, a11, a13, a33)
        Mthetaphi1 = self.Mthetaphi1(theta, phi, Beta1, A, B, C, H, NF, a11, a13, a33)
        QMtp1 = np.sum(np.divide(Qthetaphi1, Mthetaphi1))
        
        Qthetaphi2 = self.Qthetaphi2(theta, phi, Beta2, I2, A, B, C, H, a11, a13, a33)
        Mthetaphi2 = self.Mthetaphi2(theta, phi, Beta2, A, B, C, H, NF, a11, a13, a33)
        QMtp2 = np.sum(np.divide(Qthetaphi2, Mthetaphi2))
        
        Qthetaphi3 = self.Qthetaphi3(theta, phi, Beta3, I3, A, B, C, H, a11, a13, a33)
        Mthetaphi3 = self.Mthetaphi3(theta, phi, Beta3, A, B, C, H, NF, a11, a13, a33)
        QMtp3 = np.sum(np.divide(Qthetaphi3, Mthetaphi3))
        
        if n_detectors == 4:
            Qthetaphi4 = self.Qthetaphi4(theta, phi, Beta4, I4, A, B, C, H, a11, a13, a33)
            Mthetaphi4 = self.Mthetaphi4(theta, phi, Beta4, A, B, C, H, NF, a11, a13, a33)
            QMtp4 = np.sum(np.divide(Qthetaphi4, Mthetaphi4))

        if n_detectors == 4:
            FisherInfoM[0, 1] = np.multiply(time, np.sum([QMtp1, -QMtp2, -QMtp3, QMtp4]))
            FisherInfoM[1,0] = deepcopy(FisherInfoM[0,1])
        else:
            FisherInfoM[0, 1] = np.multiply(time, np.sum([QMtp1, -QMtp2, -QMtp3]))
            FisherInfoM[1,0] = deepcopy(FisherInfoM[0,1])

        Qphiphi1 = self.Qphiphi1(theta, phi, Beta1, I1, A, B, C, H, a11, a13, a33)
        Mphiphi1 = self.Mphiphi1(theta, phi, Beta1, A, B, C, H, NF, a11, a13, a33)
        QMpp1 = np.sum(np.divide(Qphiphi1, Mphiphi1))
        
        Qphiphi2 = self.Qphiphi2(theta, phi, Beta2, I2, A, B, C, H, a11, a13, a33)
        Mphiphi2 = self.Mphiphi2(theta, phi, Beta2, A, B, C, H, NF, a11, a13, a33)
        QMpp2 = np.sum(np.divide(Qphiphi2, Mphiphi2))
        
        Qphiphi3 = self.Qphiphi3(theta, phi, Beta3, I3, A, B, C, H, a11, a13, a33)
        Mphiphi3 = self.Mphiphi3(theta, phi, Beta3, A, B, C, H, NF, a11, a13, a33)
        QMpp3 = np.sum(np.divide(Qphiphi3, Mphiphi3))
        
        if n_detectors == 4:
            Qphiphi4 = self.Qphiphi4(theta, phi, Beta4, I4, A, B, C, H, a11, a13, a33)
            Mphiphi4 = self.Mphiphi4(theta, phi, Beta4, A, B, C, H, NF, a11, a13, a33)
            QMpp4 = np.sum(np.divide(Qphiphi4, Mphiphi4))
            
        if n_detectors == 4:
            FisherInfoM[1, 1] = np.multiply(time, np.sum([QMpp1, QMpp2, QMpp3, QMpp4]))
        else:
            FisherInfoM[1, 1] = np.multiply(time, np.sum([QMpp1, QMpp2, QMpp3]))

        return FisherInfoM

    def FisherMatrixFluo(self, time, theta, phi, I, Beta, A, B, C, n_detectors=4):
        # Calculates Fisher Information Matrix element for
        # fluorescence case
        # ------ INPUTS ------ #
        # time is time taken to observe photons
        # theta is angle in radians
        # phi is angle in radians
        # I is Intensity1/(1-Beta1^{-1}) should be n_detector*wavelength matrix OR scalar
        # Beta is SBR for detector 1 should be n_detector*wavelength matrix OR scalar
        # A, B, C are instrument factors (wavelength dependent)
        # n_detectors is how many detectors used
        # ------ OUTPUT ------ #
        # FisherInfoM fisher information matrix for parameters
        FisherInfoM = np.zeros([2, 2])
        if np.any(Beta):
            FisherInfoM[0,0] = self.Jthetathetaf(theta, phi, time, Beta, I, A, B, C, n_detectors)
            FisherInfoM[0,1] = self.Jthetaphif(theta, phi, time, Beta, I, A, B, C, n_detectors)
            FisherInfoM[1,0] = deepcopy(FisherInfoM[0,1])
            FisherInfoM[1,1] =self.Jphiphif(theta, phi, time, Beta, I, A, B, C, n_detectors)
        else:
            FisherInfoM[0,0] = self.JthetathetafNB(theta, phi, time, I, A, B, C, n_detectors)
            FisherInfoM[0,1] = self.JthetaphifNB(theta, phi, time, I, A, B, C, n_detectors)
            FisherInfoM[1,0] = deepcopy(FisherInfoM[0,1])
            FisherInfoM[1,1] =self.JphiphifNB(theta, phi, time, I, A, B, C, n_detectors)
        return FisherInfoM
    
    def CramerRaoScatter(self, time, thetas, phis, I, Beta, wavelengths, T, cv, NAObj,\
                         NACond, a11, a13, a33, n_detectors=4, thetadepf=lambda theta: 1):
        # Calculates CRLB for scattering case
        # ------ INPUTS ------ #
        # time is time taken to observe photons
        # thetas are angles in radians
        # phis are angle in radians
        # I is Intensity1/(1-Beta1^{-1}) should be n_detector*wavelength matrix OR scalar
        # Beta is SBR for detector 1 should be n_detector*wavelength matrix OR scalar
        # wavelengths are wavelengths of light in nm
        # T is temperature in Kelvin
        # NAObj is objective NA
        # NACond is condenser NA
        # a11, a13 and a33 are polarizability tensor elements * illumination
        # n_detectors is how many detectors used
        # thetadepf is theta intensity dependence (optional)
        # ------ OUTPUTS ------ #
        # CramerRaoMTheta is precision of theta parameter for parameters
        # CramerRaoMPhi is precision of phi parameter for parameters
        if hasattr(thetas, "__len__") and hasattr(phis, "__len__"):
            CramerRaoMTheta = np.zeros([len(thetas), len(phis)])
            CramerRaoMPhi = np.zeros([len(thetas), len(phis)])
        elif hasattr(thetas, "__len__") and not hasattr(phis, "__len__"):
            CramerRaoMTheta = np.zeros([len(thetas), 1])
            CramerRaoMPhi = np.zeros([len(thetas), 1])
        elif not hasattr(thetas, "__len__") and hasattr(phis, "__len__"):
            CramerRaoMTheta = np.zeros([1, len(phis)])
            CramerRaoMPhi = np.zeros([1, len(phis)])
        else:
            CramerRaoMTheta = np.zeros([1, 1])
            CramerRaoMPhi = np.zeros([1, 1])
            
        n0 = self.n_m(wavelengths, T, cv);
        A, B, C, H = self.InstrResp(NACond, NAObj, n0)[2:]

        if hasattr(thetas, "__len__") and hasattr(phis, "__len__"):
            for i in enumerate(thetas):
                loct = i[0]
                theta = i[1]
                NF = self.NormFactorScatter(theta, a11, a13, a33, A, B, H)
                Ival = np.multiply(I, thetadepf(theta))
                for j in enumerate(phis):
                    locp = j[0]
                    phi = j[1]
                    J = self.FisherMatrixScatter(time, theta, phi, Ival, Beta,\
                A, B, C, H, NF, a11, a13, a33, n_detectors)
                    InvJ = self.InvertMatrix(J)
                    CramerRaoMTheta[loct, locp] = np.real(np.sqrt(InvJ[0,0]))
                    CramerRaoMPhi[loct, locp] = np.real(np.sqrt(InvJ[1,1]))
        elif hasattr(thetas, "__len__") and not hasattr(phis, "__len__"):
            for i in enumerate(thetas):
                loct = i[0]
                theta = i[1]
                NF = self.NormFactorScatter(theta, a11, a13, a33, A, B, H)
                Ival = np.multiply(I, thetadepf(theta))
                J = self.FisherMatrixScatter(time, theta, phis, Ival, Beta,\
                A, B, C, H, NF, a11, a13, a33, n_detectors)
                InvJ = self.InvertMatrix(J)
                CramerRaoMTheta[loct, 0] = np.real(np.sqrt(InvJ[0,0]))
                CramerRaoMPhi[loct, 0] = np.real(np.sqrt(InvJ[1,1]))
        elif not hasattr(thetas, "__len__") and hasattr(phis, "__len__"):
            NF = self.NormFactorScatter(thetas, a11, a13, a33, A, B, H)
            Ival = np.multiply(I, thetadepf(theta))
            for j in enumerate(phis):
                locp = j[0]
                phi = j[1]
                J = self.FisherMatrixScatter(time, thetas, phi, Ival, Beta,\
            A, B, C, H, NF, a11, a13, a33, n_detectors)
                InvJ = self.InvertMatrix(J)
                CramerRaoMTheta[0, locp] = np.real(np.sqrt(InvJ[0,0]))
                CramerRaoMPhi[0, locp] = np.real(np.sqrt(InvJ[1,1]))
        else:
            NF = self.NormFactorScatter(thetas, a11, a13, a33, A, B, H)
            Ival = np.multiply(I, thetadepf(theta))
            J = self.FisherMatrixScatter(time, thetas, phis, Ival, Beta,\
            A, B, C, H, NF, a11, a13, a33, n_detectors)
            InvJ = self.InvertMatrix(J)
            CramerRaoMTheta[0, 0] = np.real(np.sqrt(InvJ[0,0]))
            CramerRaoMPhi[0, 0] = np.real(np.sqrt(InvJ[1,1]))
        
        CramerRaoMTheta, CramerRaoMPhi = self.DefineLimits(CramerRaoMTheta, CramerRaoMPhi)
        return CramerRaoMTheta, CramerRaoMPhi

    def CramerRaoFluo(self, time, thetas, phis, I, Beta, wavelengths, T, cv, NAObj,\
                         n_detectors=4, thetadepf=lambda theta: 1):
        # Calculates CRLB for fluorescence case
        # ------ INPUTS ------ #
        # time is time taken to observe photons
        # thetas are angles in radians
        # phis are angle in radians
        # I is Intensity1/(1-Beta1^{-1}) should be n_detector*wavelength matrix OR scalar
        # Beta is SBR for detector 1 should be n_detector*wavelength matrix OR scalar
        # wavelengths are wavelengths of light in nm
        # T is temperature in Kelvin
        # NAObj is objective NA
        # n_detectors is how many detectors used
        # thetadepf is theta intensity dependence (optional)
        # ------ OUTPUTS ------ #
        # CramerRaoMTheta is precision of theta parameter for parameters
        # CramerRaoMPhi is precision of phi parameter for parameters
        if hasattr(thetas, "__len__") and hasattr(phis, "__len__"):
            CramerRaoMTheta = np.zeros([len(thetas), len(phis)])
            CramerRaoMPhi = np.zeros([len(thetas), len(phis)])
        elif hasattr(thetas, "__len__") and not hasattr(phis, "__len__"):
            CramerRaoMTheta = np.zeros([len(thetas), 1])
            CramerRaoMPhi = np.zeros([len(thetas), 1])
        elif not hasattr(thetas, "__len__") and hasattr(phis, "__len__"):
            CramerRaoMTheta = np.zeros([1, len(phis)])
            CramerRaoMPhi = np.zeros([1, len(phis)])
        else:
            CramerRaoMTheta = np.zeros([1, 1])
            CramerRaoMPhi = np.zeros([1, 1])
            
            
        n0 = self.n_m(wavelengths, T, cv);
        A, B, C = self.InstrResp(1, NAObj, n0)[2:5]

        if hasattr(thetas, "__len__") and hasattr(phis, "__len__"):
            for i in enumerate(thetas):
                loct = i[0]
                theta = i[1]
                Ival = np.multiply(I, thetadepf(theta))
                for j in enumerate(phis):
                    locp = j[0]
                    phi = j[1]
                    J = self.FisherMatrixFluo(time, theta, phi, Ival, Beta,\
                                              A, B, C, n_detectors)
                    InvJ = self.InvertMatrix(J)
                    CramerRaoMTheta[loct, locp] = np.real(np.sqrt(InvJ[0,0]))
                    CramerRaoMPhi[loct, locp] = np.real(np.sqrt(InvJ[1,1]))
        elif hasattr(thetas, "__len__") and not hasattr(phis, "__len__"):
            for i in enumerate(thetas):
                loct = i[0]
                theta = i[1]
                Ival = np.multiply(I, thetadepf(theta))
                J = self.FisherMatrixFluo(time, theta, phis, Ival, Beta,\
                                          A, B, C, n_detectors)
                InvJ = self.InvertMatrix(J)
                CramerRaoMTheta[loct, 0] = np.real(np.sqrt(InvJ[0,0]))
                CramerRaoMPhi[loct, 0] = np.real(np.sqrt(InvJ[1,1]))
        elif not hasattr(thetas, "__len__") and hasattr(phis, "__len__"):
            Ival = np.multiply(I, thetadepf(theta))
            for j in enumerate(phis):
                locp = j[0]
                phi = j[1]
                J = self.FisherMatrixFluo(time, thetas, phi, Ival, Beta,\
                                          A, B, C, n_detectors)
                InvJ = self.InvertMatrix(J)
                CramerRaoMTheta[0, locp] = np.real(np.sqrt(InvJ[0,0]))
                CramerRaoMPhi[0, locp] = np.real(np.sqrt(InvJ[1,1]))
        else:
            Ival = np.multiply(I, thetadepf(theta))
            J = self.FisherMatrixFluo(time, thetas, phis, Ival, Beta,\
                                      A, B, C, n_detectors)
            InvJ = self.InvertMatrix(J)
            CramerRaoMTheta[0, 0] = np.real(np.sqrt(InvJ[0,0]))
            CramerRaoMPhi[0, 0] = np.real(np.sqrt(InvJ[1,1]))
            
        
        CramerRaoMTheta, CramerRaoMPhi = self.DefineLimits(CramerRaoMTheta, CramerRaoMPhi)
        return CramerRaoMTheta, CramerRaoMPhi
    
