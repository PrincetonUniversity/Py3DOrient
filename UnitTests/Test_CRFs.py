#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thursday June 17 11.13am

@author: jbeckwith
"""
import sys
sys.path.append("..") # Adds higher directory to python modules path.
from CramerRaoFunctions import CRFs
import pytest
import numpy as np

""" run from the command line with pytest Test_CRFs.py """
""" Checking gives same results as mathematica notebook """

class TestClass:
        
    @classmethod
    def setup_class(self):
        # setup things
        self.cv = 1
        self.wavelength = 309.9604
        self.T = 293.15
        self.CR = CRFs()
        self.n0 = self.CR.n_m(self.wavelength, self.T, self.cv)
        return
    
    @classmethod
    def teardown_class(self):
        # teardown things
        del self.CR
        del self.cv
        del self.wavelength
        del self.T
        del self.n0
        return
    
    def test_Jthetathetaf(self):
        A, B, C = self.CR.InstrResp(1,1.3,self.n0)[2:5]
        theta = np.pi/4
        phi = np.pi/2
        time= 1
        Beta = 3
        I = 25
        assert pytest.approx(self.CR.Jthetathetaf(theta, phi, time, Beta, I, A, B, C, 4), 2.2548) 
        assert pytest.approx(self.CR.JthetathetafNB(theta, phi, time, I, A, B, C, 3), 16.3171) 
        assert pytest.approx(self.CR.JthetathetafNB(theta, phi, time, I, A, B, C, 4), 16.3171) 
        
    def test_Jthetaphif(self):
        A, B, C = self.CR.InstrResp(1,1.3,self.n0)[2:5]
        theta = np.pi/4
        phi = np.pi/3
        time= 1
        Beta = 3
        I = 25
        assert pytest.approx(self.CR.Jthetaphif(theta, phi, time, Beta, I, A, B, C, 4), 0.0339) 
        assert pytest.approx(self.CR.JthetaphifNB(theta, phi, time, I, A, B, C, 3), 4.80397) 
        assert pytest.approx(self.CR.JthetaphifNB(theta, phi, time, I, A, B, C, 4), -1.97489) 

    def test_Jphiphif(self):
        A, B, C = self.CR.InstrResp(1,1.3,self.n0)[2:5]
        theta = np.pi/4
        phi = np.pi/2
        time= 1
        Beta = 3
        I = 25
        assert pytest.approx(self.CR.Jphiphif(theta, phi, time, Beta, I, A, B, C, 4), 4.51353) 
        assert pytest.approx(self.CR.JphiphifNB(theta, phi, time, I, A, B, C, 3), 10.1554) 
        assert pytest.approx(self.CR.JphiphifNB(theta, phi, time, I, A, B, C, 4), 20.3109) 
    
    def testQthetatheta1(self):
        A, B, C, H = self.CR.InstrResp(1.3,0.7,self.n0)[2:]
        a11 = 0.3
        a13 = 0.1
        a33 = 0.6
        theta = np.pi/4
        phi = np.pi/2
        Beta = 1e4
        I = 500
        assert pytest.approx(self.CR.Qthetatheta1(theta, phi, Beta, I, A, B, C, H, a11, a13, a33), 0.1166) 
        assert pytest.approx(self.CR.Qthetatheta1(theta, phi, 0, Beta, A, B, C, H, a11, a13, a33), 2.33325) 
      
    def testMthetatheta1(self):
        A, B, C, H = self.CR.InstrResp(1.3,0.7,self.n0)[2:]
        a11 = 0.3
        a13 = 0.1
        a33 = 0.6
        theta = np.pi/4
        phi = np.pi/2
        Beta = 1e4
        NF = self.CR.NormFactorScatter(theta, a11, a13, a33, A, B, H)
        assert pytest.approx(self.CR.Mthetatheta1(theta, phi, Beta, A, B, C, H, NF, a11, a13, a33), 0.00893141) 
        assert pytest.approx(self.CR.Mthetatheta1(theta, phi, 0, A, B, C, H, a11, NF, a13, a33), 0.00892771) 

    def testQthetatheta2(self):
        A, B, C, H = self.CR.InstrResp(1.3,0.7,self.n0)[2:]
        a11 = 0.3
        a13 = 0.1
        a33 = 0.6
        theta = np.pi/4
        phi = np.pi/3
        Beta = 1e4
        I = 500
        assert pytest.approx(self.CR.Qthetatheta2(theta, phi, Beta, I, A, B, C, H, a11, a13, a33), 0.218698) 
        assert pytest.approx(self.CR.Qthetatheta2(theta, phi, 0, Beta, A, B, C, H, a11, a13, a33), 0.437484) 

    def testMthetatheta2(self):
        A, B, C, H = self.CR.InstrResp(1.3,0.7,self.n0)[2:]
        a11 = 0.3
        a13 = 0.1
        a33 = 0.6
        theta = np.pi/4
        phi = np.pi/2
        Beta = 1e4
        NF = self.CR.NormFactorScatter(theta, a11, a13, a33, A, B, H)
        assert pytest.approx(self.CR.Mthetatheta2(theta, phi, Beta, A, B, C, H, NF, a11, a13, a33), 0.00287733) 
        assert pytest.approx(self.CR.Mthetatheta2(theta, phi, 0, A, B, C, H, a11, NF, a13, a33), 0.00287646) 

    def testQthetatheta3(self):
        A, B, C, H = self.CR.InstrResp(1.3,0.7,self.n0)[2:]
        a11 = 0.3
        a13 = 0.1
        a33 = 0.6
        theta = np.pi/4
        phi = np.pi/2
        Beta = 1e4
        I = 500
        assert pytest.approx(self.CR.Qthetatheta3(theta, phi, Beta, I, A, B, C, H, a11, a13, a33), 1.16639) 
        assert pytest.approx(self.CR.Qthetatheta3(theta, phi, 0, Beta, A, B, C, H, a11, a13, a33), 2.33325) 

    def testMthetatheta3(self):
        A, B, C, H = self.CR.InstrResp(1.3,0.7,self.n0)[2:]
        a11 = 0.3
        a13 = 0.1
        a33 = 0.6
        theta = np.pi/4
        phi = np.pi/2
        Beta = 1e4
        NF = self.CR.NormFactorScatter(theta, a11, a13, a33, A, B, H)
        assert pytest.approx(self.CR.Mthetatheta3(theta, phi, Beta, A, B, C, H, NF, a11, a13, a33), 0.0140872) 
        assert pytest.approx(self.CR.Mthetatheta3(theta, phi, 0, A, B, C, H, a11, NF, a13, a33), 0.014084) 

    def testQthetatheta4(self):
        A, B, C, H = self.CR.InstrResp(1.3,0.7,self.n0)[2:]
        a11 = 0.3
        a13 = 0.1
        a33 = 0.6
        theta = np.pi/4
        phi = np.pi/3
        Beta = 1e4
        I = 500
        assert pytest.approx(self.CR.Qthetatheta4(theta, phi, Beta, I, A, B, C, H, a11, a13, a33), 0.218698) 
        assert pytest.approx(self.CR.Qthetatheta4(theta, phi, 0, Beta, A, B, C, H, a11, a13, a33), 0.437484) 

    def testMthetatheta4(self):
        A, B, C, H = self.CR.InstrResp(1.3,0.7,self.n0)[2:]
        a11 = 0.3
        a13 = 0.1
        a33 = 0.6
        theta = np.pi/4
        phi = np.pi/2
        Beta = 1e4
        NF = self.CR.NormFactorScatter(theta, a11, a13, a33, A, B, H)
        assert pytest.approx(self.CR.Mthetatheta3(theta, phi, Beta, A, B, C, H, NF, a11, a13, a33), 0.00287733) 
        assert pytest.approx(self.CR.Mthetatheta3(theta, phi, 0, A, B, C, H, a11, NF, a13, a33), 0.00287646) 
        
    def testQthetaphi1(self):
        A, B, C, H = self.CR.InstrResp(1.3,0.7,self.n0)[2:]
        a11 = 0.3
        a13 = 0.1
        a33 = 0.6
        theta = np.pi/4
        phi = np.pi/3
        Beta = 1e4
        I = 500
        assert pytest.approx(self.CR.Qthetaphi1(theta, phi, Beta, I, A, B, C, H, a11, a13, a33), 0.0857901) 
        assert pytest.approx(self.CR.Qthetaphi1(theta, phi, 0, Beta, A, B, C, H, a11, a13, a33), 1.71615) 

    def testMthetaphi1(self):
        A, B, C, H = self.CR.InstrResp(1.3,0.7,self.n0)[2:]
        a11 = 0.3
        a13 = 0.1
        a33 = 0.6
        theta = np.pi/4
        phi = np.pi/3
        Beta = 1e4
        NF = self.CR.NormFactorScatter(theta, a11, a13, a33, A, B, H)
        assert pytest.approx(self.CR.Mthetaphi1(theta, phi, Beta, A, B, C, H, NF, a11, a13, a33), 0.0110329) 
        assert pytest.approx(self.CR.Mthetaphi1(theta, phi, 0, A, B, C, H, a11, NF, a13, a33), 0.0110291) 

    def testQthetaphi2(self):
        A, B, C, H = self.CR.InstrResp(1.3,0.7,self.n0)[2:]
        a11 = 0.3
        a13 = 0.1
        a33 = 0.6
        theta = np.pi/4
        phi = np.pi/3
        Beta = 1e4
        I = 500
        assert pytest.approx(self.CR.Qthetaphi1(theta, phi, Beta, I, A, B, C, H, a11, a13, a33), 0.0428951) 
        assert pytest.approx(self.CR.Qthetaphi1(theta, phi, 0, Beta, A, B, C, H, a11, a13, a33), 0.858073) 

    def testMthetaphi2(self):
        A, B, C, H = self.CR.InstrResp(1.3,0.7,self.n0)[2:]
        a11 = 0.3
        a13 = 0.1
        a33 = 0.6
        theta = np.pi/4
        phi = np.pi/3
        Beta = 1e4
        NF = self.CR.NormFactorScatter(theta, a11, a13, a33, A, B, H)
        assert pytest.approx(self.CR.Mthetaphi2(theta, phi, Beta, A, B, C, H, NF, a11, a13, a33), 0.00741719) 
        assert pytest.approx(self.CR.Mthetaphi2(theta, phi, 0, A, B, C, H, a11, NF, a13, a33), 0.00741545) 

    def testQthetaphi3(self):
        A, B, C, H = self.CR.InstrResp(1.3,0.7,self.n0)[2:]
        a11 = 0.3
        a13 = 0.1
        a33 = 0.6
        theta = np.pi/4
        phi = np.pi/3
        Beta = 1e4
        I = 500
        assert pytest.approx(self.CR.Qthetaphi3(theta, phi, Beta, I, A, B, C, H, a11, a13, a33), -0.0857901) 
        assert pytest.approx(self.CR.Qthetaphi3(theta, phi, 0, Beta, A, B, C, H, a11, a13, a33), -1.71615) 

    def testMthetaphi3(self):
        A, B, C, H = self.CR.InstrResp(1.3,0.7,self.n0)[2:]
        a11 = 0.3
        a13 = 0.1
        a33 = 0.6
        theta = np.pi/4
        phi = np.pi/3
        Beta = 1e4
        NF = self.CR.NormFactorScatter(theta, a11, a13, a33, A, B, H)
        assert pytest.approx(self.CR.Mthetaphi3(theta, phi, Beta, A, B, C, H, NF, a11, a13, a33), 0.0138158) 
        assert pytest.approx(self.CR.Mthetaphi3(theta, phi, 0, A, B, C, H, a11, NF, a13, a33), 0.0138122) 

    def testQthetaphi4(self):
        A, B, C, H = self.CR.InstrResp(1.3,0.7,self.n0)[2:]
        a11 = 0.3
        a13 = 0.1
        a33 = 0.6
        theta = np.pi/4
        phi = np.pi/3
        Beta = 1e4
        I = 500
        assert pytest.approx(self.CR.Qthetaphi4(theta, phi, Beta, I, A, B, C, H, a11, a13, a33), -0.0428951) 
        assert pytest.approx(self.CR.Qthetaphi4(theta, phi, 0, Beta, A, B, C, H, a11, a13, a33), -0.858073)
        
    def testMthetaphi4(self):
        A, B, C, H = self.CR.InstrResp(1.3,0.7,self.n0)[2:]
        a11 = 0.3
        a13 = 0.1
        a33 = 0.6
        theta = np.pi/4
        phi = np.pi/3
        Beta = 1e4
        NF = self.CR.NormFactorScatter(theta, a11, a13, a33, A, B, H)
        assert pytest.approx(self.CR.Mthetaphi4(theta, phi, Beta, A, B, C, H, NF, a11, a13, a33), 0.00500718) 
        assert pytest.approx(self.CR.Mthetaphi4(theta, phi, 0, A, B, C, H, a11, NF, a13, a33), 0.00500519) 
        
    def testQphiphi1(self):
        A, B, C, H = self.CR.InstrResp(1.3,0.7,self.n0)[2:]
        a11 = 0.3
        a13 = 0.1
        a33 = 0.6
        theta = np.pi/4
        phi = np.pi/3
        Beta = 1e4
        I = 500
        assert pytest.approx(self.CR.Qphiphi1(theta, phi, Beta, I, A, B, C, H, a11, a13, a33), 0.252401) 
        assert pytest.approx(self.CR.Qphiphi1(theta, phi, 0, Beta, A, B, C, H, a11, a13, a33), 5.04903)

    def testMphiphi1(self):
        A, B, C, H = self.CR.InstrResp(1.3,0.7,self.n0)[2:]
        a11 = 0.3
        a13 = 0.1
        a33 = 0.6
        theta = np.pi/4
        phi = np.pi/3
        Beta = 1e4
        NF = self.CR.NormFactorScatter(theta, a11, a13, a33, A, B, H)
        assert pytest.approx(self.CR.Mphiphi1(theta, phi, Beta, A, B, C, H, NF, a11, a13, a33), 0.0119101) 
        assert pytest.approx(self.CR.Mphiphi1(theta, phi, 0, A, B, C, H, a11, NF, a13, a33), 0.011906) 

    def testQphiphi2(self):
        A, B, C, H = self.CR.InstrResp(1.3,0.7,self.n0)[2:]
        a11 = 0.3
        a13 = 0.1
        a33 = 0.6
        theta = np.pi/4
        phi = np.pi/3
        Beta = 1e4
        I = 500
        assert pytest.approx(self.CR.Qphiphi2(theta, phi, Beta, I, A, B, C, H, a11, a13, a33), 0.0841336) 
        assert pytest.approx(self.CR.Qphiphi2(theta, phi, 0, Beta, A, B, C, H, a11, a13, a33), 1.68301)

    def testMphiphi2(self):
        A, B, C, H = self.CR.InstrResp(1.3,0.7,self.n0)[2:]
        a11 = 0.3
        a13 = 0.1
        a33 = 0.6
        theta = np.pi/4
        phi = np.pi/3
        Beta = 1e4
        NF = self.CR.NormFactorScatter(theta, a11, a13, a33, A, B, H)
        assert pytest.approx(self.CR.Mphiphi2(theta, phi, Beta, A, B, C, H, NF, a11, a13, a33), 0.0160138) 
        assert pytest.approx(self.CR.Mphiphi2(theta, phi, 0, A, B, C, H, a11, NF, a13, a33), 0.0160101) 

    def testQphiphi3(self):
        A, B, C, H = self.CR.InstrResp(1.3,0.7,self.n0)[2:]
        a11 = 0.3
        a13 = 0.1
        a33 = 0.6
        theta = np.pi/4
        phi = np.pi/3
        Beta = 1e4
        I = 500
        assert pytest.approx(self.CR.Qphiphi3(theta, phi, Beta, I, A, B, C, H, a11, a13, a33), 0.252401) 
        assert pytest.approx(self.CR.Qphiphi3(theta, phi, 0, Beta, A, B, C, H, a11, a13, a33), 5.04903)

    def testMphiphi3(self):
        A, B, C, H = self.CR.InstrResp(1.3,0.7,self.n0)[2:]
        a11 = 0.3
        a13 = 0.1
        a33 = 0.6
        theta = np.pi/4
        phi = np.pi/3
        Beta = 1e4
        NF = self.CR.NormFactorScatter(theta, a11, a13, a33, A, B, H)
        assert pytest.approx(self.CR.Mphiphi3(theta, phi, Beta, A, B, C, H, NF, a11, a13, a33), 0.0149142) 
        assert pytest.approx(self.CR.Mphiphi3(theta, phi, 0, A, B, C, H, a11, NF, a13, a33), 0.0149104) 

    def testQphiphi4(self):
        A, B, C, H = self.CR.InstrResp(1.3,0.7,self.n0)[2:]
        a11 = 0.3
        a13 = 0.1
        a33 = 0.6
        theta = np.pi/4
        phi = np.pi/3
        Beta = 1e4
        I = 500
        assert pytest.approx(self.CR.Qphiphi4(theta, phi, Beta, I, A, B, C, H, a11, a13, a33), 0.0841336) 
        assert pytest.approx(self.CR.Qphiphi4(theta, phi, 0, Beta, A, B, C, H, a11, a13, a33), 1.68301)

    def testMphiphi4(self):
        A, B, C, H = self.CR.InstrResp(1.3,0.7,self.n0)[2:]
        a11 = 0.3
        a13 = 0.1
        a33 = 0.6
        theta = np.pi/4
        phi = np.pi/3
        Beta = 1e4
        NF = self.CR.NormFactorScatter(theta, a11, a13, a33, A, B, H)
        assert pytest.approx(self.CR.Mphiphi4(theta, phi, Beta, A, B, C, H, NF, a11, a13, a33), 0.0108106) 
        assert pytest.approx(self.CR.Mphiphi4(theta, phi, 0, A, B, C, H, a11, NF, a13, a33), 0.0108063) 
        
    def testFisherInfoS(self):
        A, B, C, H = self.CR.InstrResp(1.3,0.7,self.n0)[2:]
        a11 = 0.3
        a13 = 0.1
        a33 = 0.6
        I = 1000
        Beta = 1e4
        n_detectors = 4
        theta = np.pi/3
        phi = np.pi/4
        NF = self.CR.NormFactorScatter(theta, a11, a13, a33, A, B, H)
        FisherInfoM = self.CR.FisherMatrixScatter(1, theta, phi, I, Beta,\
                A, B, C, H, NF, a11, a13, a33, n_detectors)
        ExactRes = np.array([[1.37303094e+01, 8.40072365e-16], [8.40072365e-16, 1.58893415e+02]])
        assert pytest.approx(FisherInfoM, ExactRes)
        theta = np.pi/3.5
        phi = np.pi/4.5
        FisherInfoM = self.CR.FisherMatrixScatter(1, theta, phi, I, Beta,\
                A, B, C, H, NF, a11, a13, a33, n_detectors)
        ExactRes = np.array([[ 28.06394351,   3.07356373], [3.07356373, 128.39992549]])
        assert pytest.approx(FisherInfoM, ExactRes)
        theta = np.pi/2
        phi = np.pi/4
        FisherInfoM = self.CR.FisherMatrixScatter(1, theta, phi, I, Beta,\
                A, B, C, H, NF, a11, a13, a33, n_detectors)
        ExactRes = np.array([[ 0,   0], [0, 202.0738]])
        assert pytest.approx(FisherInfoM, ExactRes)
        