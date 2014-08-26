#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2014 - Swang <swangi@outlook.com>
# Filename: EAM_AM.py
# +-+-+-+-+-+-+ +-+-+-+-+-+-+-+-+-+
# |I|m|p|o|r|t| |s|o|m|e|t|h|i|n|g|
# +-+-+-+-+-+-+ +-+-+-+-+-+-+-+-+-+
#
import numpy as np

# from sympy.functions.special.delta_functions import Heaviside


def HeavisideTheta(x1, x2):
    if x1 - x2 == 0:
        return 0.5

    return 0 if x1 - x2 < 0 else 1


class pairwise_e:
                # adjust the out V2 and V3 for different app
                        # (times r (LAMMPS) or not (QUIP and ...))

    """Pairwise energy term"""

    def __init__(self, Z1=26, Z2=26, Zcoeff=14.39975826818165680473, r1=0, r2=0,
                 aphik=[], rphik=[], Bn=[], **kwargs):
        self.Z1 = Z1
        self.Z2 = Z2
        self.Zcoeff = Zcoeff
        self.r1 = r1
        self.r2 = r2
        self.aphik = aphik
        self.rphik = rphik
        self.Bn = Bn

    def V1(self, r):
        x = r / (0.88534 * 0.529167 /
                (np.sqrt(self.Z1 ** (2.0 / 3) +
                         self.Z2 ** (2.0 / 3))
                 )
                 )
        V = self.Zcoeff * \
            self.Z1 * self.Z2 * (0.1818 * np.exp(-3.2 * x) +
                                 0.5099 * np.exp(-0.9423 * x) +
                                 0.2802 * np.exp(-0.4029 * x) +
                                 0.02817 * np.exp(-0.2016 * x))

        return V

    def V2(self, r):
        sum_temp = 0
        for n, b in enumerate(self.Bn):
            sum_temp += b * r ** n
        V = np.exp(sum_temp)  # * r
        return V

    def V2_ram(self, r):
        sum_temp = 0
        for n, b in enumerate(self.Bn):
            sum_temp += b * r ** n
        V = sum_temp  # * r
        return V

    def V3(self, r):
        sum_temp = 0
        for k, a in enumerate(self.aphik):
            sum_temp += a * (
                self.rphik[k] - r) ** 3 * HeavisideTheta(self.rphik[k], r)
        V = sum_temp  # * r
        return V

    def phi(self, r):
        V = self.V1(r) * \
            HeavisideTheta(self.r1, r) + \
            self.V2(r) * \
            HeavisideTheta(self.r2, r) * HeavisideTheta(r, self.r1) + \
            self.V3(r) * \
            HeavisideTheta(r, self.r2)
        return V

    def phi_ram(self, r):
        V = self.V1(r) * \
            HeavisideTheta(self.r1, r) + \
            self.V2_ram(r) * \
            HeavisideTheta(self.r2, r) * HeavisideTheta(r, self.r1) + \
            self.V3(r) * \
            HeavisideTheta(r, self.r2)
        return V

    def phi_lammps(self, r):
        V = self.V1(r) * \
            HeavisideTheta(self.r1, r) + \
            self.V2(r) * r * \
            HeavisideTheta(self.r2, r) * HeavisideTheta(r, self.r1) + \
            self.V3(r) * r *\
            HeavisideTheta(r, self.r2)
        return V

    def phi_lammps_ram(self, r):
        V = self.V1(r) * \
            HeavisideTheta(self.r1, r) + \
            self.V2_ram(r) * r * \
            HeavisideTheta(self.r2, r) * HeavisideTheta(r, self.r1) + \
            self.V3(r) * r *\
            HeavisideTheta(r, self.r2)
        return V


class embed_e:

    """embedding energy"""

    def __init__(self, ap=-0.00035387096579929, ap_p=0.0):
        self.ap = ap
        self.ap_p = ap_p

    def F(self, rho):
        V = -rho ** 0.5 + self.ap * rho ** 2 + self.ap_p * rho ** 4
        return V


class embed_d:

    """embedding density"""

    def __init__(self, arhok=[], rrhok=[]):
        self.arhok = arhok
        self.rrhok = rrhok

    def psi(self, r):
        sum_temp = 0
        for k, a in enumerate(self.arhok):
            sum_temp += a * (
                self.rrhok[k] - r) ** 3 * HeavisideTheta(self.rrhok[k], r)
        psi = sum_temp
        return psi


def write_plt(ele, rho, r, embed_energy, embed_density, pair_pot):
    with open('F_%s.plt' % (ele), 'w') as F,  \
        open('f%s.plt' % (ele), 'w') as ff, \
        open('p%s.plt' % (ele), 'w') as p:
        F.write(
            '#  Embedding function for %s:\n#  Electron density Energy (eV)\n'
            % (ele))
        F_fe = zip(rho, embed_energy)
        for key, value in F_fe:
            F.write('%.15E %.15E\n' % (key, value))
        ff.write(
            '# Electron density for %s:\n#  R (A) Electron density \n'
            % (ele))
        ffe = zip(r, embed_density)
        for key, value in ffe:
            ff.write('%.15E %.15E\n' % (key, value))
        p.write(
            '#  Pair interaction function for %s:\n#  R (A) Energy (eV) \n'
            % (ele))
        pfe = zip(r, pair_pot)
        for key, value in pfe:
            p.write('%.15E %.15E\n' % (key, value))

#
# +-+-+-+-+-+ +-+-+ +-+-+-+-+
# |S|t|a|r|t| |o|f| |m|a|i|n|
# +-+-+-+-+-+ +-+-+ +-+-+-+-+
#
# aphik_men = [0.0000000000000, -24.028204854115, 11.300691696477,
#              5.3144495820462, -4.6659532856049, 5.9637758529194,
#              -1.7710262006061, 0.85913830768731, -2.1845362968261,
#              2.6424377007466, -1.0358345370208, 0.33548264951582,
#              -0.046448582149334, -0.0070294963048689, 0.000000000000]
# rphik_men = [0.0, 2.2, 2.3,
#              2.4, 2.5, 2.6,
#              2.7, 2.8, 3.0,
#              3.3, 3.7, 4.2,
#              4.7, 5.3, 0.0]
# r1_men = 1.0
# r2_men = 2.0
# Bn_men = [6.4265260576348, 1.7900488524286, -4.5108316729807, 1.0866199373306]
# arhok_men = [11.686859407970, -0.014710740098830, 0.47193527075943]
# rrhok_men = [2.4, 3.2, 4.2]
