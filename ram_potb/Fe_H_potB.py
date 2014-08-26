#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2014 - Swang <swangi@outlook.com>
# Filename:
# +-+-+-+-+-+-+ +-+-+-+-+-+-+-+-+-+
# |I|m|p|o|r|t| |s|o|m|e|t|h|i|n|g|
# +-+-+-+-+-+-+ +-+-+-+-+-+-+-+-+-+
#
import numpy as np
import os
from EAM_AM import pairwise_e
from EAM_AM import embed_d
# from EAM_AM import embed_e
from EAM_AM import write_plt
import Fe_ackland
import pylab as plt
from time import ctime
from sympy.functions.special.delta_functions import Heaviside

# warning is not logged here. Perfect for clean unit test output

np.errstate(over='ignore')


def pair(r, pairwise_e):
    pair_pot = [0] * len(r)
    pair_pot_lammps = [0] * len(r)
    for i, r in enumerate(r):
        pair_pot[i] = pairwise_e.phi_ram(r)
        pair_pot_lammps[i] = pairwise_e.phi_lammps_ram(r)
    return pair_pot, pair_pot_lammps


def embed_density(r, embed_d):
    embed = [0] * len(r)
    for i, r in enumerate(r):
        embed[i] = embed_d.psi(r)
    return embed


def embed_energy_hh_ram(rhoH, afn):
    FH = [0] * len(rhoH)
    for i, r in enumerate(rhoH):
        FH[i] = afn[0] * rhoH[i] \
            + afn[1] * rhoH[i] ** 2 \
            + afn[2] * rhoH[i] ** 3 \
            + afn[3] * rhoH[i] ** 4  \
            + afn[4] * rhoH[i] ** 5  \
            + afn[5] * rhoH[i] ** 6
    return FH


def write_lammps(ele, func, nrho, nr, dr,
                 drmax, comment_1, comment_2,
                 header=0):
    fname = '%s.txt' % (pot_name)
    with open(fname, 'a') as f:
        if header == 1:
            f.write(
                'Sourse: ' + pot_sourse + '\n' +
                'Made by Swang using Python ' + '//' + ctime() + '//' + '\n' +
                '<swangi@outlook.com>\n')
            f.write('%d ' % (len(ele)))
            for e in ele:
                f.write('%s ' % (e))
            f.write('\n')
            f.write('%d %.15E %d %.15E %.15E\n' % (nrho, drho, nr, dr, drmax))
            f.write('%s\n' % (comment_1))
        if header == 2:
            f.write('%s\n' % (comment_2))
        for i, x in enumerate(func):
            f.write('%.15E ' % (x))
            if (i + 1) % 5 == 0:
                f.write('\n')


def vhh_densityhh_ram(r, afn, r_cut_hh, c1=0.0, c2=0.0):
    a0 = 0.52917721092
    rhoH = [0] * len(r)
    FH = [0] * len(r)
    phi_hh = [0] * len(r)
    phi_hh_lammps = [0] * len(r)
    for i, r in enumerate(r):
        if r > r_cut_hh:
            rhoH[i] = 0
        else:
            rhoH[i] = 1800.0 * r ** 2.0 * \
                np.exp((-2.0 * r / a0) + 1.0 / (r - r_cut_hh))
        FH[i] = afn[0] * rhoH[i] \
            + afn[1] * rhoH[i] ** 2 \
            + afn[2] * rhoH[i] ** 3 \
            + afn[3] * rhoH[i] ** 4  \
            + afn[4] * rhoH[i] ** 5  \
            + afn[5] * rhoH[i] ** 6
        if r > r_cut_hh:
            phi_hh[i] = 0
        else:

            s = 0.5 * (1 - np.tanh(25.0 * (r - 0.9)))
            a = (r - 0.74) / 0.74 / 0.4899
            e_mol = -2.0 * 2.37 * (1.0 + a) * np.exp(-a)
            phi_hh[i] = s * (
                e_mol - 2.0 * FH[i]) \
                + (1.0 - s) * (c1 * np.exp(1 / (r - r_cut_hh)) + c2 * rhoH[i])
        phi_hh_lammps[i] = phi_hh[i] * r
    return phi_hh, phi_hh_lammps, rhoH


def vhh_densityhh_ram_song(r, afn, r_cut_hh, c1=0.0, c2=0.0,
                           lmd=1.0, k=1.5, C0=0.19, B0=1.44, r0=0.9, r1=3.0
                           ):
    a0 = 0.52917721092
    rhoH = [0] * len(r)
    FH = [0] * len(r)
    phi_hh = [0] * len(r)
    phi_hh_lammps = [0] * len(r)
    fsong = [0] * len(r)
    for i, r in enumerate(r):
        if r > r_cut_hh:
            rhoH[i] = 0
        else:
            rhoH[i] = 1800.0 * r ** 2.0 * \
                np.exp((-2.0 * r / a0) + 1.0 / (r - r_cut_hh))
        FH[i] = afn[0] * rhoH[i] \
            + afn[1] * rhoH[i] ** 2 \
            + afn[2] * rhoH[i] ** 3 \
            + afn[3] * rhoH[i] ** 4  \
            + afn[4] * rhoH[i] ** 5  \
            + afn[5] * rhoH[i] ** 6
        s = 0.5 * (1 - np.tanh(25 * (r - 0.9)))
        a = (r - 0.74) / 0.74 / 0.4899
        e_mol = -2 * 2.37 * (1 + a) * np.exp(-a)
        if r > r_cut_hh:
            phi_hh[i] = 0
        else:

            s = 0.5 * (1 - np.tanh(25.0 * (r - 0.9)))
            a = (r - 0.74) / 0.74 / 0.4899
            e_mol = -2.0 * 2.37 * (1.0 + a) * np.exp(-a)
            phi_hh[i] = s * (
                e_mol - 2.0 * FH[i]) \
                + (1.0 - s) * (c1 * np.exp(1 / (r - r_cut_hh)) + c2 * rhoH[i])
        if r >= r1:
            fsong[i] = C0 * ((r - r0) / lmd) ** (k - 1) * np.exp(
                -((r - r0) / lmd) ** k - B0 * (r - r1) ** 2)
        elif r < r1 and r >= r0:
            fsong[i] = C0 * ((r - r0) / lmd) ** (k - 1) * np.exp(
                -((r - r0) / lmd) ** k)
        else:
            fsong[i] = 0

        phi_hh[i] += fsong[i]
        phi_hh_lammps[i] = phi_hh[i] * r
    return phi_hh, phi_hh_lammps, rhoH
# +-+-+-+ +-+-+-+-+
# |Fe|H| |P|a|i|r|
# +-+-+-+ +-+-+-+-+
# +-+-+-+ +-+
# |P|O|T| |A|
# +-+-+-+ +-+
pot_type = "pot_b"

if pot_type == "pot_a":
    r_cut_hh = 2.3
    aphik = [
        14.0786236789212005, -4.4526835887173704,  5.5025121262565992,
        -1.0687489808214079, -0.3461498208163201,
        -0.0064991947759021, -0.0357435602984102]
    rphik = [1.6, 1.7, 1.8, 2.0, 2.5, 3.2, 4.2]
    arhok_feh = [
        10.0073629216300581, 32.4861983261490295, -0.9494226032063788,
        11.6659812262450338, -0.0147080251458273, 0.4943383753319843]
    rrhok_feh = [1.6, 1.8, 2.0, 2.4, 3.2, 4.2]
    arhok_hfe = [11.1667357634216433, -3.0351307365078730,
                 3.6096144794370653, 0.0212509034775648, 0.0303914939946250]
    rrhok_hfe = [1.5, 2.0, 2.5, 3.0, 4.2]
    r1, r2 = 0.6, 1.2
    Bn = [1242.1709168218642, -6013.566711223783, 12339.540893927151,
          -12959.66163724488, 6817.850021676971, -1422.1723964897117]
    afn = [-0.0581256120818134,
           0.0022854552833736,
           - 0.0000314202805805,
           0.0000013764132084,
           - 0.0000000253707731,
           0.0000000001483685]
    Zfe, Zh = 26, 1

# +-+-+-+ +-+
# |P|O|T| |B|
# +-+-+-+ +-+
r_cut_hh = 2.4
aphik = [
    14.0786236789212005, -4.4526835887173704,  5.5025121262565992,
    -1.0687489808214079, -0.3461498208163201,
    -0.0064991947759021, -0.0357435602984102]
rphik = [1.6, 1.7, 1.8, 2.0, 2.5, 3.2, 4.2]
arhok_feh = [
    10.0073629218346891, 32.4862873850836635, -0.9494211670931015,
    11.6683860903729624, -0.0147079871493827, 0.4945807618408609]
rrhok_feh = [1.6, 1.8, 2.0, 2.4, 3.2, 4.2]
arhok_hfe = [11.1667357634216433, -3.0351469477486712,
             3.6092404272928578, 0.0212508491354509, 0.0303904795842773]
rrhok_hfe = [1.5, 2.0, 2.5, 3.0, 4.2]
r1, r2 = 0.6, 1.2
Bn = [1242.154614241987, -6013.4610429013765, 12339.275191444543,
      -12959.339514470237, 6817.662603221567, -1422.130403271231]
afn = [-0.0581047132616673,
       0.0022873205657864,
       -0.0000313966169286,
       0.0000013788174098,
       -0.0000000253074673,
       0.0000000001487789]
Zfe, Zh = 26, 1

dr = 0.000530000
drho = 0.0300000
nr = 10000
nrho = 10000
drmax = dr * nr
drhomax = drho * nrho
r = np.arange(0, drmax, dr)
rho = np.arange(0, drhomax, drho)

pairwise_e = pairwise_e(Z1=Zfe, Z2=Zh, r1=r1, r2=r2,
                        aphik=aphik, rphik=rphik, Bn=Bn)
embed_d_feh = embed_d(arhok=arhok_feh, rrhok=rrhok_feh)
embed_d_hfe = embed_d(arhok=arhok_hfe, rrhok=rrhok_hfe)

embed_density_fefe = Fe_ackland.embed_density
embed_energy_fefe = Fe_ackland.embed_energy
pair_pot_fefe_lammps = Fe_ackland.pair_pot_lammps
pair_pot_fefe = Fe_ackland.pair_pot
pair_pot_feh, pair_pot_feh_lammps = pair(r, pairwise_e)
embed_density_feh = embed_density(r, embed_d_feh)
embed_density_hfe = embed_density(r, embed_d_hfe)
# +-+-+-+-+-+-+-+-+
# |O|r|i|g|i|n|a|l|
# +-+-+-+-+-+-+-+-+

pair_pot_hh, pair_pot_hh_lammps, embed_density_hh =\
    vhh_densityhh_ram(r, afn, r_cut_hh)
embed_energy_hh = embed_energy_hh_ram(rho, afn)
pot_sourse = 'Potential B from \
Ramasubramaniam A, Itakura M, Carter EA. Physical Review B 2009'
pot_name = 'Ram_potB'
comment_1 = '26 5.5847000000000E+0001 2.8557E+0000 BCC'
comment_2 = '1 1.008 1.8 BCC'
ele = ['Fe', 'H']
# +-+-+-+ +-+-+-+-+-+-+ +-+-+-+-+-+-+
# |u|s|e| |s|o|n|g|'|s| |r|e|v|i|s|e|
# +-+-+-+ +-+-+-+-+-+-+ +-+-+-+-+-+-+
# pair_pot_hh, pair_pot_hh_lammps, embed_density_hh =\
#     vhh_densityhh_ram_song(r, afn, r_cut_hh=2.3)

# embed_energy_hh = embed_energy_hh_ram(rho, afn)
# pot_sourse = 'Potential A from \
# Ramasubramaniam A, Itakura M, Carter EA. Physical Review B 2009, \
# revised by Song J, Curtin WA. Nature Materials 2012;12:145.'
# pot_name = 'Ram_song'
# comment_1 = '26 5.5847000000000E+0001 2.8557E+0000 BCC'
# comment_2 = '1 1.008 1.8 BCC'
# ele = ['Fe', 'H']


#
# +-+-+-+-+-+ +-+-+ +-+-+-+-+
# |S|t|a|r|t| |o|f| |m|a|i|n|
# +-+-+-+-+-+ +-+-+ +-+-+-+-+
#
if __name__ == "__main__":
    fname = '%s.txt' % (pot_name)
    f = open(fname, 'w')
    f.close
    # strat from 7
    write_lammps(ele, embed_energy_fefe, nrho,
                 nr, dr, drmax, comment_1, comment_2, header=1)
    # strat from 2007
    write_lammps(ele, embed_density_fefe,
                 nrho, nr, dr, drmax, comment_1, comment_2)
    # strat from 4007
    write_lammps(ele, embed_density_feh, nrho,
                 nr, dr, drmax, comment_1, comment_2)
    # strat from 6008
    write_lammps(ele, embed_energy_hh, nrho,
                 nr, dr, drmax, comment_1, comment_2, header=2)
    # strat from 8008
    write_lammps(ele, embed_density_hfe,
                 nrho, nr, dr, drmax, comment_1, comment_2)
    # strat from 10008
    write_lammps(ele, embed_density_hh, nrho,
                 nr, dr, drmax, comment_1, comment_2)
    # strat from 12008
    write_lammps(ele, pair_pot_fefe_lammps,
                 nrho, nr, dr, drmax, comment_1, comment_2)
    # strat from 14008
    write_lammps(ele, pair_pot_feh_lammps,
                 nrho, nr, dr, drmax, comment_1, comment_2)
    # strat from 16008
    write_lammps(ele, pair_pot_hh_lammps,
                 nrho, nr, dr, drmax, comment_1, comment_2)
    # +-+-+-+-+-+ +-+-+-+ +-+-+-+-+-+-+
    # |w|r|i|t|e| |p|l|t| |f|o|r|m|a|t|
    # +-+-+-+-+-+ +-+-+-+ +-+-+-+-+-+-+
    write_plt(ele[0], rho, r, embed_energy_fefe,
              embed_density_fefe, pair_pot_fefe)
    write_plt(ele[1], rho, r, embed_energy_hh,
              embed_density_hh, pair_pot_hh)
    write_plt(ele[0] + ele[1], rho, r, [0],
              embed_density_feh, pair_pot_feh)
    write_plt(ele[1] + ele[0], rho, r, [0],
              embed_density_hfe, [0])
    # plt.plot(r, pair_pot_hh)
    # plt.show()
