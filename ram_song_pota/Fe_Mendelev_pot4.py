#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2014 - Swang <swangi@outlook.com>
# Filename:
# +-+-+-+-+-+-+ +-+-+-+-+-+-+-+-+-+
# |I|m|p|o|r|t| |s|o|m|e|t|h|i|n|g|
# +-+-+-+-+-+-+ +-+-+-+-+-+-+-+-+-+
#
import numpy as np
from EAM_AM import pairwise_e
from EAM_AM import embed_d
from EAM_AM import embed_e
from EAM_AM import write_plt
# import pylab as plt
from time import ctime

# +-+-+-+-+-+-+-+-+-+-+
# |P|a|r|a|m|e|t|e|r|s|
# +-+-+-+-+-+-+-+-+-+-+
pot_sourse = 'Potential #4 from [Phil. Mag. A, 83, 3977-3994 (2003).]'
pot_name = 'Mendelev_pot4'
aphik_men = [
    195.92322853994, 17.516698453315, 1.4926525164290,
    6.4129476125197, -6.8157461860553, 9.6582581963600,
    -5.3419002764419, 1.7996558048346, -1.4788966636288,
    1.8530435283665, -0.64164344859316, 0.24463630025168,
    -0.057721650527383, 0.023358616514826, -0.0097064921265079]
rphik_men = [2.1, 2.2, 2.3,
             2.4, 2.5, 2.6,
             2.7, 2.8, 3.0,
             3.3, 3.7, 4.2,
             4.7, 5.3, 6.0]
r1_men = 0.9
r2_men = 1.95
Bn_men = [14.996917289290,
          -20.533174190155,
          14.002591780752,
          -3.6473736591143]
arhok_men = [11.686859407970, -0.014710740098830, 0.47193527075943]
rrhok_men = [2.4, 3.2, 4.2]
pairwise_e = pairwise_e(r1=r1_men, r2=r2_men,
                        aphik=aphik_men, rphik=rphik_men, Bn=Bn_men)
embed_d = embed_d(arhok=arhok_men, rrhok=rrhok_men)
embed_e = embed_e(ap=-0.00034906178363530)

dr = 0.00060000
drho = 0.0300000
nr = 10000
nrho = 10000
drmax = dr * nr
drhomax = drho * nrho
r = np.arange(0, drmax, dr)
rho = np.arange(0, drhomax, drho)

pair_pot = [0] * len(r)
embed_density = [0] * len(r)
embed_energy = [0] * len(rho)
for i in range(len(r)):
    embed_density[i] = embed_d.psi(r[i])
    pair_pot[i] = pairwise_e.phi(r[i])
for j in range(len(rho)):
    embed_energy[j] = embed_e.F(rho[j])


pair_pot_lammps = [0] * len(r)
for i in range(len(r)):
    pair_pot_lammps[i] = pairwise_e.phi_lammps(r[i])
#
# +-+-+-+-+-+ +-+-+ +-+-+-+-+
# |S|t|a|r|t| |o|f| |m|a|i|n|
# +-+-+-+-+-+ +-+-+ +-+-+-+-+
#
#
if __name__ == "__main__":
    write_plt(rho, r, embed_energy, embed_density, pair_pot)
    f = open('%s.txt' % (pot_name), 'w')
    f.write(
        'Sourse: ' + pot_sourse + '\n' +
        'Made by Swang using python ' + ctime() + '\n' +
        '<swangi@outlook.com>\n')
    f.close
    f = open('%s.txt' % (pot_name), 'a')
    f.write('1  Fe \n')
    f.write('%d %.15E %d %.15E %.15E\n' % (nrho, drho, nr, dr, drmax))
    f.write('26 5.58500000000000E+0001 2.8557E+0000 bcc\n')
    i = 0
    for x in embed_energy:
        f.write('%.15E ' % (x))
        i += 1
        if i % 5 == 0:
            f.write('\n')
    i = 0
    for x in embed_density:
        f.write('%.15E ' % (x))
        i += 1
        if i % 5 == 0:
            f.write('\n')
    i = 0
    for x in pair_pot_lammps:
        f.write('%.15E ' % (x))
        i += 1
        if i % 5 == 0:
            f.write('\n')
    f.close
    plt.plot(r, pair_pot_lammps)
    # plt.show()
