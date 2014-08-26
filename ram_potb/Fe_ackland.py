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
import pylab as plt
from time import ctime

# +-+-+-+-+-+-+-+-+-+-+
# |P|a|r|a|m|e|t|e|r|s|
# +-+-+-+-+-+-+-+-+-+-+
pot_sourse = 'Potential Fe-Fe from [Journal of Physics: Condensed Matter, 16(27), S2629–S2642]'
pot_name = 'Ackland_Pot'
aphik_ack = [
    -27.444805994228, 15.738054058489,
    2.2077118733936, -2.4989799053251, 4.2099676494795,
    -0.77361294129713, 0.80656414937789, -2.3194358924605,
    2.6577406128280, -1.0260416933564, 0.35018615891957,
    -0.058531821042271, -0.0030458824556234]
rphik_ack = [2.2, 2.3,
             2.4, 2.5, 2.6,
             2.7, 2.8, 3.0,
             3.3, 3.7, 4.2,
             4.7, 5.3]
r1_ack = 1.00
r2_ack = 2.05
Bn_ack = [7.4122709384068,
          -0.64180690713367,
          -2.6043547961722,
          0.6262539393123]
arhok_ack = [11.686859407970, -0.014710740098830, 0.47193527075943]
rrhok_ack = [2.4, 3.2, 4.2]
pairwise_e = pairwise_e(r1=r1_ack, r2=r2_ack,
                        aphik=aphik_ack, rphik=rphik_ack, Bn=Bn_ack)
embed_d = embed_d(arhok=arhok_ack, rrhok=rrhok_ack)
embed_e = embed_e(ap=-6.7314115586063 *
                  10 ** (-4), ap_p=7.6514905604792 * 10 ** (-8))

# dr = 0.00060000
drho = 0.0300000
dr = 0.000530000
# drho = 0.0500000
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
    write_plt('Fe', rho, r, embed_energy,
              embed_density, pair_pot)
    f = open('%s.txt' % (pot_name), 'w')
    f.write(
        'Sourse: ' + pot_sourse + '\n' +
        'Made by Swang using python ' + ctime() + '\n' +
        '<swangi@outlook.com>\n')
    f.close
    f = open('%s.txt' % (pot_name), 'a')
    f.write('1  Fe \n')
    f.write('%d %.15E %d %.15E %.15E\n' % (nrho, drho, nr, dr, drmax))
    f.write('26 5.58500000000000E+0001 2.85531200000000E+0000 bcc\n')
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
    plt.show()
