from __future__ import print_function
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
from tqdm import trange
import scipy.stats.distributions as dist
from matplotlib.patches import Circle
from matplotlib.patches import FancyArrow
import matplotlib
import matplotlib.image as mpimg
from astropy.modeling import models, fitting
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import rcParams
import scipy.stats as st

##################

def king(r, n0, rc, b):
    """King model for density profile"""
    return n0 * (1. + (r/rc)**2)**b

def king2(r, n0, rc0, b0, n1, rc1, b1):
    """King model for density profile"""
    return n0 * (1. + (r/rc0)**2)**b0 + n1 * (1. + (r/rc1)**2)**b1

def qFrac_w(x, ssfr, vmax, bins, c):
    frac = np.zeros(len(bins)-1)
    xmed = np.zeros(len(bins)-1)
    el = np.zeros(len(bins)-1)
    eh = np.zeros(len(bins)-1)
    xerr = np.zeros(len(bins)-1)
    for i in range(len(bins)-1):
        ind = np.where((x>=bins[i]) & (x<bins[i+1]))
        x_i = x[ind]
        ssfr_i = ssfr[ind]
        vmax_i = vmax[ind]

        num = np.sum(1 / vmax_i[ssfr_i < -11])
        denom = np.sum(1 / vmax_i)
        frac[i] = num / denom
        xmed[i] = np.median(x_i)

        k = int(frac[i] * len(x_i))
        N = len(x_i)
        el[i] = frac[i] - dist.beta.ppf((1-c)/2., k+1, N-k+1)
        eh[i] = dist.beta.ppf(1-(1-c)/2., k+1, N-k+1) - frac[i]

    return xmed, frac, el, eh

###############################

# Speeds: 1.8,2.8 - 0.9,2.1 - 2.9,3.5

grpID, z_cluster, mh, ra, dec, z, r500, ra0, dec0, logM, vmax, dens, vdisp, sfr  = np.loadtxt('gal_data5000_lowz_Mstar9.dat', usecols=(0,4,5,6,7,8,16,17,18,19,33,34,35,36), unpack=True)

vout = np.loadtxt('/2/home/idroberts/xray/gal_env/rcrit_25/infall_velocity_distributions_outer_monotonic_decrease_3d.dat', usecols=(4,), unpack=True)

vin = np.loadtxt('/2/home/idroberts/xray/gal_env/rcrit_25/infall_velocity_distributions_inner_monotonic_decrease_3d.dat', usecols=(4,), unpack=True)

ssfr = sfr - logM

IDs, u = np.unique(grpID, return_index=True)

#vin, vout = 2.8, 1.8

fQ200, fQthresh, fQsub = 0.43, 0.51, 0.35

vdisp_Y07 = 397.9e5 * ((0.7 * mh[u]) / 1e14) ** 0.3214

mproton = 1.67e-24 # grams

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
scale = cosmo.kpc_proper_per_arcmin(z)

c = 3e10 # cm/s

kpc2cm = 3.086e21 # cm kpc^-1
Gyr2s = 3.15e16 # s Gyr^-1

Nit = 1000

tQslowMC = np.zeros(Nit)
tQrapidMC = np.zeros(Nit)
tdelayMC = np.zeros(Nit)

for it in trange(Nit):

    tQslow = np.zeros(len(IDs))
    tQrapid = np.zeros(len(IDs))
    tdelay = np.zeros(len(IDs))
    Rcrit1 = np.zeros(len(IDs))
    Rcrit2 = np.zeros(len(IDs))
    rho_vir = np.zeros(len(IDs))

    ID2 = np.array([5, 11, 23, 24, 32, 57, 145, 191, 2360])

    voutMC = np.random.choice(vout)
    vinMC = np.random.choice(vin)

    for i in range(len(IDs)):
    
        r = np.linspace(0, 5 * r500[u][i], 500)

        if len(np.where(IDs[i] == ID2)[0]) > 0:
            n00, rc00, b00, n01, rc01, b01 = np.loadtxt('/2/home/idroberts/xray/gal_env/data/y%d/y%d_fit_apec2.dat' % (grpID[i],grpID[i]), unpack=True)
            n = king2(r, n00, rc00, b00, n01, rc01, b01)
        else:
            n0, rc, b = np.loadtxt('/2/home/idroberts/xray/gal_env/data/y%d/y%d_fit_apec.dat' % (IDs[i],IDs[i]), unpack=True)
            n = king(r, n0, rc, b)
        rho = n * mproton # g /cm3
        k1 = np.argmax(np.log10(rho) < -28.3)
        k2 = np.argmax(np.log10(rho) < -27)
        kvir = np.argmax(r > 2.7*r500[u][i])
        Rcrit1[i] = r[k1]
        Rcrit2[i] = r[k2]
        rho_vir[i] = rho[kvir]
        dr1 = (2.7*r500[u][i] - Rcrit1[i]) * scale[u][i]
        if dr1 < 0:
            continue
        dr2 = (Rcrit1[i] - Rcrit2[i]) * scale[u][i]
        dr1_cgs = dr1.value * kpc2cm
        dr2_cgs = dr2.value * kpc2cm
    
        v1_cgs = voutMC * (vdisp[u][i] / (1 + z_cluster[u][i]))
        v2_cgs = vinMC * (vdisp[u][i] / (1 + z_cluster[u][i]))
    
        tslow = (dr1_cgs / v1_cgs) / Gyr2s # Gyr
        trapid = (dr2_cgs / v2_cgs) / Gyr2s # Gyr

        tQslow[i] = tslow * (1. / np.log((1 - fQ200) / (1 - fQthresh)))
        tQrapid[i] = trapid * (1. / np.log((1 - 0.) / (1 - fQsub)))
        tdelay[i] = tslow

    tQslowMC[it] = np.median(tQslow)
    tQrapidMC[it] = np.median(tQrapid)
    tdelayMC[it] = np.median(tdelay)

#print('rho_vir:', np.median(np.log10(rho_vir)), np.std(np.log10(rho_vir)))

print('tQslow:', np.percentile(tQslowMC, (50,16,84)))

print('tQrapid:', np.percentile(tQrapidMC, (50,16,84)))

print('tdelay:', np.percentile(tdelayMC, (50,16,84)))

#print('Critical radius: ', np.median(Rcrit1 / r500[u]), 'R500')

###################################

m1 = np.where(logM < 10)

dens_q, frac_q, el_q, eh_q = np.loadtxt('/2/home/idroberts/xray/gal_env/qFrac_dens_m1.dat', unpack=True)
e_q = np.average((el_q,eh_q), axis=0)

mf, vmax_f, sfr_f = np.loadtxt('/2/home/idroberts/xray/gal_env/field_z10.dat', usecols=(4,7,8), unpack=True)

ssfr_f = sfr_f - mf

bins_f = np.array([9.0, 10.0, 10.5, 12.0])

mf_bin, fracf, elf, ehf = qFrac_w(mf, ssfr_f, np.ones(len(vmax_f)), bins_f, 0.997)

coral = '#cd5b45'
gray = '#555555'
purple = '#5d478b'
red = '#ee2c2c'
blue = '#1874CD'

fig = plt.figure()

grid = gs.GridSpec(2,2)

fig.subplots_adjust(wspace=0.05)

ax1 = fig.add_subplot(grid[0,0])

xband = np.logspace(-32, -27, 500)
crit_dens = np.zeros(20)
#for i in range(6,26):
    #bins = np.linspace(-31.08, -27, i)
    #dens_bin, frac, el, eh = qFrac_w(np.log10(dens[m1]), ssfr[m1], vmax[m1], bins, 0.68)
    #pl_init = models.PowerLaw1D(10**0.45, 10**-29, -0.2)
    #pl_fit = fitting.LevMarLSQFitter()
    #pl = pl_fit(pl_init, 10**dens_bin, 10**frac)
    
    #pl2_init = models.BrokenPowerLaw1D(10**0.45, 10**-29, 0., -0.2)
    #pl2_fit = fitting.LevMarLSQFitter()
    #pl2 = pl2_fit(pl2_init, 10**dens_bin, 10**frac)

    #crit_dens[i-6] = pl2.x_break.value
    
    #ax1.plot(np.log10(xband), np.log10(pl2(xband)), color=purple, ls='-', lw=0.75, alpha=0.25)

pl2_init = models.BrokenPowerLaw1D(10**0.45, 10**-29, 0., -0.2)
pl2_fit = fitting.LevMarLSQFitter()
pl2 = pl2_fit(pl2_init, 10**dens_q, 10**frac_q)

my_cmap = LinearSegmentedColormap.from_list('my_cmap', ['#ffffff','#5d478b'], N=250)

RedBlue = LinearSegmentedColormap.from_list('RedBlue', [red,blue], N=250)

def gauss_smear(x0, y0, err, dx, dy):
    x, y = np.meshgrid(np.linspace(x0-dx/2., x0+dx/2., 50), np.linspace(y0-dy/2., y0+dy/2., 100))
    return x, y, np.exp(-(y0-y)**2 / err**2)

for i in range(len(dens_q)):
    X, Y, Z = gauss_smear(dens_q[i], frac_q[i], e_q[i], 0.1, 0.3)
    ax1.pcolor(X, Y, Z, cmap=my_cmap, alpha=1.0)

ax1.plot(np.log10(xband), np.log10(pl2(xband)), color=purple, lw=1.5)

#ax1.plot(dens_q, frac_q, color=purple, ls='none', marker='v', mec='k', mew=0.75, ms=5)

#ax1.errorbar(dens_q, frac_q, yerr=(el_q,eh_q), color=purple, mec='k', marker='v', lw=0, capsize=0, mew=0.6, ms=5, elinewidth=0.8)

ax1.set_ylim([0.0,1.03])
ax1.set_xlim([-31,-26.7])
ax1.set_xlabel(r'$\log\;\rho_\mathrm{ICM}$', fontsize=11)
ax1.set_yticks([])
ax1.set_xticks([np.log10(pl2.x_break.value)])
ax1.set_xticklabels([r'$\log\,\rho_\mathrm{thresh}$'])
ax1.set_ylabel(r'Quenched fraction', fontsize=10, labelpad=10, fontname='stix', family='sans-serif')
ax1.tick_params(which='both', labelsize=8)

ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.spines['left'].set_visible(False)

ax1.annotate("", xy=(-26.7,0.0), xytext=(-31,0.0), arrowprops=dict(width=0.1, headwidth=5, headlength=6, shrink=0.0, color='k'))
ax1.annotate("", xy=(-31,1.03), xytext=(-31,0.0), arrowprops=dict(width=0.1, headwidth=5, headlength=6, shrink=0.0, color='k'))

ax1.annotate("", xy=(np.log10(pl2.x_break.value),0.64), xytext=(-30.8,0.56), arrowprops=dict(arrowstyle="->"))

ax1.annotate("", xy=(-27.1,1.02), xytext=(np.log10(pl2.x_break.value),0.64), arrowprops=dict(arrowstyle="->"))

ax1.annotate("", xy=(-30.9,0.35), xytext=(-30.9,0.08), arrowprops=dict(arrowstyle="->"))

ax1.plot([np.log10(pl2.x_break.value),np.log10(pl2.x_break.value)], [0,np.log10(pl2.amplitude.value)], color='k', lw=0.8, ls='--')

ax1.plot([-31,-26.7], [fracf[0],fracf[0]], color=purple, ls='-', lw=5, alpha=0.4)

ax1.text(0.05, 0.9, 'a.', fontsize=8, weight='black', fontname='stix', family='sans-serif', transform=ax1.transAxes)

ax1.text(0.75, 0.023, 'Field', color='k', fontsize=7, fontname='stix', family='sans-serif', bbox=dict(facecolor='w', lw=0, pad=0.5), transform=ax1.transAxes)

ax1.text(0.06, 0.15, 'Pre-\n processing?', color='k', fontsize=7, fontname='stix', family='sans-serif', transform=ax1.transAxes)

ax1.text(0.05, 0.75, r'$t_\mathrm{slow}\!:\,1.5 - 2.5\,\mathrm{Gyr}$', fontsize=8, transform=ax1.transAxes)

#ax1.text(0.05, 0.65, r'$\tau_{Q,\mathrm{slow}}\!:\,4 - 8\,\mathrm{Gyr}$', fontsize=7, transform=ax1.transAxes)

ax1.text(0.4, 0.9, r'$\tau_{Q,\mathrm{rapid}}\!:\, 0.5-1\,\mathrm{Gyr}$', fontsize=8, transform=ax1.transAxes)

ax1.text(0.1, 0.58, 'Slow', fontsize=7, fontname='stix', family='sans-serif', transform=ax1.transAxes, rotation=4)

ax1.text(0.62, 0.775, 'Rapid', fontsize=7, fontname='stix', family='sans-serif', transform=ax1.transAxes, rotation=42)

ax2 = fig.add_subplot(grid[0,1])

cmap = matplotlib.cm.get_cmap('coolwarm_r')
#img = mpimg.imread('/2/home/idroberts/xray/gal_env/galaxy-clipart.pdf')

ax2.set_xlim(-1.2,1.4)
ax2.set_ylim(-1.05,1.05)
ax2.set_xticks([])
ax2.set_yticks([])

ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.spines['left'].set_visible(False)

x = np.linspace(-1,1,100)
y = np.linspace(-1,1,100)
X, Y = np.meshgrid(x, y)

def func(x, y):
    #r = np.sqrt(x**2 + y**2)
    r = (x**2 + y**2)**1.1
    return 1. - r

Z = func(X, Y)   

t = np.linspace(-1, 0.03, 100)

p = np.polyfit([-1,-0.5,0], [-0.5,0.05,0.23], 2)
rquad = np.sqrt(t**2 + (p[0]*t**2+p[1]*t+p[2])**2)

im = ax2.imshow(Z, cmap='binary', vmin=0.0, extent=(-1,1,-1,1))
#im1 = ax2.imshow(img, extent=(-1.2,-0.9,-0.4,-0.1))

ax2.scatter(t, p[0]*t**2+p[1]*t+p[2], s=3.5, c=rquad, marker='.', cmap=RedBlue, vmax=0.5, zorder=5)

ax2.annotate("", xy=(0.15,0.23), xytext=(0.0,0.23), arrowprops=dict(arrowstyle="simple", fc=red, ec='none'))
ax2.annotate("", xy=(-0.85,-0.3), xytext=(-0.86,-0.312), arrowprops=dict(arrowstyle="simple", fc=blue, ec='none'))
ax2.annotate("", xy=(-0.50,0.0565), xytext=(-0.51,0.0475), arrowprops=dict(arrowstyle="simple", fc=blue, ec='none'))

c1 = Circle(xy=(0,0), radius=1, edgecolor='k', facecolor='none', ls='--', lw=1.5, alpha=0.75)
c2 = Circle(xy=(0,0), radius=0.37, edgecolor=red, facecolor='none', ls='--', lw=1.5, alpha=0.75)
#c3 = Circle(xy=(1.15,0.55), radius=0.25, edgecolor='k', facecolor='none', ls='--', lw=1, alpha=0.75)

ax2.add_patch(c1)
ax2.add_patch(c2)
#ax2.add_patch(c3)

ax2.text(-0.95, -0.95, r'$\mathbf{R_\mathrm{\mathbf{vir}}}$', color='k', fontsize=10)

ax2.text(0.6, 0.85, 'Pre- \n processing', color='k', fontsize=7, fontname='stix', family='sans-serif', weight='black')

ax2.text(-0.6, 0.45, 'Slow', color='k', fontsize=7, fontname='stix', family='sans-serif', weight='black', ha='center')
#ax2.text(-0.65, 0.27, r'$\tau_{Q,\mathrm{slow}}$', color='k', fontsize=9, ha='center')

ax2.text(0, -0.05, 'Rapid', color=red, fontsize=7, ha='center', fontname='stix', family='sans-serif', weight='black')
#ax2.text(0, -0.2, r'$\tau_{Q,\mathrm{rapid}}$', color=red, fontsize=9, ha='center')

bold_math = {
    "mathtext.fontset": "stixsans",
    }

rcParams.update(bold_math)

ax2.text(0, -0.5, r'$\mathbf{R\,(\rho_\mathrm{\mathbf{thresh}})}$', color=red, fontsize=8.5, ha='center')

ax2.text(0.05, 0.9, 'b.', fontsize=8, weight='black', fontname='stix', family='sans-serif', transform=ax2.transAxes)

fig.savefig('/2/home/idroberts/xray/gal_env/plots/schematic.png', dpi=800, bbox_inches='tight')

#################

kern_in = st.gaussian_kde(vin)
kern_out = st.gaussian_kde(vout)

vdist = np.linspace(0,4.5,500)

fig = plt.figure()

ax1 = fig.add_subplot(grid[0,0])

ax1.set_xlim(0,4.4)
ax1.set_xlabel('$v_{1D} \, / \sigma_{1D}$', fontsize=9)
ax1.set_ylabel('Prob. density', fontsize=9)
ax1.tick_params(labelsize=7)

ax1.plot(vdist, kern_out(vdist), color='#8C1717', ls='-', lw=1)
ax1.plot(vdist, kern_in(vdist), 'k--', lw=1)

ax1.text(0.05, 0.9, '$R < R_\mathrm{thresh}$', fontsize=8, color='k', transform=ax1.transAxes)
ax1.text(0.05, 0.8, '$R > R_\mathrm{thresh}$', fontsize=8, color='#8C1717', transform=ax1.transAxes)


fig.savefig('/2/home/idroberts/xray/gal_env/plots/vdists.png', dpi=800, bbox_inches='tight')
