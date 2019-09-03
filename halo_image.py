from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
import cmocean
import matplotlib
import astropy.io.fits as fits
from astropy.coordinates.matrix_utilities import rotation_matrix
from astropy.coordinates import CartesianRepresentation
from astropy import units as u
from astropy.stats import biweight_scale
import scipy.stats as st
import scipy.integrate
from scipy.integrate import cumtrapz

ngID = 12595208726
gID = 12578248056

####################

hdu_parent = fits.open('/2/home/idroberts/mdpl2/md100000_parent.fits')
hdu_ga = fits.open('/2/home/idroberts/mdpl2/md100000_ga_N10.fits')
hdu_ADp = fits.open('/2/home/idroberts/mdpl2/md100000_upid_ADp_H0_randProj.fits')

data_parent = hdu_parent[1].data
data_ga = hdu_ga[1].data
data_ADp = hdu_ADp[1].data

pvalG = data_ADp[np.where(data_ADp['upid']==gID)[0]]['pvalAD']
pvalNG = data_ADp[np.where(data_ADp['upid']==ngID)[0]]['pvalAD']

print(pvalG)
print(pvalNG)

indNG = np.where(data_parent['col1_x']==ngID)
indG = np.where(data_parent['col1_x']==gID)

gaNG = np.where(data_ga['col6']==ngID)
gaG = np.where(data_ga['col6']==gID)

##########

thetax_ng = data_ADp['thetax'][np.where(data_ADp['upid'] == ngID)[0]]
thetay_ng = data_ADp['thetay'][np.where(data_ADp['upid'] == ngID)[0]]
thetaz_ng = data_ADp['thetaz'][np.where(data_ADp['upid'] == ngID)[0]]

rotx_ng = rotation_matrix(thetax_ng * u.deg, axis='x')
roty_ng = rotation_matrix(thetay_ng * u.deg, axis='y')
rotz_ng = rotation_matrix(thetaz_ng * u.deg, axis='z')

vec_ng = CartesianRepresentation(np.array(data_ga['col17'][gaNG])*u.mpc, np.array(data_ga['col18'][gaNG])*u.mpc, np.array(data_ga['col19'][gaNG])*u.mpc)
vec_v_ng = CartesianRepresentation(np.array(data_ga['col20'][gaNG])*u.km/u.s, np.array(data_ga['col21'][gaNG])*u.km/u.s, np.array(data_ga['col22'][gaNG])*u.km/u.s)

vec_par_ng = CartesianRepresentation(np.array(data_parent['col17'][indNG])*u.mpc, np.array(data_parent['col18'][indNG])*u.mpc, np.array(data_parent['col19'][indNG])*u.mpc)
vec_par_v_ng = CartesianRepresentation(np.array(data_parent['col20'][indNG])*u.km/u.s, np.array(data_parent['col21'][indNG])*u.km/u.s, np.array(data_parent['col22'][indNG])*u.km/u.s)

vec_x_ng = vec_ng.transform(rotx_ng)
vec_xy_ng = vec_x_ng.transform(roty_ng)
vec_xyz_ng = vec_xy_ng.transform(rotz_ng)

vec_par_x_ng = vec_par_ng.transform(rotx_ng)
vec_par_xy_ng = vec_par_x_ng.transform(roty_ng)
vec_par_xyz_ng = vec_par_xy_ng.transform(rotz_ng)

vec_v_x_ng = vec_v_ng.transform(rotx_ng)
vec_v_xy_ng = vec_v_x_ng.transform(roty_ng)
vec_v_xyz_ng = vec_v_xy_ng.transform(rotz_ng)

vec_par_v_x_ng = vec_par_v_ng.transform(rotx_ng)
vec_par_v_xy_ng = vec_par_v_x_ng.transform(roty_ng)
vec_par_v_xyz_ng = vec_par_v_xy_ng.transform(rotz_ng)

newx_ng = vec_xyz_ng.xyz.value[0,:]
newy_ng = vec_xyz_ng.xyz.value[1,:]
newz_ng = vec_xyz_ng.xyz.value[2,:]

newx_par_ng = vec_par_xyz_ng.xyz.value[0,:]
newy_par_ng = vec_par_xyz_ng.xyz.value[1,:]
newz_par_ng = vec_par_xyz_ng.xyz.value[2,:]

newvx_ng = vec_v_xyz_ng.xyz.value[0,:]
newvy_ng = vec_v_xyz_ng.xyz.value[1,:]
newvz_ng = vec_v_xyz_ng.xyz.value[2,:]

newvx_par_ng = vec_par_v_xyz_ng.xyz.value[0,:]
newvy_par_ng = vec_par_v_xyz_ng.xyz.value[1,:]
newvz_par_ng = vec_par_v_xyz_ng.xyz.value[2,:]

#

thetax_g = data_ADp['thetax'][np.where(data_ADp['upid'] == gID)[0]]
thetay_g = data_ADp['thetay'][np.where(data_ADp['upid'] == gID)[0]]
thetaz_g = data_ADp['thetaz'][np.where(data_ADp['upid'] == gID)[0]]

rotx_g = rotation_matrix(thetax_g * u.deg, axis='x')
roty_g = rotation_matrix(thetay_g * u.deg, axis='y')
rotz_g = rotation_matrix(thetaz_g * u.deg, axis='z')

vec_g = CartesianRepresentation(np.array(data_ga['col17'][gaG])*u.mpc, np.array(data_ga['col18'][gaG])*u.mpc, np.array(data_ga['col19'][gaG])*u.mpc)
vec_v_g = CartesianRepresentation(np.array(data_ga['col20'][gaG])*u.km/u.s, np.array(data_ga['col21'][gaG])*u.km/u.s, np.array(data_ga['col22'][gaG])*u.km/u.s)

vec_par_g = CartesianRepresentation(np.array(data_parent['col17'][indG])*u.mpc, np.array(data_parent['col18'][indG])*u.mpc, np.array(data_parent['col19'][indG])*u.mpc)
vec_par_v_g = CartesianRepresentation(np.array(data_parent['col20'][indG])*u.km/u.s, np.array(data_parent['col21'][indG])*u.km/u.s, np.array(data_parent['col22'][indG])*u.km/u.s)

vec_x_g = vec_g.transform(rotx_g)
vec_xy_g = vec_x_g.transform(roty_g)
vec_xyz_g = vec_xy_g.transform(rotz_g)

vec_par_x_g = vec_par_g.transform(rotx_g)
vec_par_xy_g = vec_par_x_g.transform(roty_g)
vec_par_xyz_g = vec_par_xy_g.transform(rotz_g)

vec_v_x_g = vec_v_g.transform(rotx_g)
vec_v_xy_g = vec_v_x_g.transform(roty_g)
vec_v_xyz_g = vec_v_xy_g.transform(rotz_g)

vec_par_v_x_g = vec_par_v_g.transform(rotx_g)
vec_par_v_xy_g = vec_par_v_x_g.transform(roty_g)
vec_par_v_xyz_g = vec_par_v_xy_g.transform(rotz_g)

newx_g = vec_xyz_g.xyz.value[0,:]
newy_g = vec_xyz_g.xyz.value[1,:]
newz_g = vec_xyz_g.xyz.value[2,:]

newx_par_g = vec_par_xyz_g.xyz.value[0,:]
newy_par_g = vec_par_xyz_g.xyz.value[1,:]
newz_par_g = vec_par_xyz_g.xyz.value[2,:]

newvx_g = vec_v_xyz_g.xyz.value[0,:]
newvy_g = vec_v_xyz_g.xyz.value[1,:]
newvz_g = vec_v_xyz_g.xyz.value[2,:]

newvx_par_g = vec_par_v_xyz_g.xyz.value[0,:]
newvy_par_g = vec_par_v_xyz_g.xyz.value[1,:]
newvz_par_g = vec_par_v_xyz_g.xyz.value[2,:]

##########

print('Mvir, NG: ', data_parent['col10'][indNG], 'Mvir, G: ',data_parent['col10'] [indG])
print('Last MM scale, NG: ', data_parent['col15'][indNG], 'Last MM scale, G: ', data_parent['col15'][indG])

cNG = Circle((0, 0), radius=data_parent['col11'][indNG]/1000., fc='none', ec='#666666', ls='--')
cG = Circle((0, 0), radius=data_parent['col11'][indG]/1000., fc='none', ec='#666666', ls='--')

cNG1 = Circle((0, 0), radius=data_parent['col11']/1000., fc='none', ec='#666666', ls='--')
cG1 = Circle((0, 0), radius=data_parent['col11']/1000., fc='none', ec='#666666', ls='--')

patchesNG = []
patchesG = []

for xNG, yNG, rNG in zip(newx_ng, newy_ng, data_ga['col11'][gaNG]):
    circ = Circle((xNG-newx_par_ng,yNG-newy_par_ng), 1.*rNG/1000.)
    patchesNG.append(circ)

for xG, yG, rG in zip(newx_g, newy_g, data_ga['col11'][gaG]):
    circ = Circle((xG-newx_par_g,yG-newy_par_g), 1.*rG/1000.)
    patchesG.append(circ)

##################

green = '#33825e'
purple = '#8f5e99'

normalize = matplotlib.colors.Normalize(vmin=-1500, vmax=1500)

fig = plt.figure()

grid = gs.GridSpec(2,2)
grid.update(wspace=0.2)

ax1 = fig.add_subplot(grid[0,0])

pad = 0.2

ax1.set_xlim(-data_parent['col11'][indNG]/1000.-pad, data_parent['col11'][indNG]/1000.+pad)
ax1.set_ylim(-data_parent['col11'][indNG]/1000.-pad, data_parent['col11'][indNG]/1000.+pad)
ax1.set_xlabel('$\Delta x \quad (\mathrm{Mpc})$', fontsize=9)
ax1.set_ylabel('$\Delta y \quad (\mathrm{Mpc})$', fontsize=9)
ax1.set_xticks(np.linspace(-1,1,3))
ax1.set_yticks(np.linspace(-1,1,3))
ax1.tick_params(labelsize=8)

ax1.add_patch(cNG)

colorsNG = newvz_ng - np.average(newvz_ng)
pNG = PatchCollection(patchesNG, cmap=cmocean.cm.balance_r, norm=normalize, edgecolors='#666666', linewidths=0.5, alpha=0.3)
pNG.set_array(np.array(colorsNG))
ax1.add_collection(pNG)

cbarNG = fig.colorbar(pNG, ax=ax1)
cbarNG.ax.tick_params(labelsize=6)
cbarNG.ax.set_title('$\Delta v_\mathrm{los}\,(\mathrm{km/s})$', fontsize=7)

#ax1.scatter(x[gaNG]-X[indNG], y[gaNG]-Y[indNG], s=r[gaNG]/10., linewidths=0, alpha=0.5)

ax1.set_aspect('equal')

ax1.text(0.02, 0.93, 'NG$_{1D}$', fontsize=8, fontname='stix', family='sans-serif', transform=ax1.transAxes)

ax1.text(0.93, 0.93, 'a.', fontsize=8, fontname='stix', family='sans-serif', transform=ax1.transAxes)

ax2 = fig.add_subplot(grid[0,1])

ax2.set_xlim(-data_parent['col11'][indG]/1000.-pad, data_parent['col11'][indG]/1000.+pad)
ax2.set_ylim(-data_parent['col11'][indG]/1000.-pad, data_parent['col11'][indG]/1000.+pad)
ax2.set_xlabel('$\Delta x \quad (\mathrm{Mpc})$', fontsize=9)
ax2.set_ylabel('$\Delta y \quad (\mathrm{Mpc})$', fontsize=9)
ax2.set_xticks(np.linspace(-1,1,3))
ax2.set_yticks(np.linspace(-1,1,3))
ax2.tick_params(labelsize=8)

ax2.add_patch(cG)

colorsG = newvz_g - np.average(newvz_g)
pG = PatchCollection(patchesG, cmap=cmocean.cm.balance_r, norm=normalize, edgecolors='#666666', linewidths=0.5, alpha=0.3)
pG.set_array(np.array(colorsG))
ax2.add_collection(pG)

cbarG = fig.colorbar(pG, ax=ax2)
cbarG.ax.tick_params(labelsize=6)
cbarG.ax.set_title('$\Delta v_\mathrm{los}\,(\mathrm{km/s})$', fontsize=7)

#ax2.scatter(x[gaG]-X[indG], y[gaG]-Y[indG], s=r[gaG]/10., linewidths=0, alpha=0.5)

ax2.set_aspect('equal')

ax2.text(0.02, 0.93, 'G$_{1D}$', fontsize=8, fontname='stix', family='sans-serif', transform=ax2.transAxes)
ax2.text(0.9, 0.93, 'b.', fontsize=8, fontname='stix', family='sans-serif', transform=ax2.transAxes)

fig.savefig('/2/home/idroberts/mdpl2/plots/halo_image.png', dpi=800, bbox_inches='tight')

######

sigNG = biweight_scale(newvz_ng)
sigG = biweight_scale(newvz_g)

#kdeNG = st.gaussian_kde((newvz_ng - newvz_par_ng)/sigNG)
kdeNG = st.gaussian_kde((newvz_ng - np.average(newvz_ng))/sigNG)
kdeG = st.gaussian_kde((newvz_g - np.average(newvz_g))/sigG)

x = np.linspace(-3.8, 3.8, 500)

cdfNG = cumtrapz(kdeNG(x), x=x, initial=0)
cdfG = cumtrapz(kdeG(x), x=x, initial=0)

fig = plt.figure()

grid = gs.GridSpec(2,2)
grid.update(wspace=0.2)

ax1 = fig.add_subplot(grid[0,0])

ax1.set_xlim(-3.5,3.5)
ax1.set_xticks(np.linspace(-3,3,7))
ax1.set_xlabel(r'$\Delta v / \sigma$', fontsize=9)
ax1.set_ylabel('Density', fontsize=9)
ax1.tick_params(labelsize=8)

ax1.plot(x, st.norm.pdf(x, 0, 1), color="k", lw=0.75, ls="--", alpha=0.75)

ax1.plot(x, kdeNG(x), color=purple, ls="-", lw=1)

ax1.plot(x, kdeG(x), color=green, ls="-", lw=1)

ax1.text(0.1, 0.9, '$G_{1D}$', fontsize=8, color=green, transform=ax1.transAxes)
ax1.text(0.1, 0.8, '$NG_{1D}$', fontsize=8, color=purple, transform=ax1.transAxes)

ax1.text(0.68, 0.9, '$p_{{AD}} = {:.2g}$'.format(pvalG[0]), fontsize=8, color=green, transform=ax1.transAxes)
ax1.text(0.68, 0.8, '$p_{{AD}} = {:.2g}$'.format(pvalNG[0]), fontsize=8, color=purple, transform=ax1.transAxes)

#

#ax2 = fig.add_subplot(grid[0,1])

#ax2.set_xlim(-3.5,3.5)
#ax2.set_xticks(np.linspace(-3,3,7))
#ax2.tick_params(labelsize=8)

#ax2.plot(x, st.norm.cdf(x, 0, 1), color="k", lw=0.75, ls="--", alpha=0.75)

#ax2.plot(x, cdfNG, color=purple, ls="-", lw=1)

#ax2.plot(x, cdfG, color=green, ls="-", lw=1)

fig.savefig('/2/home/idroberts/mdpl2/plots/sample_vdist.pdf', dpi=800, bbox_inches='tight')
