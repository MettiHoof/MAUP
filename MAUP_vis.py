##############################################################################################
# -*- coding: utf-8 -*-
# Script to create visualisations that help explain the MAUP
# The visuals show the effect of aggregating point observations for an imaginary correlation
# Developed for a job interview for a Lecture position at Newcastle University 2019
# Written by Maarten Vanhoof, July 2019, in Shanghai
# Python 2.7
##############################################################################################

print"The script is starting"


########################################################
#0. Setup environment
########################################################

############################
#0.1 Import dependencies
############################
import matplotlib.pyplot as plt  #For interactive plotting
import matplotlib.gridspec as gridspec #For specifying grids in matplotlib figure
import pandas as pd #For data handling
import numpy as np #For fast array handling
import math # For some math expressions

############################
#0.2 Setup in and output paths
############################


########################################################
#1. Create data 
########################################################

x=np.random.random_sample(10)
y=np.random.random_sample(10)

x_coord=np.random.random_sample(10)
y_coord=np.random.random_sample(10)

########################################################
#2. Create visualisation
########################################################

############################
#2.1 Setup figure and subplots using gridspec
############################

#Setting up figure:
figsize_x_inches=13.3#matplotlibs figsize (currently) is in inches only. 
figsize_y_inches=7.5

fig=plt.figure(figsize=(figsize_x_inches,figsize_y_inches))

#Setting up gridSpecs
gs_map=gridspec.GridSpec(3,1) #nrows,ncols
gs_map.update(left=0.02, right=0.33, bottom=0.02, top=0.98, hspace=0.1) #update position of grid in figure

gs_hist=gridspec.GridSpec(3,1) #nrows,ncols
gs_hist.update(left=0.36, right=0.63, bottom=0.02, top=0.98, hspace=0.1) #update position of grid in figure

gs_corr=gridspec.GridSpec(3,1) #nrows,ncols
gs_corr.update(left=0.66, right=0.98, bottom=0.02, top=0.98, hspace=0.1) #update position of grid in figure


#Naming gridblocks
ax_map0=plt.subplot(gs_map[0,0])
ax_map1=plt.subplot(gs_map[1,0])
ax_map2=plt.subplot(gs_map[2,0])

ax_hist0=plt.subplot(gs_hist[0,0])
ax_hist1=plt.subplot(gs_hist[1,0])
ax_hist2=plt.subplot(gs_hist[2,0])


ax_corr0=plt.subplot(gs_corr[0,0])
ax_corr1=plt.subplot(gs_corr[1,0])
ax_corr2=plt.subplot(gs_corr[2,0])


############################
#2.2 Plot figure, insert
############################

#Create maps based on coordinates of data
ax_map0.scatter(x_coord,y_coord)
ax_map0.scatter(x_coord,y_coord)
ax_map0.scatter(x_coord,y_coord)

#Create hist based on input data
ax_hist0.hist(x)
ax_hist1.hist(x)
ax_hist2.hist(x)

#Create corr based on input data
ax_corr0.scatter(x,y)

#Get and plot trendilne
z = np.polyfit(x,y, 1)
p = np.poly1d(z)
xhelp=range(int(math.floor(ax_corr0.get_xlim()[0])),
			int(math.ceil(ax_corr0.get_xlim()[1])))
eq_line=ax_corr0.plot(xhelp, p(xhelp),ls='--',lw=1,color='.5',alpha=1)

############################
#2.3 Make up figures
############################

#Map subplots -we hard code for each subplot here, probably there is a more efficient way
ax_map0.get_xaxis().set_visible(False)
ax_map0.get_yaxis().set_visible(False)

ax_map1.get_xaxis().set_visible(False)
ax_map1.get_yaxis().set_visible(False)

ax_map2.get_xaxis().set_visible(False)
ax_map2.get_yaxis().set_visible(False)



############################
#2.4 Save or show
############################
plt.show()

'''

# the line equation:
equation_text='y = %.2fx+%.2f'%(z[0],z[1])
ax_corr0.text(2, z[0]*2+z[1]+0.3, equation_text, ha='left', va='bottom', rotation=23, fontsize=30, color='grey')


#Calculate and print Pearson correlation coefficient
correlation=np.corrcoef(x,y)[0][1]
correlation_text= 'Pearson = %.3f' %correlation
ax_corr0.text(8, 0.5, correlation_text, ha='left', va='bottom', fontsize=8, color='0.2',
           bbox={'facecolor':'white', 'alpha':0.9,'edgecolor':'0.2','ls':'--','lw':'0.5'},
           axes=ax_corr0)



plot.axhline(0, color='red', lw=2)

sc0=ax0.scatter(df1_filled['lon'],df1_filled['lat'], c=df1_filled['user_count-divided1000'],
    cmap=cm.coolwarm,norm=MidpointNormalize(midpoint=mid_value,vmin=color_min,vmax=color_max),
    s=1,alpha=1) 


#Setting up gridSpec, bovenste rij, 2 subplots, onderste rij, 1 subplot.  
ncol_top=11
gs=gridspec.GridSpec(1,ncol_top)
gs.update(left=0.1, right=0.95, wspace=0.2, bottom=0.48, top=0.95)
#ax0=plt.subplot(gs[0,0]) #For a SubplotSpec that spans multiple cells, use slice.
#ax1=plt.subplot(gs[0,1])
ax0=plt.subplot(gs[0, :int(math.floor(ncol_top/2)) ]) #For a SubplotSpec that spans multiple cells, use slice.
leg0=plt.subplot(gs[0, int(math.floor(ncol_top/2)) ])
ax1=plt.subplot(gs[0, int(math.floor(ncol_top/2)+1): ])


gs2=gridspec.GridSpec(1,1)
gs2.update(left=0.1, right=0.95, bottom=0.05, top=0.40)
ax2=plt.subplot(gs2[0,0])

'''
