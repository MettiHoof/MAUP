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
import matplotlib.patches as patches #For rectangles
import pandas as pd #For data handling
import numpy as np #For fast array handling
import math # For some math expressions
import statsmodels.api as sm #For statistical analysis
from statsmodels.stats.outliers_influence import summary_table #For summarizing statistical analysis
import brewer2mpl #For color brewing
############################
#0.2 Setup in and output paths
############################


########################################################
#1. Create example data 
########################################################

#Number of points we will consider
n=80

############################
#1.1 Create data for spatial pattern (x and y coordinates)
############################
#We try to mimic two streets and some random houses

#Mimic a street with houses around by means of random divergence from a linear regression
#Street one has got 60% of the houses
share_street1=0.6
intercept_y_street1= 0.9
coef_street1= -0.74
error_street1=np.random.random_sample(int(n*share_street1))*0.04 #small error term

x_coord_street1=np.random.random_sample(int(n*share_street1))
y_coord_street1 = intercept_y_street1 + x_coord_street1*coef_street1 + error_street1

#Street two has got 30% of the houses
share_street2=0.30
intercept_y_street2=-0.4
x_min_street2=0.4
x_max_street2=1
coef_street2= 1.4
error_street2=np.random.random_sample(int(n*share_street2))*0.04 #small error term

x_coord_street2=np.random.randint(int(x_min_street2*1000), int(x_max_street2*1000), size=int(n*share_street2))*0.001
y_coord_street2 = intercept_y_street2 + x_coord_street2*coef_street2 + error_street2

# Ten percent of households are located at random
share_random=0.1
x_coord_rand=np.random.random_sample(int(n*share_random))
y_coord_rand=np.random.random_sample(int(n*share_random))

#Put created coordinate data together
x_coord=np.concatenate((x_coord_street1, x_coord_street2, x_coord_rand))
y_coord=np.concatenate((y_coord_street1, y_coord_street2, y_coord_rand))


############################
#1.2 Create data for value observations (x and y)
############################

#Create x and y data, which we pretend to be figures on income and annual sick days

#income (x) from normal poission distribution around 40k
lamda = 40 # mean and standard deviation
x=np.random.poisson(lamda,n)*1000
x_label='Income per year'

#number of annual sick days (y) from uniform distribution
#y_min, y_max = 1,20
#y = np.random.randint(low=y_min, high=y_max, size=n)

#number of annual sick days (y) from normal poission distribution around 9
y=np.random.poisson(9,n)
y_label='Sick days per year'

############################
#1.3 Put all data together in a dataframe
############################

#Creata and store a pandas dataframe
df = pd.DataFrame(columns = ['x_coord', 'y_coord', 'income','sickdays'])
df['x_coord'] = x_coord
df['y_coord'] = y_coord
df['income'] = x
df['sickdays'] = y


############################
#1.4 Create some kind of spatial effect
############################

#We create a spatial effect by introducing an income gradient from south-west to north-east
df['income']=df['income']*(df['y_coord']*df['y_coord']+0.6)

df['test']=(df['x_coord']*10-5)
df['sickdays']=df['sickdays']+(df['x_coord']*10-5)
df['sickdays'] = df.sickdays.astype(int)
df.loc[df['sickdays'] < 0, 'sickdays'] = 1

'''
df.loc[df['First Season'] > 1990, 'First Season'] = 1
'''
############################
#1.5 Split dataframe for different deliniations
############################

#df[df['first_name'].notnull() & (df['nationality'] == "USA")]

df2_1=df[df['x_coord']<0.5].copy()
df2_2=df[df['x_coord']>0.5].copy()


df4_1=df[(df['x_coord']<0.5) & (df['y_coord']>0.5)].copy()
df4_2=df[(df['x_coord']<0.5) & (df['y_coord']<0.5)].copy()
df4_3=df[(df['x_coord']>0.5) & (df['y_coord']>0.5)].copy()
df4_4=df[(df['x_coord']>0.5) & (df['y_coord']<0.5)].copy()

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


#Setting up large gridSpecs
gs_map=gridspec.GridSpec(3,1) #nrows,ncols
gs_map.update(left=0.02, right=0.28, bottom=0.03, top=0.89, hspace=0.15) #update position of grid in figure

gs_hist=gridspec.GridSpec(3,1) #nrows,ncols
gs_hist.update(left=0.35, right=0.70, bottom=0.05, top=0.89, hspace=0.3) #update position of grid in figure

gs_corr=gridspec.GridSpec(3,1) #nrows,ncols
gs_corr.update(left=0.76, right=0.98, bottom=0.05, top=0.89, hspace=0.3) #update position of grid in figure


#Setting up inner gridspecs
gs_map_inner=gridspec.GridSpecFromSubplotSpec(11,20,subplot_spec=gs_map[0,0])

gs_hist0_inner=gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=gs_hist[0,0])
gs_hist1_inner=gridspec.GridSpecFromSubplotSpec(2,2,subplot_spec=gs_hist[1,0])

gs_hist2_inner=gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=gs_hist[2,0])

gs_hist2_inner_inner0=gridspec.GridSpecFromSubplotSpec(2,2,subplot_spec=gs_hist2_inner[0,0])
gs_hist2_inner_inner1=gridspec.GridSpecFromSubplotSpec(2,2,subplot_spec=gs_hist2_inner[0,1])

#Naming gridblocks
#ax_map0=plt.subplot(gs_map[0,0]) #row,col
ax_map0_0=plt.subplot(gs_map_inner[0:10,0:14]) #For a SubplotSpec that spans multiple cells, use slice.
ax_map0_1=plt.subplot(gs_map_inner[0:4,14:20])
ax_map1_1=plt.subplot(gs_map_inner[6:10,14:20])

ax_map1=plt.subplot(gs_map[1,0])
ax_map2=plt.subplot(gs_map[2,0])


ax_hist0_0=plt.subplot(gs_hist0_inner[0,0])#row,col
ax_hist0_1=plt.subplot(gs_hist0_inner[0,1])


ax_hist1_0=plt.subplot(gs_hist1_inner[0,0])#row,col
ax_hist1_1=plt.subplot(gs_hist1_inner[0,1])
ax_hist2_0=plt.subplot(gs_hist1_inner[1,0])
ax_hist2_1=plt.subplot(gs_hist1_inner[1,1])
'''
ax_hist3_0=plt.subplot(gs_hist2_inner[0,0])#row,col
ax_hist3_1=plt.subplot(gs_hist2_inner[0,1])
ax_hist3_2=plt.subplot(gs_hist2_inner[0,2])#row,col
ax_hist3_3=plt.subplot(gs_hist2_inner[0,3])

ax_hist4_0=plt.subplot(gs_hist2_inner[1,0])
ax_hist4_1=plt.subplot(gs_hist2_inner[1,1])
ax_hist4_2=plt.subplot(gs_hist2_inner[1,2])
ax_hist4_3=plt.subplot(gs_hist2_inner[1,3])
'''

ax_hist3_0=plt.subplot(gs_hist2_inner_inner0[0,0])#row,col
ax_hist3_1=plt.subplot(gs_hist2_inner_inner0[0,1])
ax_hist4_0=plt.subplot(gs_hist2_inner_inner0[1,0])#row,col
ax_hist4_1=plt.subplot(gs_hist2_inner_inner0[1,1])

ax_hist3_2=plt.subplot(gs_hist2_inner_inner1[0,0])
ax_hist3_3=plt.subplot(gs_hist2_inner_inner1[0,1])
ax_hist4_2=plt.subplot(gs_hist2_inner_inner1[1,0])
ax_hist4_3=plt.subplot(gs_hist2_inner_inner1[1,1])

ax_corr0=plt.subplot(gs_corr[0,0])
ax_corr1=plt.subplot(gs_corr[1,0])
ax_corr2=plt.subplot(gs_corr[2,0])


#Create combinations of different subplots together to be more efficient
maps=[ax_map0_0,ax_map0_1,ax_map1_1,ax_map1,ax_map2]
hists=[ax_hist0_0,ax_hist0_1,ax_hist1_0,ax_hist1_1,ax_hist2_0,ax_hist2_1,ax_hist3_0,ax_hist3_1,ax_hist3_2,ax_hist3_3,ax_hist4_0,ax_hist4_1,ax_hist4_2,ax_hist4_3]

hists_left=[ax_hist0_0,ax_hist1_0,ax_hist2_0,ax_hist3_0,ax_hist3_1,ax_hist4_0,ax_hist4_1]
hists_right=[ax_hist0_1,ax_hist1_1,ax_hist2_1,ax_hist3_2,ax_hist3_3,ax_hist4_2,ax_hist4_3]

corrs=[ax_corr0,ax_corr1,ax_corr2]
subplots=[]
subplots.extend(maps)
subplots.extend(hists)
subplots.extend(corrs)

############################
#2.1 Setup colors
############################

bmap = brewer2mpl.get_map('Set2', 'qualitative', 6)
colorbrew=bmap.hex_colors
#colorbrew.append('#808080') #put extra color grey to end because set2 only goes to 8 and we need 9 colors.
color_dict2={'2_1':colorbrew[0],
			'2_2':colorbrew[1]}


bmap = brewer2mpl.get_map('Set2', 'qualitative', 6)
colorbrew=bmap.hex_colors
#colorbrew.append('#808080') #put extra color grey to end because set2 only goes to 8 and we need 9 colors.
color_dict4={'4_1':colorbrew[2],
			'4_2':colorbrew[3],
			'4_3':colorbrew[4],
			'4_4':colorbrew[5]}

############################
#2.2 Plot figures based on input data
############################

#############
#Maps
#############
#Create maps based on input data for coordinates
ax_map0_0.scatter(df['x_coord'],df['y_coord'])

ax_map0_1.scatter(df['x_coord'],df['y_coord'],c=df['income'],cmap='Greys', s=6)
ax_map1_1.scatter(df['x_coord'],df['y_coord'],c=df['sickdays'],cmap='Greys', s=6)

ax_map1.scatter(df2_1['x_coord'],df2_1['y_coord'],color=color_dict2['2_1'])#, c=color_dict2['21'])
ax_map1.scatter(df2_2['x_coord'],df2_2['y_coord'],color=color_dict2['2_2'])#, c=color_dict2['22'])

ax_map2.scatter(df4_1['x_coord'],df4_1['y_coord'],c=color_dict4['4_1'])#, c=color_dict4['4_1'])
ax_map2.scatter(df4_2['x_coord'],df4_2['y_coord'],c=color_dict4['4_2'])#, c=color_dict4['4_2'])
ax_map2.scatter(df4_3['x_coord'],df4_3['y_coord'],c=color_dict4['4_3'])#, c=color_dict4['4_3'])
ax_map2.scatter(df4_4['x_coord'],df4_4['y_coord'],c=color_dict4['4_4'])#, c=color_dict4['4_4'])


#############
#Hists
#############
#Create hist based on input data for variables
#for hist_ax in hists:
#	hist_ax.hist(x)

#ax_hist all together delineation
ax_hist0_0.hist(df['income'])
ax_hist0_1.hist(df['sickdays'])

#ax_hist first delineation
ax_hist1_0.hist(df2_1['income'],color=color_dict2['2_1'])
ax_hist1_1.hist(df2_1['sickdays'],color=color_dict2['2_1'])

ax_hist2_0.hist(df2_2['income'],color=color_dict2['2_2'])
ax_hist2_1.hist(df2_2['sickdays'],color=color_dict2['2_2'])

#ax_hist second delineation
ax_hist3_0.hist(df4_1['income'],color=color_dict4['4_1'])
ax_hist3_1.hist(df4_3['income'],color=color_dict4['4_3'])
ax_hist4_0.hist(df4_2['income'],color=color_dict4['4_2'])
ax_hist4_1.hist(df4_4['income'],color=color_dict4['4_4'])

ax_hist3_2.hist(df4_1['sickdays'],color=color_dict4['4_1'])
ax_hist3_3.hist(df4_3['sickdays'],color=color_dict4['4_3'])
ax_hist4_2.hist(df4_2['sickdays'],color=color_dict4['4_2'])
ax_hist4_3.hist(df4_4['sickdays'],color=color_dict4['4_4'])


#############
#Corrs
#############
#Create corr based on input data for variables 

#Define helper function to calculated trendlines and confidence intervals
def regr_to_plot(x,y):
	z = np.polyfit(x,y, 1)
	p = np.poly1d(z)

	fitted_y_values=p(x)

	return fitted_y_values,1,1,1 #CI_df['x_data'], CI_df['low_CI'], CI_df['upper_CI']

#ax_corr0
ax_corr0.scatter(df['income'],df['sickdays'])
fitted_y_values0, x_for_shade0, low_CI_for_shade0, high_CI_for_shade0=regr_to_plot(df['income'],df['sickdays'])
ax_corr0.plot(df['income'],fitted_y_values0,lw = 2,color = '#539caf', alpha = 1)
#ax_corr0.fill_between(x_for_shade0, low_CI_for_shade0, high_CI_for_shade0, color = '#539caf', alpha = 0.4, label = '95% CI')

#ax_corr1
ax_corr1.scatter(df2_1['income'],df2_1['sickdays'],marker=".",s=44,alpha=0.4,edgecolors='none',color=color_dict2['2_1'])#, c=color_dict4['41'])
ax_corr1.scatter(df2_2['income'],df2_2['sickdays'],marker=".",s=44,alpha=0.4,edgecolors='none',color=color_dict2['2_2'])#

ax_corr1.scatter(df2_1['income'].mean(),df2_1['sickdays'].mean(),marker="s",s=52,color=color_dict2['2_1'])#, c=color_dict2['21'])
ax_corr1.scatter(df2_2['income'].mean(),df2_2['sickdays'].mean(),marker="s",s=52,color=color_dict2['2_2'])#, c=color_dict2['22'])

income_agg_for_2=[df2_1['income'].mean(),df2_2['income'].mean()]
sickdays_agg_for_2=[df2_1['sickdays'].mean(),df2_2['sickdays'].mean()]

fitted_y_values1, x_for_shade1, low_CI_for_shade1, high_CI_for_shade1=regr_to_plot(income_agg_for_2,sickdays_agg_for_2)
ax_corr1.plot(income_agg_for_2,fitted_y_values1,lw = 2,color = '#539caf', alpha = 1)
#ax_corr1.fill_between(x_for_shade1, low_CI_for_shade1, high_CI_for_shade1, color = '#539caf', alpha = 0.4, label = '95% CI')

#ax_corr2

ax_corr2.scatter(df4_1['income'],df4_1['sickdays'],marker=".",s=44,alpha=0.3,edgecolors='none',color=color_dict4['4_1'])#, c=color_dict4['41'])
ax_corr2.scatter(df4_2['income'],df4_2['sickdays'],marker=".",s=44,alpha=0.3,edgecolors='none',color=color_dict4['4_2'])#, c=color_dict4['42'])
ax_corr2.scatter(df4_3['income'],df4_3['sickdays'],marker=".",s=44,alpha=0.3,edgecolors='none',color=color_dict4['4_3'])#, c=color_dict4['43'])
ax_corr2.scatter(df4_4['income'],df4_4['sickdays'],marker=".",s=44,alpha=0.3,edgecolors='none',color=color_dict4['4_4'])

ax_corr2.scatter(df4_1['income'].mean(),df4_1['sickdays'].mean(),marker="s",s=52,edgecolors='none',color=color_dict4['4_1'])#, c=color_dict4['41'])
ax_corr2.scatter(df4_2['income'].mean(),df4_2['sickdays'].mean(),marker="s",s=52,edgecolors='none',color=color_dict4['4_2'])#, c=color_dict4['42'])
ax_corr2.scatter(df4_3['income'].mean(),df4_3['sickdays'].mean(),marker="s",s=52,edgecolors='none',color=color_dict4['4_3'])#, c=color_dict4['43'])
ax_corr2.scatter(df4_4['income'].mean(),df4_4['sickdays'].mean(),marker="s",s=52,edgecolors='none',color=color_dict4['4_4'])#, c=color_dict4['44'])


income_agg_for_4=[df4_1['income'].mean(),df4_2['income'].mean(),df4_3['income'].mean(),df4_4['income'].mean()]
sickdays_agg_for_4=[df4_1['sickdays'].mean(),df4_2['sickdays'].mean(),df4_3['sickdays'].mean(),df4_4['sickdays'].mean()]

fitted_y_values2, x_for_shade2, low_CI_for_shade2, high_CI_for_shade2=regr_to_plot(income_agg_for_4,sickdays_agg_for_4)
ax_corr2.plot(income_agg_for_4,fitted_y_values2,lw = 2,color = '#539caf', alpha = 1)
#ax_corr2.fill_between(x_for_shade2, low_CI_for_shade2, high_CI_for_shade2, color = '#539caf', alpha = 0.4, label = '95% CI')


############################
#2.3 Make up figures
############################

#############
#All subplots
#############
#Fade out the standard frame
for subplot in subplots:
	for pos in ['top','bottom','left','right']:
		#subplot.spines[pos].set_linewidth(0.7)
		subplot.spines[pos].set_color('0.6')


#############
#All maps
#############
for map_ax in maps:
	#Omit x and y axis of the coordinates
	map_ax.get_xaxis().set_visible(False)
	map_ax.get_yaxis().set_visible(False)

	#Fix y-axis
	map_ax.set_ylim(0,1)


#############
#Ax_map0_0
#############
#Set title
title_ax_map0_0= 'Household locations'
ax_map0_0.set_title(title_ax_map0_0,fontsize=10)

#Set n=x box
n_obs_1=_str=df.shape[0]
n_text_1= 'n = %s' %n_obs_1
ax_map0_0.text(0.03,0.05, n_text_1, ha='left', va='bottom', fontsize=7, color='0.2',
           bbox={'facecolor':'white', 'alpha':0.9,'edgecolor':'0.2','ls':'--','lw':'0.4'},
           transform=ax_map0_0.transAxes)


#############
#Ax_map0_1 and ax_map1_1
#############

#Set title
title_ax_map0_1= 'Income'
ax_map0_1.set_title(title_ax_map0_1,fontsize=8)

title_ax_map1_1= 'Sick days'
ax_map1_1.set_title(title_ax_map1_1,fontsize=8)


#############
#Ax_map1
#############

ax_map1.axvline(0.5, color='grey', lw=2)

#Set n=x box
n_obs_21=_str=df2_1.shape[0]
n_text_21= 'n = %s' %n_obs_21
ax_map1.text(0.03,0.05, n_text_21, ha='left', va='bottom', fontsize=7, color='0.2',
           bbox={'facecolor':'white', 'alpha':0.9,'edgecolor':'0.2','ls':'--','lw':'0.4'},
           transform=ax_map1.transAxes)

n_obs_22=_str=df2_2.shape[0]
n_text_22= 'n = %s' %n_obs_22
ax_map1.text(0.97,0.05, n_text_22, ha='right', va='bottom', fontsize=7, color='0.2',
           bbox={'facecolor':'white', 'alpha':0.9,'edgecolor':'0.2','ls':'--','lw':'0.4'},
           transform=ax_map1.transAxes)

#############
#Ax_map2
#############

ax_map2.axvline(0.5, color='grey', lw=2)
ax_map2.axhline(0.5, color='grey', lw=2)

#Set n=x box

n_obs_41=_str=df4_1.shape[0]
n_text_41= 'n = %s' %n_obs_41
ax_map2.text(0.03,0.55, n_text_41, ha='left', va='bottom', fontsize=7, color='0.2',
           bbox={'facecolor':'white', 'alpha':0.9,'edgecolor':'0.2','ls':'--','lw':'0.4'},
           transform=ax_map2.transAxes)


n_obs_42=_str=df4_2.shape[0]
n_text_42= 'n = %s' %n_obs_42
ax_map2.text(0.03,0.05, n_text_42, ha='left', va='bottom', fontsize=7, color='0.2',
           bbox={'facecolor':'white', 'alpha':0.9,'edgecolor':'0.2','ls':'--','lw':'0.4'},
           transform=ax_map2.transAxes)

n_obs_43=_str=df4_3.shape[0]
n_text_43= 'n = %s' %n_obs_43
ax_map2.text(0.97,0.55, n_text_43, ha='right', va='bottom', fontsize=7, color='0.2',
           bbox={'facecolor':'white', 'alpha':0.9,'edgecolor':'0.2','ls':'--','lw':'0.4'},
           transform=ax_map2.transAxes)

n_obs_44=_str=df4_4.shape[0]
n_text_44= 'n = %s' %n_obs_44
ax_map2.text(0.97,0.05, n_text_44, ha='right', va='bottom', fontsize=7, color='0.2',
           bbox={'facecolor':'white', 'alpha':0.9,'edgecolor':'0.2','ls':'--','lw':'0.4'},
           transform=ax_map2.transAxes)


#############
#All hists
#############
for hist_ax in hists:
	hist_ax.tick_params(axis = 'x', which = 'major', length=2, labelsize = 8, direction = 'out',color='0.4')
	hist_ax.tick_params(axis = 'y', which = 'major', length=2, labelsize = 8, direction = 'in',color='0.4') 

	hist_ax.grid(axis='x',which = 'major', alpha = 0.7,ls='--')
	hist_ax.grid(axis='y',which = 'major', alpha = 0.7,ls='--')
	hist_ax.set_axisbelow(True)

'''
# Specify different settings for major and minor grids
#ax2.grid(which = 'minor', alpha = 0.4,ls=':')
ax2.grid(axis='y',which = 'major', alpha = 0.7,ls='--')
ax2.grid(axis='x',which = 'major', alpha = 0.7,ls='--')

# put the grid behind
ax2.set_axisbelow(True)

ax0.set_xlabel(xlabel,
ax0.set_ylabel(ylabel,fontsize=9)
# Specify tick label size and direction
ax0.tick_params(axis = 'both', which = 'major', length=0, labelsize = 8, direction = 'in') 
ax0.set_xticklabels([])
ax0.set_yticklabels([])
'''

#############
#All hists left
#############
for hist_left_ax in hists_left:
	hist_left_ax.set_xlim(15000,80000)
	hist_left_ax.xaxis.set_ticks([20000,40000,60000,80000])
	hist_left_ax.set_xticklabels(['20k', '40k', '60k','80k']) 

	hist_left_ax.set_ylim(0,10)

#excemption for the first histogram
ax_hist0_0.set_ylim(0,25)


ax_hist1_0.set_xticklabels(['','', '', '','']) 
ax_hist1_0.tick_params(axis = 'x', direction = 'in')
  
ax_hist3_0.set_xticklabels(['','', '', ''])
ax_hist3_0.tick_params(axis = 'x', direction = 'in') 

ax_hist3_1.set_xticklabels(['','', '', ''])
ax_hist3_1.tick_params(axis = 'x', direction='in') 
ax_hist3_1.set_yticklabels(['','', '', ''])

ax_hist4_1.set_yticklabels(['','', '', ''])
 


#############
#All hists right
#############
for hist_right_ax in hists_right:
	hist_right_ax.set_xlim(0,20)
	hist_right_ax.xaxis.set_ticks([0,5,10,15,20])
	#hist_right_ax.set_xticklabels(['20k', '40k', '60k','80k']) 

	hist_right_ax.set_ylim(0,10)

#excemption for the first histogram
ax_hist0_1.set_ylim(0,25)


ax_hist1_1.set_xticklabels(['','', '', '','']) 
ax_hist1_1.tick_params(axis = 'x', direction = 'in') 


ax_hist3_2.set_xticklabels(['','', '', ''])
ax_hist3_2.tick_params(axis = 'x', direction = 'in')
 
ax_hist3_3.set_xticklabels(['','', '', ''])
ax_hist3_3.tick_params(axis = 'x', direction = 'in')
ax_hist3_3.set_yticklabels(['','', '', ''])

ax_hist4_3.set_yticklabels(['','', '', ''])



#############
#All hists middle
#############
hists_middle=[ax_hist1_0,ax_hist1_1,ax_hist2_0,ax_hist2_1]

for hist_middle_ax in hists_middle:
	hist_middle_ax.set_ylim(0,13)
	hist_middle_ax.tick_params(axis = 'both', labelsize=7)



#############
#All hists below
#############
hists_below=[ax_hist3_0,ax_hist3_1,ax_hist3_2,ax_hist3_3,
			 ax_hist4_0,ax_hist4_1,ax_hist4_2,ax_hist4_3]

for hist_below_ax in hists_below:
	hist_below_ax.set_ylim(0,11)
	hist_below_ax.tick_params(axis = 'both', labelsize=6)


#############
#Ax_hist row 0
#############

title_ax_hist_row00= 'Income per year'
title_ax_hist_row01= 'Sick days per year'

ax_hist0_0.set_title(title_ax_hist_row00,fontsize=10)
ax_hist0_1.set_title(title_ax_hist_row01,fontsize=10)

ax_hist0_0.set_ylabel('Households',fontsize=10)

#############
#Ax_hist1
#############\


#############
#Ax_hist2
#############

#############
#All corrs
#############
for corr_ax in corrs:

	corr_ax.set_xlim(15000,80000)
	corr_ax.xaxis.set_ticks([20000,40000,60000,80000])
	corr_ax.set_xticklabels(['20k', '40k', '60k','80k']) 

	#corr_ax.set_ylim(0,df['sickdays'].max()+1)

	corr_ax.set_ylim(0,16)
	corr_ax.yaxis.set_ticks([0,5,10,15,20])
	corr_ax.set_yticklabels(['0', '5', '10','15','20']) 

	corr_ax.tick_params(axis = 'x', which = 'major', length=2, labelsize = 8, direction = 'out',color='0.4')
	corr_ax.tick_params(axis = 'y', which = 'major', length=2, labelsize = 8, direction = 'in',color='0.4')

	corr_ax.grid(axis='x',which = 'major', alpha = 0.7,ls='--')
	corr_ax.grid(axis='y',which = 'major', alpha = 0.7,ls='--')
	corr_ax.set_axisbelow(True)

#############
#Ax_corr0
#############
title_ax_corr0= 'Income per year'
ax_corr0.set_title(title_ax_corr0,fontsize=10)

#ax_corr0.set_xlabel(x_label, fontsize=8)
ax_corr0.set_ylabel(y_label, fontsize=10)

#############
#Ax_corr1
#############

#############
#Ax_corr2
#############


#############
#Large figure
#############
#Set figure titles
title_text_1= 'Spatial Delineations'
fig.text(0.16,0.96, title_text_1, ha='center', va='center', fontsize=16, color='black')

#title_text_1= 'Spatial Delineations'
#fig.text(0.15,0.62, title_text_1, ha='center', va='center', fontsize=14, color='black')

title_text_2= 'Empirical Observations'
fig.text(0.65,0.96, title_text_2, ha='center', va='center', fontsize=16, color='black')




############################
#2.4 Save or show
############################

plt.show()

#outputfile='/Users/Metti_Hoof/Desktop/maupvis.png'
#plt.savefig(outputfile)



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
