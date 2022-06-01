# -*- coding: utf-8 -*-

import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import mne
import matplotlib.pyplot as plt 
import pymc3 as pm
import arviz as az
import theano.tensor as tt
import pickle

#####plotting parameters
plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'figure.titlesize': 18})
plt.rcParams['font.family'] = "DeJavu Serif"
plt.rcParams['font.serif'] = "Cambria Math"


############################## Import and prepare epochs ######################
os.chdir("/grw_lkj_learners/")

amps = np.load("/grw_lkj_learners/data/leaerners_200ms_baseline.npy")

times = np.load("/grw_lkj_learners/data/times_200ms_baseline.npy")

C = amps.shape[0] #number of conditions C
E = amps.shape[1] #number of electrodes E 
S = amps.shape[2] #number of time-samples S

ts = np.arange(S)/256


######### GRW LKJ Moedl 4 tones ##############
with pm.Model() as mod:
    sds = [pm.HalfNormal.dist(1.0) for c in range(C)]
    lkj = [pm.LKJCholeskyCov("lkj"+str(c), n=E, eta=6.0, sd_dist=sds[c], compute_corr=True) for c in range(C)]
    L = [lkj[0][0], lkj[1][0], lkj[2][0], lkj[3][0]]
    Σ = [pm.Deterministic("Σ"+str(c), L[c].dot(L[c].T)) for c in range(C)]
    w = [pm.Normal('w'+str(c), 0, 1.0, shape=(E,S)) for c in range(C)]
    σ = [pm.HalfNormal('σ'+str(c), 1.0) for c in range(C)]
    t_sq = [tt.sqrt(ts) for c in range(C)]
    β = [pm.Deterministic('β'+str(c), w[c]*σ[c]*t_sq[c]) for c in range(C)] 
    B = [pm.Deterministic('B'+str(c), pm.math.matrix_dot(Σ[c],β[c])) for c in range(C)]
    α = [pm.Normal('α'+str(c), 0, 1.0, shape=S) for c in range(C)]
    μ = [pm.Deterministic('μ'+str(c), α[c] + B[c]) for c in range(C)]
    ϵ = [pm.HalfNormal('ϵ'+str(c), 0.05)+1 for c in range(C)]
    y = [pm.Normal("y"+str(c), mu=μ[c], sigma=ϵ[c], observed=amps[c]) for c in range(C)]


# with mod:
#     trace = pm.sample(1000, tune=1000, chains=4, cores=8, init='adapt_diag', target_accept=0.95)

    
tracedir = "/grw_lkj_learners/trace/"
# pm.backends.ndarray.save_trace(trace, directory=tracedir, overwrite=True)

with mod:
    trace = pm.load_trace(tracedir)


###### Plot Posteriors #####
fig, axs = plt.subplots(2,2, figsize=(14,10))
for i in range(3):
    if i == 0:
        ax = axs[0,0]
        t = 1
        c='teal'
    if i == 1:
        ax = axs[0,1]#
        t = 2
        c='limegreen'
    if i == 2:
        ax = axs[1,0]
        t = 3
        c='sienna'
    odiff = amps[0,12,:]-amps[t,12,:]
    pdiff = trace['μ0'][:,12,:]-trace['μ'+str(i+1)][:,12,:]
    postm = pdiff.mean(axis=0)
    posth5, posth95 = az.hdi(pdiff, hdi_prob=0.9).T
    ax.set_ylim([-3,9])
    ax.grid(alpha=0.2, zorder=-1)
    ax.axvline(0, color='k', zorder=-1, linestyle=':')
    ax.axhline(0, color='k', zorder=-1, linestyle=':')
    ax.plot(times, odiff, alpha=0.3, color='k', label="observed voltage")
    ax.plot(times, postm, color=c, label="posterior mean")
    ax.fill_between(times, posth5, posth95, color=c, alpha=0.3, label="90% HDI")
    ax.set_ylabel('Amplitude (μV)')
    ax.set_xlabel('Time (s)')
    ax.legend(fontsize=16, loc='lower right')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_title('Pz: Tone4 - Tone'+str(i+1))
axs[1,1].axis("off")
plt.tight_layout()
plt.savefig('posteriors_learners.png', dpi=300)
plt.close()

###### Plot Predictions #####
with mod:
    preds = pm.sample_posterior_predictive(trace)

fig, axs = plt.subplots(2,2, figsize=(14,10))
for i in range(3):
    if i == 0:
        ax = axs[0,0]
        t = 1
        c='teal'
    if i == 1:
        ax = axs[0,1]#
        t = 2
        c='limegreen'
    if i == 2:
        ax = axs[1,0]
        t = 3
        c='sienna'
    odiff = amps[0,12,:]-amps[t,12,:]
    pdiff = preds['y0'][:,12,:]-preds['y'+str(i+1)][:,12,:]
    predm = pdiff.mean(axis=0)
    pred_sdl = predm - pdiff.std(axis=0)
    pred_sdh = predm + pdiff.std(axis=0)
    ax.set_ylim([-3,9])
    ax.grid(alpha=0.2, zorder=-1)
    ax.axvline(0, color='k', zorder=-1, linestyle=':')
    ax.axhline(0, color='k', zorder=-1, linestyle=':')
    ax.plot(times, odiff, alpha=0.3, color='k', label="observed voltage")
    ax.plot(times, predm, color=c, label="predicted mean")
    ax.fill_between(times, pred_sdl, pred_sdh, color=c, alpha=0.3, label="predicted SD")
    ax.set_ylabel('Amplitude (μV)')
    ax.set_xlabel('Time (s)')
    ax.legend(fontsize=16, loc='lower right')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_title('Pz: Tone4 - Tone'+str(i+1))
axs[1,1].axis("off")
plt.tight_layout()
plt.savefig('predictions_learners.png', dpi=300)
plt.close()


###### Plot All Electrodes Posterior Contrasts ######
chans = pd.read_csv("/data/chans.csv")['chans'].values
path = "/grw_lkj_learners/electrodes_contrasts/"
for e in tqdm(range(E)): 
    chan = chans[e]
    fig, axs = plt.subplots(2,2, figsize=(14,10))
    for i in range(3):
        if i == 0:
            ax = axs[0,0]
            t = 1
            c='teal'
        if i == 1:
            ax = axs[0,1]#
            t = 2
            c='limegreen'
        if i == 2:
            ax = axs[1,0]
            t = 3
            c='sienna'
        odiff = amps[0,e,:]-amps[t,e,:]
        pdiff = trace['μ0'][:,e,:]-trace['μ'+str(i+1)][:,e,:]
        postm = pdiff.mean(axis=0)
        posth5, posth95 = az.hdi(pdiff, hdi_prob=0.9).T
        ax.set_ylim([-3,9])
        ax.grid(alpha=0.2, zorder=-1)
        ax.axvline(0, color='k', zorder=-1, linestyle=':')
        ax.axhline(0, color='k', zorder=-1, linestyle=':')
        ax.plot(times, odiff, alpha=0.3, color='k', label="observed voltage")
        ax.plot(times, postm, color=c, label="posterior mean")
        ax.fill_between(times, posth5, posth95, color=c, alpha=0.3, label="90% HDI")
        ax.set_ylabel('Amplitude (μV)')
        ax.set_xlabel('Time (s)')
        ax.legend(fontsize=16, loc='lower right')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_title(chan+': Tone4 - Tone'+str(i+1))
    axs[1,1].axis("off")
    plt.tight_layout()
    plt.savefig(path+chan+'_posteriors_learners.png', dpi=300)
    plt.close()



###### Plot Topomaps #####
non_targets = np.array([trace['μ1'],trace['μ2'],trace['μ3']]).mean(axis=0)
pdiff = trace['μ0']-non_targets
#pdiff = pdiff[:,:,77:205].mean(axis=2)
mdiff = pdiff.mean(axis=0)
h5diff,h95diff = np.array([az.hdi(pdiff[:,e,:], hdi_prob=0.9) for e in range(E)]).T

info_path = "/grw_lkj_learners/data/info.pickle"
with open(info_path, 'rb') as handle:
    info = pickle.load(handle)
    
h5ev = mne.EvokedArray(h5diff[51:].T, info)
mev = mne.EvokedArray(mdiff.T[51:].T, info)
h95ev = mne.EvokedArray(h95diff[51:].T, info)

selt = [0.2,0.4,0.6,0.8]

mne.viz.plot_evoked_topomap(h5ev, times=selt, scalings=1, vmin=-5, vmax=5, show=False)
plt.savefig('topomap_learners_h5.png', dpi=300)
plt.close()
mne.viz.plot_evoked_topomap(mev, times=selt, scalings=1, vmin=-5, vmax=5, show=False)
plt.savefig('topomap_learners_mean.png', dpi=300)
plt.close()
mne.viz.plot_evoked_topomap(h95ev, times=selt, scalings=1, vmin=-5, vmax=5, show=False)
plt.savefig('topomap_learners_h95.png', dpi=300)
plt.close()


############### Plot Correlations Topomaps #################
chans = pd.read_csv("/grw_lkj_learners/data/chans.csv")['chans'].values
tnames = ['tone_4', 'tone_1', 'tone_2', 'tone_3']
plt.rcParams['text.color'] = "white"
plt.rcParams.update({'font.size': 32})


fig, axes = plt.subplots(2,2, figsize=(20,20))
axs = [axes[0][0], axes[0][1], axes[1][0], axes[1][1]]
for c in range(C):
    corr_pz = trace['lkj'+str(c)+'_corr'].mean(axis=0)[12]
    ctopo = mne.viz.plot_topomap(corr_pz, info, names=chans, vmin=-1, vmax=1,
                                 show_names=True, cmap='viridis', show=False, axes=axs[c])
    axs[c].set_title('Tone '+tnames[c].replace('tone_','')+': Pz Correlations',
             fontsize=40, color='k')
    cbar = fig.colorbar(ctopo[0], ax=axs[c], shrink=0.6, pad=0.05, orientation='horizontal')
    cbar.set_label('LKJ ϱ mean')
plt.tight_layout()
plt.savefig('learners_tones_correlations_pz.png', dpi=300)
plt.close()


#############################################

######### Save summaries ##########
#summpath = "/grw_lkj_learners/tranks/"
summ = az.summary(trace, hdi_prob=0.9, round_to=4)
summ = pd.DataFrame(summ)
summ.to_csv('summary.csv')
print("summary saved")

bfmi = az.bfmi(trace)
bfmi = pd.DataFrame(bfmi)
bfmi.to_csv('bfmi.csv')
print("bfmi saved") 

ener = az.plot_energy(trace)
plt.savefig("energy.png", dpi=300)
plt.close()

########### Model fit

# loo = az.loo(trace, pointwise=True)
# loo = pd.DataFrame(loo)
# loo.to_csv("loo.csv")

# waic = az.waic(trace, pointwise=True)
# waic = pd.DataFrame(waic)
# waic.to_csv('waic.csv')

###plot rank

path = "/grw_lkj_learners/tranks/"

summ = pd.read_csv("summary.csv")
summ = summ.rename(columns={'Unnamed: 0':'variables'})
summ = summ[summ.ess_bulk < 1600] #lower than 10% ess
varias = summ.variables.values
#varias = [v for v in trace.varnames if not "__" in v]
for v in tqdm(varias):
    name = v.split('[')[0]
    if ',' in v:
        n1 = v.split('[')[1].replace(']', '').split(',')[0]
        n2 = v.split('[')[1].replace(']', '').split(',')[0]
        tp = trace[name].T[int(n2),int(n1)]
    else:
        n1 = v.split('[')[1].replace(']', '')
        tp = trace[name].T[int(n1)]
    # err = az.plot_rank(tp, kind='vlines', ref_line=True,
    #                        vlines_kwargs={'lw':1}, marker_vlines_kwargs={'lw':2})
    err = az.plot_trace(tp)
    plt.savefig(path+v+'_trace.png', dpi=300)
    plt.close()
    err = az.plot_autocorr(tp)
    plt.savefig(path+v+'_autocorr.png', dpi=300)
    plt.close()

# for var in tqdm(varias):
#     err = az.plot_rank(trace, var_names=[var], kind='vlines', ref_line=True,
#                        vlines_kwargs={'lw':1}, marker_vlines_kwargs={'lw':2})
#     plt.savefig(path+var+'_trank.png', dpi=300)
#     plt.close()
