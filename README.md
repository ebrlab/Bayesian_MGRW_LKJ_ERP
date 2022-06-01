<h1> Bayesian Multivariate Gaussian Random Walk Regression for ERP Estimation with LKJ prior </h1>

<p>
	The present analysis implements a multivariate Gaussian random walk (MGRW) using a Lewandowski-Kurowicka-Joe (LKJ) prior distribution for estimating event-related potentials (ERPS).  
</p>
<p></p>

<h1> Model </h1>

<p> The present model attempts to estimate event-related potentials (ERPs) across the whole epoch and capture electrodes covariation/correlations at the same time. To that aim, we use an MGRW prior to model the voltage changes across time plus a Gaussian noise parameter. We reparametrised the gaussian random walk by simplifying to the assumption of an independent random walk per electrode, related via the covariance of a multivariate Gaussian. So that we could write the GRWs in the form &beta; = w&sigma;t, where w ~ Normal(0,1), sigma ~ HalfNormal(1) being a second moment parameter, and t = &Sqrt;times (i.e. the squared root of each of 295 sampled times from the start fo the baseline time<sub>0</sub> to the end of the epoch time<sub>S</sub>, S=295). Covariance is modelled as a Cholesky decomposition, where the Cholesky factor comes from a LKJ prior. A normal distribution (likelihood) is used for the observed voltages with a half-normal distribution as error parameter. </p>

<p align="center"> L<sub>1</sub>...L<sub>C</sub>, &rhov;<sub>1</sub>...&rhov;<sub>C</sub>, SD<sub>1</sub>...SD<sub>C</sub> = LKJ(n=E, &eta;=6)</p>
<p align="center"> &Sigma;<sub>1</sub>...&Sigma;<sub>C</sub> = L<sub>c</sub>L<sub>c</sub><sup>T</sup> </p>
<p align="center"> w<sub>1</sub>... w<sub>C</sub> ~ Normal(0,1) , S &times; E </p>	 
<p align="center"> <em>&sigma;</em><sub>1</sub>... <em>&sigma;</em><sub>C</sub> ~ HalfNormal(1) , S &times; 1 </p>
<p align="center"> t<sub>c,s</sub> = &Sqrt;time<sub>c,0</sub>... &Sqrt;time<sub>C,S</sub> </p>
<p align="center"> &beta;<sub>c</sub> = w<sub>c</sub>&sigma;<sub>c</sub>t<sub>c</sub> </p>
<p align="center"> B = &Sigma;<sub>c</sub>&beta;<sub>c</sub> , E &times; S </p>
<p align="center"> &alpha;<sub>c,s</sub> ~ Normal(0, 1) </p>
<p align="center"> &mu;<sub>c</sub> = &alpha;<sub>c,s</sub> + B </p>
<p align="center"> &varepsilon;<sub>c</sub> ~ HalfNormal(0.05) + 1 </p>
<p align="center"> y<sub>c</sub> ~ Normal(&mu;<sub>c</sub>, &varepsilon;<sub>c</sub>) </p>

<p> Where C = 4 mandarin tones (tone 1... tone 4), E = EEG electrodes (32), and S = number of samples (282, 100ms baseline, 1s epoch). Data comes from a tone detection oddball task (Tone 4 was the deviant target, 25% of total stimuli), completed by learners and non-learners of Chinese Mandarin. We fit two models, as described above, to data from each group: learners and non-learners. </p>

<p> We sampled the model using Markov chain Monte Carlo (MCMC) No U-turn sampling (NUTS) with 2000 tuning steps, 2000 samples, 4 chains. The model sampled well, with 1.01 > R&#770; > 0.99; BFMIs > 0.9, and bulk ESS > 1000 for all parameters (few parameters, 3-6 per model, however, were below 500 effective samples; which may indicate some sampling issues). Note that due to the high number of parameters, the trank/ folder contains traceplots and autocorrelation plots from parameters below 10% (1600) ESS instead (these plots show good mixing). </p>

<h1> Results </h1>

<p> The estimates from learners indicate that the target tone (tone 4) induced a strong positive voltage deflection after ~200ms respect to the non-target tones at Pz (i.e., tone 4 induced a P3b). Image below shows the contrasts between tone 4 and each other tone from posterior distributions. </p>

<p align="center">
	<img src="grw_lkj_learners/posteriors_learners.png" width="600" height="400" />
</p>

<p> The estimates from non-learners indicate that the target tone (tone 4) induced a milder positive voltage deflection after ~200ms respect to the non-target tones at Pz. Image below shows the contrasts between tone 4 and each other tone from posterior distributions. </p>

<p align="center">
	<img src="grw_lkj_non_learners/posteriors_non_learners.png" width="600" height="400" />
</p>


<p> Predictions from the posterior for learners indicate more uncertainty but the P3b is still present. Image below shows contrasts between tone 4 and each other tone from Pz predictions. </p>

<p align="center">
	<img src="grw_lkj_learners/predictions_learners.png" width="600" height="400" />
</p>

<p> Predictions from the posterior for non-learners also indicate more uncertainty but there is still a mild P3b. Image below shows contrasts between tone 4 and each other tone from Pz predictions. </p>

<p align="center">
	<img src="grw_lkj_non_learners/predictions_non_learners.png" width="600" height="400" />
</p>


<p> Images below show posterior distributions from the learners’ model as scalp topographies (posterior of tone 4 minus all other tones combined). </p>

<p align="center"><strong>5% highest density intervals (HDI)</strong></p>
<p align="center"> <img src="grw_lkj_learners/topomap_learners_h5.png" width="600" height="150" /> </p>

<p align="center"><strong>Posterior means</strong></p>
<p align="center"> <img src="grw_lkj_learners/topomap_learners_mean.png" width="600" height="150" /> </p>

<p align="center"><strong>95% highest density intervals (HDI)</strong></p>
<p align="center"> <img src="grw_lkj_learners/topomap_learners_h95.png" width="600" height="150" /> </p>


<p> Images below show posterior distributions from the non-learners’ model as scalp topographies (posterior of tone 4 minus all other tones combined). </p>

<p align="center"><strong>5% highest density intervals (HDI)</strong></p>
<p align="center"> <img src="grw_lkj_non_learners/topomap_non_learners_h5.png" width="600" height="150" /> </p>

<p align="center"><strong>Posterior means</strong></p>
<p align="center"> <img src="grw_lkj_non_learners/topomap_non_learners_mean.png" width="600" height="150" /> </p>

<p align="center"><strong>95% highest density intervals (HDI)</strong></p>
<p align="center"> <img src="grw_lkj_non_learners/topomap_non_learners_h95.png" width="600" height="150" /> </p>


<p> Correlations across electrodes per tone indicate that the target tone (Tone 4) shows strong correlations between the electrode showing maximum P3b amplitude Pz and surrounding electrodes (PO4, PO3, P4, P3, CP1, CP2). Pz is also moderately anticorrelated with frontal electrodes (as commonly observed for P3b). Differently, non-target tones show very low correlations and anti-correlations. This may indicate that these tones do not generate a P3b, i.e. their activity remains around zero independent of scalp topography, as electrodes do not capture activity corresponding to an elicited ERP. Images below show topomaps of correlations to Pz. </p>

<h3 align="center"> Learners: correlations to Pz </h3>
<p align="center"> <img src="grw_lkj_learners/learners_tones_correlations_pz.png" width="600" height="600" /> </p>

<h3 align="center"> Non-learners: correlations to Pz </h3>
<p align="center"> <img src="grw_lkj_non_learners/non_learners_tones_correlations_pz.png" width="600" height="600" /> </p>


<h1> Conclusion </h1>  

<p> The estimates show that there is a difference of P3b amplitude between learners and non-learners, but predictions remain somewhat uncertain. The present model includes a correlation matrix for electrodes, solving one problem from our previous model (https://github.com/ebrlab/Bayesian_MGRW_LKJ_ERP). However, there are minor sampling issues that may require more work on priors. Further work on developing a scientific model which can better model voltage activity is a relevant additional step. </p> 