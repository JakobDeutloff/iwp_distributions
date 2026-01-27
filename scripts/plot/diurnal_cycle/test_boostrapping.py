# %%
import matplotlib.pyplot as plt
import xarray as xr
import pickle
import numpy as np

# %%
with open("/work/bm1183/m301049/diurnal_cycle_dists/ccic_bootstrap_feedback_2d_sample_size_test.pkl", "rb") as f:
    ds = pickle.load(f)

with open("/work/bm1183/m301049/diurnal_cycle_dists/ccic_bootstrap_feedback_2d_length_test.pkl", "rb") as f:
    ds_length = pickle.load(f)


# %%
n_iterations = sorted(ds.keys())
mean_feedback = np.zeros(len(n_iterations))
std_feedback = np.zeros(len(n_iterations))

for i, n in enumerate(n_iterations):
    mean_feedback[i] = ds[n].sum(['iwp', 'local_time']).mean('iteration').mean('repeat_iteration')
    std_feedback[i] = ds[n].sum(['iwp', 'local_time']).mean('iteration').std('repeat_iteration')
    

# %% 
fig, ax = plt.subplots(figsize=(6,4))

ax.plot(n_iterations, mean_feedback, marker='o', color='k', label='Mean')
ax.fill_between(n_iterations, mean_feedback - std_feedback, mean_feedback + std_feedback, alpha=0.3, color='gray', label=r'$\pm$  $\sigma$')
ax.set_xscale('log')
ax.set_xticks(n_iterations)
ax.set_xticklabels(n_iterations)
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlabel("Sample Size")
ax.set_ylabel(r"$\lambda$ / W m$^{-2}$ K$^{-1}$")
ax.legend(frameon=False)
fig.savefig("plots/diurnal_cycle/publication/ccic_bootstrap_size_test.pdf", bbox_inches='tight')

# %% test window length
windows = sorted(ds_length.keys())
mean_feedback_length = np.zeros(len(windows))
std_feedback_length = np.zeros(len(windows))
for i, w in enumerate(windows):
    mean_feedback_length[i] = ds_length[w].sum(['iwp', 'local_time']).mean('iteration').mean('repeat_iteration')
    std_feedback_length[i] = ds_length[w].sum(['iwp', 'local_time']).mean('iteration').std('repeat_iteration')

# %% 
fig, ax = plt.subplots(figsize=(6,4))   
ax.plot(windows, mean_feedback_length, marker='o', color='k')
ax.fill_between(windows, mean_feedback_length - std_feedback_length, mean_feedback_length + std_feedback_length, alpha=0.3, color='gray')
ax.set_xlabel("Bootstrap Block Length / months")
ax.set_ylabel("Mean Bootstrap Feedback / W m$^{-2}$ K$^{-1}$")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()

# %%
