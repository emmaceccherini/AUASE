#%%
import pickle
import matplotlib.pyplot as plt
import numpy as np
#%%
# load accuracies_AUASE_temp.pkl
with open('DBLP_V3/Temporal Classification/accuracies_AUASE_temp.pkl', 'rb') as f:
    DBLP_accuracies_AUASE_temp = pickle.load(f)

with open('DBLP_V3/Temporal Classification/accuracies_UASE_temp.pkl', 'rb') as f:
    DBLP_accuracies_UASE_temp = pickle.load(f)

with open('DBLP_V3/Temporal Classification/accuracies_CONN_temp.pkl', 'rb') as f:
    DBLP_accuracies_CONN_temp = pickle.load(f)

with open('DBLP_V3/Temporal Classification/accuracies_DRLAN_temp.pkl', 'rb') as f:
    DBLP_accuracies_DRLAN_temp = pickle.load(f)

with open('DBLP_V3/Temporal Classification/accuracies_DySAT_temp.pkl', 'rb') as f:
    DBLP_accuracies_DySAT_temp = pickle.load(f)

with open('DBLP_V3/Temporal Classification/accuracies_glodyne_temp.pkl', 'rb') as f:
    DBLP_accuracies_glodyne_temp = pickle.load(f)

with open('DBLP_V3/Temporal Classification/accuracies_glodyne_temp.pkl', 'rb') as f:
    DBLP_accuracies_glodyne_temp = pickle.load(f)
    
with open('DBLP_V3/Temporal Classification/acc_random.pkl', 'rb') as f:
    DBLP_accuracies_random_temp = pickle.load(f)

with open('DBLP_V3/Temporal Classification/accuracies_dyrep_temp.pkl', 'rb') as f:
    DBLP_accuracies_dyrep_temp = pickle.load(f)
#%%
# load accuracies_AUASE_temp.pkl
with open('ACM/dynamic classification/accuracies_AUASE_temp.pkl', 'rb') as f:
    ACM_accuracies_AUASE_temp = pickle.load(f)

with open('ACM/dynamic classification/accuracies_UASE_temp.pkl', 'rb') as f:
    ACM_accuracies_UASE_temp = pickle.load(f)

with open('ACM/dynamic classification/accuracies_CONN_temp.pkl', 'rb') as f:
    ACM_accuracies_CONN_temp = pickle.load(f)

with open('ACM/dynamic classification/accuracies_DRLAN_temp.pkl', 'rb') as f:
    ACM_accuracies_DRLAN_temp = pickle.load(f)

with open('ACM/dynamic classification/accuracies_DySAT_temp.pkl', 'rb') as f:
    ACM_accuracies_DySAT_temp = pickle.load(f)

with open('ACM/dynamic classification/accuracies_glodyne_temp.pkl', 'rb') as f:
    ACM_accuracies_glodyne_temp = pickle.load(f)
    
with open('ACM/dynamic classification/acc_random.pkl', 'rb') as f:
    ACM_accuracies_random_temp = pickle.load(f)
#%%
# load accuracies_AUASE_temp.pkl
with open('Epinions/dynamic classification/accuracies_AUASE_temp.pkl', 'rb') as f:
    Epin_accuracies_AUASE_temp = pickle.load(f)

with open('Epinions/dynamic classification/accuracies_UASE_temp.pkl', 'rb') as f:
    Epin_accuracies_UASE_temp = pickle.load(f)

with open('Epinions/dynamic classification/accuracies_CONN_temp.pkl', 'rb') as f:
    Epin_accuracies_CONN_temp = pickle.load(f)

with open('Epinions/dynamic classification/accuracies_DRLAN_temp.pkl', 'rb') as f:
    Epin_accuracies_DRLAN_temp = pickle.load(f)

with open('Epinions/dynamic classification/accuracies_DySAT_temp.pkl', 'rb') as f:
    Epin_accuracies_DySAT_temp = pickle.load(f)

with open('Epinions/dynamic classification/accuracies_glodyne_temp.pkl', 'rb') as f:
    Epin_accuracies_glodyne_temp = pickle.load(f)

with open('Epinions/dynamic classification/acc_random.pkl', 'rb') as f:
    Epin_accuracies_random_temp = pickle.load(f)

with open('Epinions/dynamic classification/accuracies_Dyrep_temp.pkl', 'rb') as f:
    Epin_accuracies_dyrep_temp = pickle.load(f)
# %%
# plot accuracies
import matplotlib.pyplot as plt

# Assuming DBLP_accuracies_AUASE_temp, DBLP_accuracies_UASE_temp, etc., are defined
# and assuming you have a common x-axis range for the years (2000-2004)
years_DBLP = [ 2007, 2008, 2009]
years_ACM = [2010, 2011, 2012, 2013, 2014]
years_Epin = [2007, 2008, 2009, 2010, 2011]



# %%
# Create a 2x2 grid but merge the bottom two cells for the third plot
fig, axes = plt.subplots(2, 2, figsize=(9, 8))

# First plot (top left)
axes[0, 0].plot(years_DBLP, DBLP_accuracies_AUASE_temp, "-o", label='AUASE', lw=4)
axes[0, 0].plot(years_DBLP, DBLP_accuracies_UASE_temp, "-o", label='UASE', lw=4)
axes[0, 0].plot(years_DBLP, DBLP_accuracies_CONN_temp, "-o", label='CONN', lw=4)
axes[0, 0].plot(years_DBLP, DBLP_accuracies_DRLAN_temp, "-o", label='DRLAN', lw=4)
axes[0, 0].plot(years_DBLP, DBLP_accuracies_DySAT_temp, "-o", label='DySAT', lw=4)
axes[0, 0].plot(years_DBLP, DBLP_accuracies_glodyne_temp, "-o", label='GloDyNE', lw=4)
axes[0,0].plot(years_DBLP, DBLP_accuracies_dyrep_temp, "-o", label='DyRep', lw=4)
axes[0, 0].plot(years_DBLP, DBLP_accuracies_random_temp[6:], "k--", label='Baseline', lw=4)

# axes[0, 0].set_xlabel('Year', fontsize=19)
axes[0, 0].set_ylabel('Accuracy', fontsize=20)
axes[0, 0].set_title('(a) DBLP', fontsize=20)
axes[0, 0].tick_params(axis='both', labelsize=20)
axes[0, 0].set_xticks(years_DBLP)

# Second plot (top right)
axes[0, 1].plot(years_ACM, ACM_accuracies_AUASE_temp, "-o", lw=4)
axes[0, 1].plot(years_ACM, ACM_accuracies_UASE_temp, "-o", lw=4)
axes[0, 1].plot(years_ACM, ACM_accuracies_CONN_temp, "-o", lw=4)
axes[0, 1].plot(years_ACM, ACM_accuracies_DRLAN_temp, "-o", lw=4)
axes[0, 1].plot(years_ACM, ACM_accuracies_DySAT_temp, "-o", lw=4)
axes[0, 1].plot(years_ACM, ACM_accuracies_glodyne_temp, "-o", lw=4)
axes[0, 1].plot(years_ACM, ACM_accuracies_random_temp[10:], "k--", lw=4)

axes[0, 1].set_xlabel('Year', fontsize=20)
# axes[0, 1].set_ylabel('Accuracy', fontsize=19)
axes[0, 1].set_title('(b) ACM', fontsize=20)
axes[0, 1].tick_params(axis='both', labelsize=20)
axes[0, 1].set_xticks(years_ACM)

# Third plot (centered at bottom, merged across two columns)
axes[1, 0].plot(years_Epin, Epin_accuracies_AUASE_temp, "-o", lw=4)
axes[1, 0].plot(years_Epin, Epin_accuracies_UASE_temp, "-o", lw=4)
axes[1, 0].plot(years_Epin, Epin_accuracies_CONN_temp, "-o", lw=4)
axes[1, 0].plot(years_Epin, Epin_accuracies_DRLAN_temp, "-o", lw=4)
axes[1, 0].plot(years_Epin, Epin_accuracies_DySAT_temp, "-o", lw=4)
axes[1, 0].plot(years_Epin, Epin_accuracies_glodyne_temp, "-o", lw=4)
axes[1, 0].plot(years_Epin, Epin_accuracies_dyrep_temp, "-o", lw=4)
axes[1, 0].plot(years_Epin, Epin_accuracies_random_temp[6:], "k--", lw=4)

axes[1, 0].set_xlabel('Year', fontsize=20)
axes[1, 0].set_ylabel('Accuracy', fontsize=20)
axes[1, 0].set_title('(c) Epinions', fontsize=20)
axes[1, 0].tick_params(axis='both', labelsize=20)
axes[1, 0].set_xticks(years_Epin)

# Hide the fourth (bottom-right) plot as we don't need it
axes[1, 1].axis('off')

# Add a legend at the bottom
# fig.legend(loc='lower right', ncol=2, prop={'size': 19})

# # Adjust layout to make room for the legend
# plt.tight_layout(rect=[0, 0, 1, 0.97])
fig.legend(loc='center', bbox_to_anchor=(0.75, 0.32), ncol=2, prop={'size': 20})

# Adjust layout to make room for the legend
plt.tight_layout(rect=[0, 0.1, 1, 0.97])

# Save the figure
plt.savefig('temporal_classification.png', dpi=300)
plt.show()