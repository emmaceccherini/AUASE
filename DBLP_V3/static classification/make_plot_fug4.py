#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy import io
import scipy.sparse as sparse
import pickle
import matplotlib as mpl
import glob
#%%
with open('Y_embedded_AUASE.pkl', 'rb') as f:
    Y_embedded_AUASE = pickle.load(f)


with open('Y_embedded_DRLAN_0.9_d12.pkl', 'rb') as f:
    Y_DRLAN = pickle.load(f)

with open('Y_embedded_DySAT.pkl', 'rb') as f:
    Y_DySAT = pickle.load(f) 

with open('Y_embedded_glodyne.pkl', 'rb') as f:
    Y_GloDyNe = pickle.load(f)

#%%
consensus_labels = np.load('consensus_labels.npy')
labels_unique = np.unique(consensus_labels)
labels_unique = np.append(labels_unique, 'unlabeled')
colours = np.array(list(mpl.colors.TABLEAU_COLORS.keys())[0:len(labels_unique)])
colours[len(labels_unique)-1] = 'grey'
cmap = mpl.colors.ListedColormap(colours)
#%%
files = glob.glob('labels/labels*.pkl')

# Load each file
labels = {}
for file in files:
    with open(file, 'rb') as f:
        labels[file] = pickle.load(f)
new_labels = {}
for key in list(labels.keys()):
    interval = key.split('_')[-1].split('.')[0]
    new_labels[interval] = labels[key]
labels = new_labels

def extract_first_year(key):
    return int(key.split(' - ')[0])

# Sort the keys and reorder the dictionary
sorted_keys = sorted(labels.keys(), key=extract_first_year)
labels = {key: labels[key] for key in sorted_keys}
#%%
T =9
As = [io.loadmat(f'adj/adj_matrix_{i}_bin.mat')['A'].astype(np.float64) for i in range(T)]
idx = [np.where(np.sum(As[t], axis=1) != 0)[0] for t in range(T)]
n = As[0].shape[0]
#%%
# modify the idx so that idx 2 is the sum of idx 2 plus idx 1 
idx[1] = np.unique(np.concatenate([idx[0], idx[1]]))
idx[2] = np.unique(np.concatenate([idx[1], idx[2]]))
idx[3] = np.unique(np.concatenate([idx[2], idx[3]]))
idx[4] = np.unique(np.concatenate([idx[3], idx[4]]))
idx[5] = np.unique(np.concatenate([idx[4], idx[5]]))
idx[6] = np.unique(np.concatenate([idx[5], idx[6]]))
idx[7] = np.unique(np.concatenate([idx[6], idx[7]]))
idx[8] = np.unique(np.concatenate([idx[7], idx[8]]))
# %%
lables_glod = {key: np.array(values)[idx[t]] for t, (key, values) in enumerate(labels.items())}
#%%
labels_unique = labels_unique[:-1]
# %%
# plot 2006 to 2008 for UASE
fig, axs = plt.subplots(2, 3, figsize=(15, 7))

t = 5
for i in range(2):
    key = list(labels.keys())[t]
    labels_plot = np.array(labels[key])
    labels_plot = labels_plot[labels_plot != 'unlabeled']
    for k, label in enumerate(labels_unique):
        axs[i,0].scatter(Y_embedded_AUASE[t][labels_plot == label, 0], Y_embedded_AUASE[t][labels_plot == label, 1], color=cmap(k))
        axs[0,0].set_title('AUASE', fontsize = 19)
        axs[i,0].tick_params(left = False, labelleft = False, labelsize = 19,
                             bottom = False, labelbottom = False)
    t = t + 2

t = 5
for i in range(2):
    key = list(labels.keys())[t]
    labels_plot = np.array(labels[key])
    labels_plot = labels_plot[labels_plot != 'unlabeled']
    for k, label in enumerate(labels_unique):
        axs[i,1].scatter(Y_DySAT[t][labels_plot == label, 0], Y_DySAT[t][labels_plot == label, 1], color=cmap(k))
        axs[0,1].set_title('DySAT', fontsize = 19)
        axs[i,1].tick_params(left = False, labelleft = False, labelsize = 19,
                             bottom = False, labelbottom = False)
    t = t + 2

axs[0,0].set_ylabel('2006',fontsize = 19)

axs[1,0].set_ylabel('2007', fontsize = 19)

t = 5
for i in range(2):
    key = list(labels.keys())[t]
    labels_plot = np.array(lables_glod[key])
    for k, label in enumerate(labels_unique):
        axs[i,2].scatter(Y_GloDyNe[t][labels_plot == label, 0], Y_GloDyNe[t][labels_plot == label, 1], color=cmap(k))
        axs[0,2].set_title('GloDyNe', fontsize = 19)
        axs[i,2].tick_params(left = False, labelleft = False, labelsize = 19,
                             bottom = False, labelbottom = False)
    t = t + 2
# axs[0,2].set_ylabel('2006',fontsize = 13)
# axs[1,2].set_ylabel('2007', fontsize = 13)

# Add the legend to the figure
# fig.legend(loc='right', labels=labels_unique, ncol=1, fontsize=19)

# Adjust layout to make room for the legend
plt.tight_layout(rect=[0, 0, 1, 0.97])

plt.savefig('DBLP.png', dpi=300)
# %%
