from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import pandas as pd

variables = ["Zagreb", "Split", "Rijeka", "Osijek", "Zadar"]
hrdist = pd.DataFrame(
    #Zagreb, Split, Rijeka, Osijek, Zadar
    {
        "Zagreb": [0, 259, 132, 213, 198],
        "Split": [259, 0, 257, 289, 118],
        "Rijeka": [132, 257, 0, 333, 148],
        "Osijek": [213, 289, 333, 0, 316],
        "Zadar": [198, 118, 148, 316, 0]
    },
    index=variables
)
print(hrdist)
# exit()
# Distance matrix
model = MDS(n_components=2, dissimilarity='precomputed', metric=True)
mds_out = model.fit_transform(hrdist)

print(mds_out)

x = mds_out[:, 0]
y = mds_out[:, 1]
# z = mds_out[:,2]

# MDS plot of variables as entities
fig, ax = plt.subplots(figsize=(10, 5))
plot = ax.scatter(x, y, s=200)

# # Shift conflicting labels
for i, txt in enumerate(variables):
    ax.annotate(txt, xy=(x[i], y[i]), xytext=(10, -10),
                fontsize=15, va='top',
                xycoords='data', textcoords='offset points')

# # Colorbar
# cbar = fig.colorbar(plot, pad=0.05, fraction=0.03)
# cbar.solids.set_edgecolor("face")
# cbar.set_label(label='Y3',size=20)
# cbar.ax.tick_params(labelsize=20)

# # Ticks and labels
# ax.set_xlabel('Y1', fontsize=20)
# ax.set_ylabel('Y2', fontsize=20)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# plt.tick_params(axis='both', labelsize=15)
plt.tight_layout()

# plt.savefig('./mds_sheet_features.png', dpi=600, bbox_inches='tight', transparent=True)
plt.show()
