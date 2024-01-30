# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import sys
import pandas as pd
import time
import sys
sys.path.append('../')

# %%
from Flagship4ML.create_sims import create_simulated_images


# %%
from dask.distributed import Client

client = Client("tls://192.168.101.78:33862")
client

# %%
catalogue = pd.read_parquet('/data/astro/scratch2/lcabayol/NFphotoz/data/15527.parquet',
                    engine='pyarrow'
                    )

# %%
t0=time.time()
ImageSimulator = create_simulated_images(catalogue,
                                         Ngals=1_000,
                                         bands=['CFHT_U', 'CFHT_G', 'CFHT_R', 'CFHT_I', 'CFHT_Z'],
                                         add_poisson=True,
                                         add_constant_background=True,
                                         use_dask=True,
                                         output_dir='/data/astro/scratch2/lcabayol/NFphotoz/data/CHFT_sims_bkg_test/',
                                        )

print(time.time()-t0)

# %%
band = 'CFHT_U'
ii = 100

t = np.load(f'/data/astro/scratch2/lcabayol/NFphotoz/data/CHFT_sims_bkg/data_{ii}/cutout_{band}.npy')
m= np.load(f'/data/astro/scratch2/lcabayol/NFphotoz/data/CHFT_sims_bkg/data_{ii}/metadata_{band}.npy')

# %%
import matplotlib.pyplot as plt
plt.imshow(t)
plt.colorbar()

# %%
t.sum()

# %%
m

# %%

# %%

# %%
tnoise = np.random.poisson(100*t)

# %%
plt.imshow(tnoise/100)

# %%
