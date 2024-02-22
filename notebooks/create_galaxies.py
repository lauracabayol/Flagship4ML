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

client = Client("tls://192.168.101.78:43830")
client

# %%
catalogue = pd.read_parquet('/data/astro/scratch2/lcabayol/NFphotoz/data/15527.parquet',
                    engine='pyarrow'
                    )

# %%
catalogue['imag'] = -2.5 * np.log10(catalogue.cfht_megacam_i_1_el_model3_odonnell_ext_error) - 48.6 
catalogue = catalogue[catalogue.imag < 24]

# %%
t0=time.time()
ImageSimulator = create_simulated_images(catalogue,
                                         Ngals=10_000,
                                         bands=['CFHT_U', 'CFHT_G', 'CFHT_R', 'CFHT_I', 'CFHT_Z'],
                                         add_poisson=True,
                                         add_psf=True,
                                         add_constant_background=True,
                                         use_dask=True,
                                         num_exposures=3,
                                         output_dir='/data/astro/scratch2/lcabayol/NFphotoz/data/CHFT_sims_bkg_psf_mexp_v2/',
                                        )

print(time.time()-t0)

# %%
band = 'CFHT_U'
ii = 2
exp=1

t = np.load(f'/data/astro/scratch2/lcabayol/NFphotoz/data/CHFT_sims_bkg_psf_mexp_test/data_{ii}/cutout_{band}_exp{exp}.npy')
m= np.load(f'/data/astro/scratch2/lcabayol/NFphotoz/data/CHFT_sims_bkg_psf_mexp_test/data_{ii}/metadata_{band}_exp{exp}.npy')

# %%
a=1

# %%
import matplotlib.pyplot as plt
plt.imshow(t)
plt.colorbar()

# %%
m

# %%

# %%

# %%
tnoise = np.random.poisson(100*t)

# %%
plt.imshow(tnoise/100)

# %%
