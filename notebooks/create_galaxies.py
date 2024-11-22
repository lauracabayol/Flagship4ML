# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

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
catalogue = pd.read_parquet('/data/astro/scratch/lcabayol/NFphotoz/data/16511_PAU.parquet',
                    engine='pyarrow'
                    )

# %%
catalogue = catalogue[catalogue.observed_redshift_gal<1]

# %% [markdown]
# catalogue['imag'] = -2.5 * np.log10(catalogue.cfht_megacam_i_1_el_model3_odonnell_ext_error) - 48.6 
# catalogue = catalogue[catalogue.imag < 24]

# %%
#bands = ['CFHT_U', 'CFHT_G', 'CFHT_R', 'CFHT_I', 'CFHT_Z']
bands = [f'pau_nb{x}' for x in np.arange(455,855,10)]
bands_el = [f'pau_nb{x}_el' for x in np.arange(455,855,10)]

# %%
rename_map = dict(zip(bands_el,bands))
catalogue = catalogue.rename(mapper = rename_map, axis = 1)

# %%
#bands = ['CFHT_U', 'CFHT_G', 'CFHT_R', 'CFHT_I', 'CFHT_Z']
#bands = [f'pau_nb{x}' for x in np.arange(455,855,10)]

# %%
t0=time.time()
ImageSimulator = create_simulated_images(catalogue,
                                         Ngals=1_000,
                                         bands=bands,
                                         add_poisson=True,
                                         add_psf=True,
                                         add_constant_background=True,
                                         use_dask=False,
                                         num_exposures=3,
                                         output_dir='/data/astro/scratch/lcabayol/NFphotoz/data/PAUS_sims_test_v3/',
                                        )

print(time.time()-t0)

# %%

# %%

# %%

# %%

# %%

# %%

# %%
photometry = ImageSimulator._get_photometry(ImageSimulator.catalogue)
morphology = ImageSimulator._get_morphology(ImageSimulator.catalogue)

# %%
for ii in range(1000):
    for band in bands:
        for exp in range(3):
            ImageSimulator._create_simulated_galaxy(ii, band, exp,photometry,morphology )

# %%

# %%
a=1

# %%
band = 'pau_nb645'
ii = 0
exp=2

t = np.load(f'/data/astro/scratch/lcabayol/NFphotoz/data/PAUS_sims_v3/data_{ii}/cutout_{band}_exp{exp}.npy')
m= np.load(f'/data/astro/scratch/lcabayol/NFphotoz/data/PAUS_sims_v3/data_{ii}/metadata_{band}_exp{exp}.npy')

# %%
import matplotlib.pyplot as plt
plt.imshow(t)
plt.colorbar()

# %%
exp=1
ii=0
for ib, band in enumerate(bands):
    m= np.load(f'/data/astro/scratch/lcabayol/NFphotoz/data/PAUS_sims_v3/data_{ii}/metadata_{band}_exp{exp}.npy')
    plt.scatter(ib,m[0,1], color ='steelblue')
plt.show()

# %%
m.shape

# %%

# %%
tnoise = np.random.poisson(100*t)

# %%
plt.imshow(tnoise/100)

# %%
