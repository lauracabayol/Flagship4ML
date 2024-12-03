# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # THIS NOTEBOOK SHOWS HOW TO CREATE SIMULATED IMAGES AND GALAXIES WITH THIS MODULE.
#

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import numpy as np
import pandas as pd
from pathlib import Path

from Flagship4ML.f4ml.sims_generator import CreateSimulatedImages

# %% [markdown]
# ## 1. DEFINE THE CATALOGUE

# %% [markdown]
# #### Make sure your catalogue is already filtered for the desired properties (redshift, magnitude, etc.)
#

# %%
catalogue = Path("/data/astro/scratch/lcabayol/NFphotoz/data/16511_PAU.parquet")

# %% [markdown]
# ### Define the bands you want to use.

# %%
# bands = ['CFHT_U', 'CFHT_G', 'CFHT_R', 'CFHT_I', 'CFHT_Z']
# bands = [f'pau_nb{x}' for x in np.arange(455,855,10)]
bands_el = [f"pau_nb{x}_el" for x in np.arange(455, 855, 10)]

# %% [markdown]
# # 2. CREATE THE SIMULATED IMAGES

# %%
ImageSimulator = create_simulated_images(
    catalogue=catalogue,
    bands=bands,
    crop_size=60,
    resolution=10,
    Ngals=1_000,
    add_poisson=True,
    add_psf=True,
    add_constant_background=True,
    use_dask=False,
    calibrate_flux=True,
    num_exposures=3,
    output_dir="/data/astro/scratch/lcabayol/NFphotoz/data/PAUS_sims_test_v3/",
)


# %% [markdown]
# ## ONE CAN ALSO CREATE GALAXIES USING THE CREATE_SIMS.PY SCRIPT.

# %% [markdown]
# python Flagship4ML/bin/create_sims.py --config path/to/config.yaml
