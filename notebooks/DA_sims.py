# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: flow4z
#     language: python
#     name: python3
# ---

from Flagship4ML.utils.explore_sims import plot_sim
from pathlib import Path

# +

plot_sim(sim_dir=Path('/Users/lauracabayol/Documents/data/test_sims'),
         ii=1,
         band='pau_nb845',
         exp=0)
# -


