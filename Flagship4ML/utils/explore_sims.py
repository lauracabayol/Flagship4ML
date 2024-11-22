#!/usr/bin/env python3
"""Module for exploring simulated galaxy images."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
})

def plot_sim(sim_dir: Path,
             ii: int,
             band: str,
             exp: int) -> None:
    """Plot simulated galaxy image and metadata.

    Parameters
    ----------
    sim_dir : Path
        Directory containing simulation data
    ii : int
        Galaxy index
    band : str
        Photometric band
    exp : int
        Exposure number
    """
    image = np.load(f'{sim_dir}/data_{ii}/cutout_{band}_exp{exp}.npy')
    metadata = np.load(f'{sim_dir}/data_{ii}/metadata_{band}_exp{exp}.npy')

    plt.imshow(image)
    plt.title(f'{band} exposure {exp}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar()
    plt.show()

    logger.info(f'{band} exposure {exp}')
    logger.info("Metadata:")
    logger.info(f'Redshift: {metadata[0,0]}')
    logger.info(f'Flux: {metadata[0,1]}')
    logger.info(f'Zero point: {metadata[0,2]}')
