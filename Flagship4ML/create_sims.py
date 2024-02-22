from pathlib import Path
import numpy as np
import pandas as pd
import time
import json
import os

import dask
import dask.array as da

from astropy.io import fits
from astropy.modeling.functional_models import Sersic2D
from skimage.measure import block_reduce
from astropy.modeling.functional_models import Moffat2D
from scipy.signal import fftconvolve


class create_simulated_images():
    def __init__(self, catalogue, 
                 bands = ['CFHT_U', 'CFHT_G', 'CFHT_R', 'CFHT_I', 'CFHT_Z'],
                 background=None,
                 crop_size=60,
                 resolution=10,
                 Ngals=10,
                 add_poisson=True,
                 add_psf=True,
                 add_constant_background=True,
                 num_exposures=1,
                 use_dask=False,
                 output_dir='/data/astro/scratch2/lcabayol/NFphotoz/data/CHFT_sims'
                ):

        """
        Initialize the create_simulated_images.

        Parameters:
        - catalogue: DataFrame containing the astronomical catalogue data.
        - bands: List of band names.
        - crop_size: Size of the crop for simulated images.
        - resolution: Resolution of the simulated images.
        - Ngals: Number of galaxies to simulate.
        """
        
        self.bands = bands
        self.resolution=resolution
        self.crop_size=crop_size
        self.crop_size_psf = crop_size/2
        self.xgrid,self.ygrid = np.meshgrid(np.arange(0,self.resolution*self.crop_size,1), np.arange(0,self.resolution*self.crop_size,1))
        self.psf_xgrid,self.psf_ygrid = np.meshgrid(np.arange(0,self.resolution*self.crop_size_psf,1), np.arange(0,self.resolution*self.crop_size_psf,1))

        self.Ngals=Ngals
        self.add_poisson=add_poisson
        self.add_psf=add_psf
        self.num_exposures=num_exposures

        self.add_constant_background=add_constant_background
        self.output_dir=output_dir

                                
        with open('../mapping.json', 'r') as json_file:
            json_band_photometry = json.load(json_file)
        self.json_band_photometry = json_band_photometry 
                                 
        catalogue = self._preprocess_catalogue(catalogue)
        catalogue = self._map_band_names(catalogue) 
        self.catalogue=catalogue
        self.pix_scale = self._get_pix_scale()
        self.psf = self._get_psf_arcsec()
        self.zp = self._get_zp()
        self.exp_time = self._get_exp_time()

        if use_dask:
            self._create_simulated_catalogue_dask(self.Ngals)
        else:
            self._create_simulated_catalogue(self.Ngals)
                                 
        
        
        
    def _map_band_names(self, catalogue):
        """
        Map band names in the catalogue using the provided JSON mapping.

        Parameters:
        - catalogue: DataFrame containing the astronomical catalogue data.

        Returns:
        - catalogue: DataFrame with band names mapped according to the JSON mapping.
        """
        # Create a dictionary for mapping band names from the JSON mapping
        band_mapping = {key: value['band_name'] for key, value in self.json_band_photometry.items()}
        
        # Rename columns in the catalogue using the band_mapping dictionary
        catalogue = catalogue.rename(columns=band_mapping)
        
        return catalogue
    
    def _flux2mag(self, flux):
        """
        Convert flux to AB magnitude.

        Parameters:
        - flux: Flux values to be converted.

        Returns:
        - mag: Corresponding AB magnitudes.
        """
        mag = -2.5 * np.log10(flux) - 48.6
        return mag

    
    def _mag2e(self, mag, zp):
        """
        Convert AB magnitudes to electrons using a zero-point value.

        Parameters:
        - mag: AB magnitudes.
        - zp: Zero-point value for the conversion.

        Returns:
        - electrons: Corresponding electron values.
        """
        #temporary
        electrons = 10**(-0.4 * (mag - zp[None, :])) * self.exp_time[None,:]
        return electrons

        
    def _preprocess_catalogue(self, catalogue):
        """
        Preprocess the astronomical catalogue.

        Steps:
        - Filter galaxies based on observed redshift, bulge ellipticity, bulge r50, and disk r50.
        - Sample a specified number of galaxies.
        - Reset the index of the catalogue.

        Parameters:
        - catalogue: DataFrame containing the astronomical catalogue data.

        Returns:
        - catalogue: Preprocessed DataFrame.
        """
        # Filter galaxies based on certain criteria
        catalogue = catalogue[catalogue.observed_redshift_gal < 1]
        catalogue = catalogue[catalogue.bulge_ellipticity > 0]
        catalogue = catalogue[catalogue.bulge_r50 > 0]
        catalogue = catalogue[catalogue.disk_r50 > 0]
        
        # Sample a specified number of galaxies
        catalogue = catalogue.sample(self.Ngals)
        
        # Reset the index of the catalogue
        catalogue = catalogue.reset_index()
        
        return catalogue
    
    def _get_photometry(self, catalogue):
        """
        Extract photometry from the catalogue, convert to fluxes, and return.

        Parameters:
        - catalogue: DataFrame containing the astronomical catalogue data.

        Returns:
        - fluxes: DataFrame of flux values for each band.
        """
        # Extract photometry columns from the catalogue
        photometry = catalogue[self.bands]
        
        # Convert magnitudes to fluxes using the _flux2mag and _mag2e methods
        # temporary fix
        if not any('pau' in band for band in bands):
            mags = self._flux2mag(np.abs(photometry))
            fluxes = self._mag2e(mags, self.zp)
        else:
            fluxes = photometry
        
        return fluxes

    
    def _get_morphology(self, catalogue):
        """
        Extract morphology parameters from the catalogue and return as a DataFrame.

        Parameters:
        - catalogue: DataFrame containing the astronomical catalogue data.

        Returns:
        - morphology: DataFrame of morphology parameters.
        """
        # Extract relevant morphology parameters from the catalogue
        hlr_b = catalogue.bulge_r50.values
        hlr_d = catalogue.disk_r50.values
        nsersic_bulge = catalogue.bulge_nsersic.values
        nsersic_disk = catalogue.disk_nsersic.values
        ellip_bulge = catalogue.bulge_ellipticity.values
        ellip_disk = catalogue.disk_ellipticity.values
        bulge_disk_fraction = catalogue.bulge_fraction.values

        # Handle disk angle conversion
        catalogue.disk_angle.where(catalogue.disk_angle > 0, -catalogue.disk_angle, inplace=True)
        catalogue['disk_angle'] = catalogue.disk_angle / 360 * 2 * np.pi
        rotation_angle = catalogue.disk_angle.values

        # Extract redshift
        redshift = catalogue.observed_redshift_gal.values

        # Create a DataFrame with morphology parameters
        morphology = pd.DataFrame(
            np.c_[hlr_b, hlr_d, nsersic_bulge, nsersic_disk, ellip_bulge, ellip_disk,
                  bulge_disk_fraction, rotation_angle, redshift],
            columns=['hlr_b', 'hlr_d', 'nsersic_bulge', 'nsersic_disk', 'ellip_bulge', 'ellip_disk',
                     'bulge_disk_fraction', 'rotation_angle', 'redshift']
        )

        return morphology

    
    def _get_exp_time(self):
        """
        Get exposure times for each band from the JSON mapping.

        Returns:
        - exp_times: Dictionary of exposure times for each band.
        """
        exp_times = {value['band_name']: value['t_exp'] for key, value in self.json_band_photometry.items()}
        exp_times = np.array(list(exp_times.values()))
        return exp_times

    def _get_zp(self):
        """
        Get zero-points for each band from the JSON mapping.

        Returns:
        - zp: NumPy array of zero-point values for each band.
        """
        zp = {value['band_name']: value['ZP'] for key, value in self.json_band_photometry.items()}
        zp = np.array(list(zp.values()))
        return zp

    def _get_pix_scale(self):
        """
        Get pixel scales for each band from the JSON mapping.

        Returns:
        - pix_scales: Dictionary of pixel scales for each band.
        """
        pix_scales = {value['band_name']: value['pix_scale'] for key, value in self.json_band_photometry.items()}
        self.pix_scales = pix_scales
        return pix_scales
    

    def _get_psf_arcsec(self):
        """
        Get PSFs in arcseconds/pix for each band from the JSON mapping.

        Returns:
        - pix_scales: Dictionary of pixel scales for each band.
        """
        psfs = {value['band_name']: value['psf'] for key, value in self.json_band_photometry.items()}
        self.psfs = psfs
        return psfs
    
    
    def _simulate_galaxy(self, photometry, morphology, band):
        """
        Simulate a galaxy image based on provided photometry and morphology parameters.

        Parameters:
        - photometry: Photometry values for the galaxy in the specified band.
        - morphology: Morphology parameters for the galaxy.
        - band: Band for which the galaxy image is simulated.

        Returns:
        - gal: Simulated galaxy image.
        """

        # Extract pixel scale for the specified band
        pix_scale = self.pix_scales[band]
        psf = self.psfs[band]

        # Create 2D Sersic profiles for bulge and disk
        sersic_bulge = Sersic2D(x_0=int(self.resolution * self.crop_size / 2),
                                y_0=int(self.resolution * self.crop_size / 2),
                                ellip=morphology.ellip_bulge,
                                r_eff= self.resolution *morphology.hlr_b / pix_scale,
                                n=morphology.nsersic_bulge,
                                amplitude=1)

        gal_bulge = sersic_bulge(self.xgrid, self.ygrid)

        sersic_disk = Sersic2D(x_0=int(self.resolution * self.crop_size / 2),
                               y_0=int(self.resolution * self.crop_size / 2),
                               ellip=morphology.ellip_disk,
                               r_eff=self.resolution * morphology.hlr_d / pix_scale,
                               n=morphology.nsersic_disk,
                               amplitude=1)

        gal_disk = sersic_disk(self.xgrid, self.ygrid)

        # Calculate flux contributions from bulge and disk
        ib = self.bands.index(band)
        flux = photometry[band]*self.exp_time[ib]
        flux_bulge = flux * morphology.bulge_disk_fraction
        flux_disk = flux * (1 - morphology.bulge_disk_fraction)

        # Normalize and scale the galaxy components
        gal_bulge = gal_bulge * flux_bulge / gal_bulge.sum() * self.resolution**2
        gal_disk = gal_disk * flux_disk / gal_disk.sum() * self.resolution**2

        # Combine bulge and disk components
        gal = gal_bulge + gal_disk

        if self.add_poisson is True:
            # Add noise to the simulated galaxy image
            gal = np.random.poisson(gal)
        if self.add_constant_background is True:
            bkg = np.random.uniform(1,3,1) * self.exp_time[ib] * np.ones(shape=(self.crop_size*self.resolution, self.crop_size*self.resolution))
            bkg = np.random.poisson(bkg)
            gal = gal + bkg
            
        if self.add_psf is True:
            # Add PSF to the simulated galaxy image
            psf = self.resolution * psf / pix_scale
            gam = psf / (2. * np.sqrt(np.power(2., 1 / 4.76 ) - 1.))#alph = 4.76 is a default value for the Moffat
            amp = (4.76 - 1) / (np.pi * gam**2)
            moff = Moffat2D(amplitude=amp, 
                            x_0=int(self.resolution * self.crop_size_psf / 2),
                            y_0=int(self.resolution * self.crop_size_psf / 2), 
                            gamma=gam, 
                            alpha=4.76)
            psf_grid = moff(self.psf_xgrid, self.psf_ygrid)           
            gal = fftconvolve(gal,psf_grid, mode = 'same')
        
        #gal = gal + bkg
        gal = block_reduce(gal, (self.resolution, self.resolution), np.mean)
        gal = gal / self.exp_time[ib]

        return gal
                                                                                                   
    def _create_simulated_catalogue(self, Ngals):
        photometry = self._get_photometry(self.catalogue)
        morphology = self._get_morphology(self.catalogue)

        for ii in range(Ngals):
            os.makedirs(self.output_dir + f'/data_{ii}', exist_ok=False)
            for band in self.bands:
                for e in range(self.num_exposures):
                    gal = self._simulate_galaxy(photometry.iloc[ii], morphology.iloc[ii], band=band)
                    metadata = np.c_[morphology['redshift'][ii], photometry[band][ii]]
                    np.save(self.output_dir + f'data_{ii}/cutout_{band}_exp{e}.npy',gal)       
                    np.save(self.output_dir + f'data_{ii}/metadata_{band}_exp{e}.npy',metadata)  


    def _create_simulated_galaxy(self, ii, band, exp, photometry, morphology):
        """
        Create and save simulated galaxy images along with metadata for a specific band.

        Parameters:
        - ii: Index of the galaxy in the catalogue.
        - band: Band for which the galaxy image is simulated.
        - photometry: Photometry values for the galaxy.
        - morphology: Morphology parameters for the galaxy.
        """
        # Simulate galaxy image
        gal = self._simulate_galaxy(photometry.iloc[ii], morphology.iloc[ii], band=band)

        # Extract metadata
        metadata = np.c_[morphology['redshift'][ii], photometry[band][ii]]

        # Save simulated galaxy image and metadata
        np.save(self.output_dir + f'data_{ii}/cutout_{band}_exp{exp}.npy', gal)
        np.save(self.output_dir + f'data_{ii}/metadata_{band}_exp{exp}.npy', metadata)


    def _create_simulated_catalogue_dask(self, Ngals):
        """
        Create and save a catalog of simulated galaxy images for each band.

        Parameters:
        - Ngals: Number of galaxies to simulate.
        """
        # Get photometry and morphology for the entire catalogue
        photometry = self._get_photometry(self.catalogue)
        morphology = self._get_morphology(self.catalogue)

        # List to store delayed tasks
        delayed_tasks = []

        for ii in range(Ngals):
            # Create a directory for each galaxy
            os.makedirs(self.output_dir + f'data_{ii}', exist_ok=False)

            # Create delayed tasks for each band
            for band in self.bands:
                for e in range(self.num_exposures):
                    task = dask.delayed(self._create_simulated_galaxy)(ii, band, e, photometry, morphology)
                    delayed_tasks.append(task)

        # Compute delayed tasks using the dask scheduler
        dask.compute(*delayed_tasks, scheduler='threads')
