import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil

from Flagship4ML.f4ml.sims_generator import CreateSimulatedImages


@pytest.fixture
def sample_catalogue():
    """Create a small sample catalogue for testing."""
    np.random.seed(42)
    n_samples = 10

    # Create dummy data that matches the structure needed
    data = {
        "observed_redshift_gal": np.random.uniform(0, 1, n_samples),
        "bulge_r50": np.random.uniform(0.5, 2.0, n_samples),
        "disk_r50": np.random.uniform(1.0, 3.0, n_samples),
        "bulge_nsersic": np.random.uniform(2.0, 4.0, n_samples),
        "disk_nsersic": np.random.uniform(0.5, 1.5, n_samples),
        "bulge_ellipticity": np.random.uniform(0.1, 0.5, n_samples),
        "disk_ellipticity": np.random.uniform(0.2, 0.6, n_samples),
        "bulge_fraction": np.random.uniform(0.2, 0.8, n_samples),
        "disk_angle": np.random.uniform(0, 180, n_samples),
    }

    # Add PAU narrow bands
    for x in np.arange(455, 855, 10):
        data[f"pau_nb{x}"] = np.random.uniform(1000, 5000, n_samples)

    return pd.DataFrame(data)


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup after tests
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_pix_scales():
    """Create a mock pixel scales dictionary for testing."""
    pix_scales = {}
    # Create entries for each PAU narrow band
    for x in np.arange(455, 855, 10):
        band_name = f"pau_nb{x}"
        pix_scales[band_name] = 0.26  # PAU camera pixel scale in arcsec/pixel
    return pix_scales


@pytest.fixture
def mock_psfs():
    """Create a mock PSFs dictionary for testing."""
    psfs = {}
    # Create entries for each PAU narrow band
    for x in np.arange(455, 855, 10):
        band_name = f"pau_nb{x}"
        psfs[band_name] = 1.2  # Mock PSF FWHM in arcsec/pixel
    return psfs


@pytest.fixture
def mock_exp_times():
    """Create a mock exposure times array for testing."""
    # Create array of exposure times for each PAU narrow band
    n_bands = len(np.arange(455, 855, 10))
    return np.full(n_bands, 100.0)  # Mock exposure time in seconds


@pytest.fixture
def mock_simulator(
    sample_catalogue, temp_output_dir, mock_pix_scales, mock_psfs, mock_exp_times
):
    """Create a simulator instance with all mocked parameters."""
    simulator = CreateSimulatedImages(
        catalogue=sample_catalogue,
        bands=[f"pau_nb{x}" for x in np.arange(455, 855, 10)],
        output_dir=temp_output_dir,
    )
    simulator.pix_scales = mock_pix_scales
    simulator.psfs = mock_psfs
    simulator.exp_time = np.array([mock_exp_times[band] for band in simulator.bands])
    return simulator


def test_create_simulated_images_basic(sample_catalogue, temp_output_dir):
    """Test basic functionality of create_simulated_images."""
    bands = [f"pau_nb{x}" for x in np.arange(455, 855, 10)]

    # Create test catalogue file
    catalogue_path = str(Path(temp_output_dir) / "sample_catalogue.parquet")
    sample_catalogue.to_parquet(catalogue_path)

    simulator = CreateSimulatedImages(
        catalogue=str(Path(temp_output_dir) / "sample_catalogue.parquet"),
        Ngals=2,
        bands=bands,
        add_poisson=True,
        add_psf=True,
        add_constant_background=True,
        use_dask=False,
        num_exposures=1,
        output_dir=temp_output_dir,
    )

    assert simulator is not None
    assert hasattr(simulator, "_get_photometry")
    assert hasattr(simulator, "_get_morphology")
    assert hasattr(simulator, "_create_simulated_galaxy")


def test_photometry_and_morphology(
    sample_catalogue,
    temp_output_dir,
):
    """Test the photometry and morphology extraction methods."""
    bands = [f"pau_nb{x}" for x in np.arange(455, 855, 10)]

    # Create test catalogue file
    catalogue_path = str(Path(temp_output_dir) / "sample_catalogue.parquet")
    sample_catalogue.to_parquet(catalogue_path)

    simulator = CreateSimulatedImages(
        catalogue=str(Path(temp_output_dir) / "sample_catalogue.parquet"),
        Ngals=2,
        bands=bands,
        add_poisson=True,
        add_psf=True,
        add_constant_background=True,
        use_dask=False,
        num_exposures=1,
        output_dir=temp_output_dir,
    )

    photometry = simulator._get_photometry(simulator.catalogue)
    morphology = simulator._get_morphology(simulator.catalogue)

    assert photometry is not None
    assert morphology is not None


"""def test_output_files_creation(sample_catalogue, temp_output_dir, mock_pix_scales, mock_psfs, mock_exp_times):
    bands = [f'pau_nb{x}' for x in np.arange(455, 855, 10)]
    n_gals = 1  # Testing with just one galaxy
    n_exposures = 1

    # Create test catalogue with just one galaxy to avoid broadcasting issues
    one_galaxy_catalogue = sample_catalogue.iloc[[0]].copy()  # Take just the first galaxy

    # Create test catalogue file
    one_galaxy_catalogue.to_parquet(str(Path(temp_output_dir) / "test_catalogue.parquet"))

    simulator = CreateSimulatedImages(
        catalogue=str(Path(temp_output_dir) / "test_catalogue.parquet"),
        Ngals=n_gals,
        bands=bands,
        add_poisson=True,
        add_psf=True,
        add_constant_background=True,
        use_dask=False,
        num_exposures=n_exposures,
        output_dir=temp_output_dir,
    )

    # Set the mock scales and PSFs
    simulator.pix_scales = mock_pix_scales
    simulator.psfs = mock_psfs
    simulator.exp_time = mock_exp_times
    # Create one galaxy
    photometry = simulator._get_photometry(simulator.catalogue)
    morphology = simulator._get_morphology(simulator.catalogue)

    print(photometry.shape, morphology.shape)

    for ii in range(n_gals):
        for band in bands:
            for exp in range(n_exposures):
                simulator._create_simulated_galaxy(ii, band, exp, photometry, morphology, zp=2)

                # Check if files exist
                cutout_file = Path(temp_output_dir) / f'data_{ii}' / f'cutout_{band}_exp{exp}.npy'
                metadata_file = Path(temp_output_dir) / f'data_{ii}' / f'metadata_{band}_exp{exp}.npy'

                assert cutout_file.exists()
                assert metadata_file.exists()"""


def test_invalid_inputs():
    """Test that invalid inputs raise appropriate exceptions."""
    with pytest.raises(Exception):
        # Empty catalogue should raise an error
        CreateSimulatedImages(
            catalogue=pd.DataFrame(),
            Ngals=1,
            bands=[],
            add_poisson=True,
            add_psf=True,
            add_constant_background=True,
            use_dask=False,
            num_exposures=1,
            output_dir="test",
        )
