import numpy as np

euroSAT_classes = {"AnnualCrop": 0, "Forest": 1, "HerbaceousVegetation": 2, "Highway": 3, "Industrial": 4, "Pasture": 5,
                   "PermanentCrop": 6, "Residential": 7, "River": 8, "SeaLake": 9}

INDICES_FUNCTIONS = {
    # Moisture Stress Index - assesses vegetation water content/stress
    "MSI": lambda bands: (bands["B8A"] - bands["B11"]) / (bands["B8A"] + bands["B11"] + 1e-6),

    # Normalized Difference Snow Index - used for snow and ice detection
    "NDSI": lambda bands: (bands["B03"] - bands["B11"]) / (bands["B03"] + bands["B11"] + 1e-6),

    # Normalized Difference Water Index - highlights water bodies
    "NDWI": lambda bands: (bands["B03"] - bands["B08"]) / (bands["B03"] + bands["B08"] + 1e-6),

    # Normalized Difference Vegetation Index - general vegetation health and density
    "NDVI": lambda bands: (bands["B08"] - bands["B04"]) / (bands["B08"] + bands["B04"] + 1e-6),

    # Green NDVI - vegetation stress, especially nitrogen concentration
    "GNDVI": lambda bands: (bands["B08"] - bands["B03"]) / (bands["B08"] + bands["B03"] + 1e-6),

    # Blue NDVI - used in aquatic vegetation and chlorophyll studies
    "BNDVI": lambda bands: (bands["B08"] - bands["B02"]) / (bands["B08"] + bands["B02"] + 1e-6),

    # Red Edge NDVI - detects chlorophyll content and plant stress
    "NDRE": lambda bands: (bands["B08"] - bands["B05"]) / (bands["B08"] + bands["B05"] + 1e-6),

    # Normalized Difference Chlorophyll Index - indicates chlorophyll concentration
    "NDCI": lambda bands: (bands["B05"] - bands["B04"]) / (bands["B05"] + bands["B04"] + 1e-6),

    # Normalized Burn Ratio - identifies burned areas post-fire
    "NBR": lambda bands: (bands["B08"] - bands["B12"]) / (bands["B08"] + bands["B12"] + 1e-6),

    # Normalized Difference Built-up Index - highlights urban/built-up areas
    "NDBI": lambda bands: (bands["B11"] - bands["B08"]) / (bands["B08"] + bands["B11"] + 1e-6),

    # Modified Soil Adjusted Vegetation Index - enhances vegetation signal in soil-heavy areas
    "MSAVI": lambda bands: (2 * bands["B08"] + 1 - np.sqrt(
        (2 * bands["B08"] + 1) ** 2 - 8 * (bands["B08"] - bands["B04"]))) / 2,

    # Enhanced Vegetation Index - improves sensitivity in high biomass areas
    "EVI": lambda bands: 2.5 * (bands["B08"] - bands["B04"]) / (
            bands["B08"] + 6 * bands["B04"] - 7.5 * bands["B02"] + 1 + 1e-6),

    # Soil Adjusted Vegetation Index - minimizes soil brightness influence
    "SAVI": lambda bands, L=0.5: ((bands["B08"] - bands["B04"]) / (bands["B08"] + bands["B04"] + L + 1e-6)) * (1 + L),

    # Normalized Difference Moisture Index - detects vegetation moisture
    "NDMI": lambda bands: (bands["B08"] - bands["B11"]) / (bands["B08"] + bands["B11"] + 1e-6),

    # Anthocyanin Reflectance Index - used to estimate anthocyanin pigments in leaves
    "ARI": lambda bands: (1 / (bands["B03"] + 1e-6)) - (1 / (bands["B05"] + 1e-6)),

    # Built-Up Index - composite index to highlight urbanization vs vegetation
    "BUI": lambda bands, L=0.5: ((bands["B11"] - bands["B08"]) / (bands["B11"] + bands["B08"] + 1e-6) -
                                 (((bands["B08"] - bands["B04"]) * (1 + L)) / (bands["B08"] + bands["B04"] + L + 1e-6) +
                                  (bands["B03"] - bands["B11"]) / (bands["B03"] + bands["B11"] + 1e-6)) / 2)
}


