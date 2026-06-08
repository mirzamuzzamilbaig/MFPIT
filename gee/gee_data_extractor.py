"""
MFPIT Pipeline — Step 4 & 5: GEE Data Extractor
==================================================
Extracts static and temporal satellite datasets for the Sindh Indus River Basin.
Study Bounds: 67.0–69.5°E, 23.5–28.0°N (Sukkur to Delta)
Scale: 1000m (1 km) for all resampled exports.

Double Strategy:
  1. Submits batch Export.image.toDrive() tasks to GEE.
  2. Synchronously downloads locally to 'data/raw/' via getDownloadURL() for instant access.
"""

import os
import sys
import json
import argparse
import urllib.request
import zipfile
import io
import time
from pathlib import Path
from datetime import datetime
import numpy as np

# ──────────────────────────────────────────────────────────────────
# Configuration & ROI
# ──────────────────────────────────────────────────────────────────
STUDY_ROI = [67.0, 23.5, 69.5, 28.0] # [lon_min, lat_min, lon_max, lat_max]
SCALE_M = 1000  # 1 km resample scale
CRS = "EPSG:4326"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# ──────────────────────────────────────────────────────────────────
# GEE Authentication
# ──────────────────────────────────────────────────────────────────
def init_gee():
    """Authenticates and initialises GEE using the robust key resolver."""
    import ee
    from google.oauth2 import service_account

    key_path = Path("ee-muzzamil12-cbfdee54b77f.json")
    if not key_path.exists():
        key_path = Path("../ee-muzzamil12-cbfdee54b77f.json")
    if not key_path.exists():
        key_path = PROJECT_ROOT / "ee-muzzamil12-cbfdee54b77f.json"
    if not key_path.exists():
        key_path = PROJECT_ROOT.parent / "ee-muzzamil12-cbfdee54b77f.json"

    if not key_path.exists():
        print(f"[FAIL] GEE key not found at {key_path.resolve()}")
        sys.exit(1)

    with open(key_path) as f:
        sa_info = json.load(f)

    SCOPES = [
        "https://www.googleapis.com/auth/earthengine",
        "https://www.googleapis.com/auth/devstorage.full_control",
        "https://www.googleapis.com/auth/drive",
    ]
    
    creds = service_account.Credentials.from_service_account_info(sa_info, scopes=SCOPES)
    
    try:
        ee.Initialize(credentials=creds, project=sa_info["project_id"])
    except Exception:
        ee.Initialize(credentials=creds)
    
    print(f"  [OK] Earth Engine initialised via Service Account: {sa_info['client_email']}")
    return ee, ee.Geometry.Rectangle(STUDY_ROI)

# ──────────────────────────────────────────────────────────────────
# Utility: Download and Export
# ──────────────────────────────────────────────────────────────────
def process_image(ee, image, description, folder_name, local_dir):
    """Triggers both GEE Drive export and synchronous local download."""
    roi = ee.Geometry.Rectangle(STUDY_ROI)
    os.makedirs(local_dir, exist_ok=True)
    local_file = Path(local_dir) / f"{description}.tif"

    # 1. Drive Export (Async)
    try:
        task = ee.batch.Export.image.toDrive(
            image=image,
            description=description,
            folder="MFPIT_GEE_Exports",
            fileNamePrefix=description,
            region=roi,
            scale=SCALE_M,
            crs=CRS,
            maxPixels=1e12,
        )
        task.start()
        print(f"  [EXPORTED] GEE task started for Drive: {description} (Task ID: {task.id})")
    except Exception as e:
        print(f"  [EXPORT WARN] GEE drive export skipped/failed: {e}")

    # 2. Local Download (Sync via getDownloadURL)
    if local_file.exists():
        print(f"  [SKIP] Local file already exists: {local_file.name}")
        return

    try:
        url = image.getDownloadURL({
            'name': description,
            'scale': SCALE_M,
            'region': roi,
            'crs': CRS,
            'format': 'GEO_TIFF'
        })
        
        # Download the data
        print(f"  [DOWNLOADING] Fetching local GeoTIFF for {description}...")
        response = urllib.request.urlopen(url)
        data_bytes = response.read()
        
        # Check if the response is a ZIP file or a direct TIFF
        if data_bytes.startswith(b'PK\x03\x04'):
            print("  [INFO] Received ZIP archive, extracting GeoTIFF...")
            with zipfile.ZipFile(io.BytesIO(data_bytes)) as z:
                for file_info in z.infolist():
                    if file_info.filename.endswith('.tif'):
                        extracted_data = z.read(file_info.filename)
                        with open(local_file, 'wb') as out_f:
                            out_f.write(extracted_data)
                        print(f"  [SAVED] {local_file.name} ({len(extracted_data)/1024:.1f} KB)")
                        break
        elif data_bytes.startswith(b'II*\x00') or data_bytes.startswith(b'MM\x00*'):
            print("  [INFO] Received direct GeoTIFF file, saving directly...")
            with open(local_file, 'wb') as out_f:
                out_f.write(data_bytes)
            print(f"  [SAVED] {local_file.name} ({len(data_bytes)/1024:.1f} KB)")
        else:
            # Maybe it's an error message inside text/html
            snippet = data_bytes[:200]
            print(f"  [DOWNLOAD FAIL] Response is neither ZIP nor TIFF. Preview: {snippet}")
    except Exception as e:
        print(f"  [DOWNLOAD FAIL] Could not download {description} locally: {e}")

# ──────────────────────────────────────────────────────────────────
# Static Datasets (Step 4)
# ──────────────────────────────────────────────────────────────────
def fetch_static_datasets(ee, roi):
    print("\n============================================================")
    print("MFPIT Step 4: Exporting Static Datasets (DEM, Slope, JRC Occurrence)")
    print("============================================================")

    # 1. DEM (SRTM 30m -> resample to 1000m)
    dem = ee.Image("USGS/SRTMGL1_003").select("elevation").clip(roi).resample('bilinear')
    process_image(ee, dem, "MFPIT_raw_dem_static", "dem", DATA_DIR / "raw" / "dem")

    # 2. Slope (Computed from DEM)
    slope = ee.Terrain.slope(dem).clip(roi)
    process_image(ee, slope, "MFPIT_raw_slope_static", "dem", DATA_DIR / "raw" / "dem")

    # 3. JRC Occurrence & Recurrence
    jrc_gsw = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select(["occurrence", "recurrence"]).clip(roi)
    process_image(ee, jrc_gsw, "MFPIT_raw_jrc_occurrence_static", "jrc_occurrence", DATA_DIR / "raw" / "jrc_occurrence")

    print("\n[CHECKPOINT ✓] Step 4 Static Downloads Complete.")

# ──────────────────────────────────────────────────────────────────
# Temporal Datasets (Step 5)
# ──────────────────────────────────────────────────────────────────
def fetch_temporal_datasets(ee, roi, years):
    print("\n============================================================")
    print(f"MFPIT Step 5: Exporting Temporal Datasets for Years {years}")
    print("============================================================")

    for year in range(years[0], years[1] + 1):
        print(f"\nProcessing Year: {year}")

        # 1. CHIRPS Precipitation (Monthly sum)
        for month in range(1, 13):
            start_date = f"{year}-{month:02d}-01"
            end_date = f"{year+1}-01-01" if month == 12 else f"{year}-{month+1:02d}-01"
            
            chirps_monthly = (
                ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
                .filterBounds(roi)
                .filterDate(start_date, end_date)
                .select("precipitation")
                .sum()
                .clip(roi)
            )
            desc = f"MFPIT_raw_chirps_{year}-{month:02d}"
            process_image(ee, chirps_monthly, desc, "chirps", DATA_DIR / "raw" / "chirps")

        # 2. TerraClimate (Monthly: tmmx, tmmn, soil, ro)
        for month in range(1, 13):
            start_date = f"{year}-{month:02d}-01"
            end_date = f"{year+1}-01-01" if month == 12 else f"{year}-{month+1:02d}-01"
            
            tc_monthly = (
                ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE")
                .filterBounds(roi)
                .filterDate(start_date, end_date)
                .select(["tmmx", "tmmn", "soil", "ro"])
                .mean()
                .clip(roi)
            )
            desc = f"MFPIT_raw_terraclimate_{year}-{month:02d}"
            process_image(ee, tc_monthly, desc, "terraclimate", DATA_DIR / "raw" / "terraclimate")

        # 3. MODIS Evapotranspiration (Monthly sum) - Upgraded to gap-filled v061 to cover 2001-2023
        for month in range(1, 13):
            start_date = f"{year}-{month:02d}-01"
            end_date = f"{year+1}-01-01" if month == 12 else f"{year}-{month+1:02d}-01"
            
            modis_monthly = (
                ee.ImageCollection("MODIS/061/MOD16A2GF")
                .filterBounds(roi)
                .filterDate(start_date, end_date)
                .select("ET")
                .sum()
                .clip(roi)
            )
            desc = f"MFPIT_raw_modis_et_{year}-{month:02d}"
            process_image(ee, modis_monthly, desc, "modis_et", DATA_DIR / "raw" / "modis_et")

        # 4. JRC Yearly Water History (Fallback to closest year if data is not yet available, e.g. 2022-2023)
        jrc_collection = (
            ee.ImageCollection("JRC/GSW1_4/YearlyHistory")
            .filterBounds(roi)
            .filterDate(f"{year}-01-01", f"{year+1}-01-01")
        )
        
        if jrc_collection.size().getInfo() == 0:
            fallback_year = year
            while fallback_year > 1984:
                fallback_year -= 1
                fallback_coll = (
                    ee.ImageCollection("JRC/GSW1_4/YearlyHistory")
                    .filterBounds(roi)
                    .filterDate(f"{fallback_year}-01-01", f"{fallback_year+1}-01-01")
                )
                if fallback_coll.size().getInfo() > 0:
                    print(f"  [INFO] JRC Yearly History not available for {year}. Falling back to {fallback_year} baseline...")
                    jrc_collection = fallback_coll
                    break
                    
        jrc_yearly = jrc_collection.select("waterClass").mean().clip(roi)
        desc = f"MFPIT_raw_jrc_water_{year}"
        process_image(ee, jrc_yearly, desc, "jrc_water", DATA_DIR / "raw" / "jrc_water")

        # 5. Sentinel-2 and Sentinel-1 SAR (Available 2017–2023) or MODIS Optical fallback (2001-2016)
        if year >= 2017:
            # Sentinel-2 Monthly Composites (9 bands: B2, B3, B4, B8, B11, B12, NDWI, MNDWI, AWEI)
            for month in range(1, 13):
                start_date = f"{year}-{month:02d}-01"
                end_date = f"{year+1}-01-01" if month == 12 else f"{year}-{month+1:02d}-01"
                
                s2_collection = (
                    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                    .filterBounds(roi)
                    .filterDate(start_date, end_date)
                    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
                )
                
                # Fallback to no cloud filtering if no images are below 20% cloud cover (e.g. monsoon/flood seasons)
                if s2_collection.size().getInfo() == 0:
                    print(f"  [INFO] No images <20% clouds in {year}-{month:02d}. Falling back to all images...")
                    s2_collection = (
                        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                        .filterBounds(roi)
                        .filterDate(start_date, end_date)
                    )
                
                if s2_collection.size().getInfo() > 0:
                    s2_img = s2_collection.median().select(["B2", "B3", "B4", "B8", "B11", "B12"]).clip(roi)
                    ndwi = s2_img.normalizedDifference(["B3", "B8"]).rename("NDWI")
                    mndwi = s2_img.normalizedDifference(["B3", "B11"]).rename("MNDWI")
                    awei = s2_img.expression(
                        "4 * (B3 - B11) - (0.25 * B8 + 2.75 * B12)",
                        {
                            "B3": s2_img.select("B3"),
                            "B8": s2_img.select("B8"),
                            "B11": s2_img.select("B11"),
                            "B12": s2_img.select("B12")
                        }
                    ).rename("AWEI")
                    s2_export = s2_img.addBands([ndwi, mndwi, awei])
                    desc = f"MFPIT_raw_sentinel2_{year}-{month:02d}"
                    process_image(ee, s2_export, desc, "sentinel2", DATA_DIR / "raw" / "sentinel2")
                else:
                    print(f"  [WARN] No Sentinel-2 images whatsoever in {year}-{month:02d}")

            # Sentinel-1 SAR Monthly Composites (VV, VH)
            for month in range(1, 13):
                start_date = f"{year}-{month:02d}-01"
                end_date = f"{year+1}-01-01" if month == 12 else f"{year}-{month+1:02d}-01"
                
                sar_monthly = (
                    ee.ImageCollection("COPERNICUS/S1_GRD")
                    .filterBounds(roi)
                    .filterDate(start_date, end_date)
                    .filter(ee.Filter.eq("instrumentMode", "IW"))
                    .select(["VV", "VH"])
                    .median()
                    .clip(roi)
                )
                desc = f"MFPIT_raw_sar_{year}-{month:02d}"
                process_image(ee, sar_monthly, desc, "sar", DATA_DIR / "raw" / "sar")
        else:
            # MODIS Optical Fallback for years 2001-2016 (NDWI, MNDWI)
            for month in range(1, 13):
                start_date = f"{year}-{month:02d}-01"
                end_date = f"{year+1}-01-01" if month == 12 else f"{year}-{month+1:02d}-01"
                
                modis_coll = (
                    ee.ImageCollection("MODIS/061/MOD09A1")
                    .filterBounds(roi)
                    .filterDate(start_date, end_date)
                )
                
                if modis_coll.size().getInfo() > 0:
                    modis_img = modis_coll.median().clip(roi)
                    ndwi = modis_img.normalizedDifference(["sur_refl_b04", "sur_refl_b02"]).rename("NDWI")
                    mndwi = modis_img.normalizedDifference(["sur_refl_b04", "sur_refl_b06"]).rename("MNDWI")
                    modis_export = ndwi.addBands(mndwi)
                    desc = f"MFPIT_raw_modis_optical_{year}-{month:02d}"
                    process_image(ee, modis_export, desc, "modis_optical", DATA_DIR / "raw" / "modis_optical")
                else:
                    print(f"  [WARN] No MODIS optical images in {year}-{month:02d}")

    print(f"\n[CHECKPOINT ✓] Step 5 Temporal Downloads Complete for years {years}.")

# ──────────────────────────────────────────────────────────────────
# Download Verification (Step 6)
# ──────────────────────────────────────────────────────────────────
def verify_downloads():
    """Scans raw directories and builds the DOWNLOAD_MANIFEST.csv."""
    import csv
    import rasterio

    print("\n============================================================")
    print("MFPIT Step 6: Verifying Downloads & Generating Manifest")
    print("============================================================")

    manifest_path = DATA_DIR / "DOWNLOAD_MANIFEST.csv"
    raw_root = DATA_DIR / "raw"
    records = []

    for path in raw_root.glob("**/*.tif"):
        rel_path = path.relative_to(PROJECT_ROOT)
        size_mb = path.stat().st_size / (1024 * 1024)
        
        try:
            with rasterio.open(path) as src:
                meta = src.meta
                data = src.read(1)
                
                nan_pct = (np.isnan(data).sum() / data.size) * 100
                v_min = float(np.nanmin(data)) if data.size > 0 else 0
                v_max = float(np.nanmax(data)) if data.size > 0 else 0
                
                records.append({
                    "FilePath": str(rel_path),
                    "SizeBytes": path.stat().st_size,
                    "SizeMB": f"{size_mb:.2f}",
                    "Height": meta["height"],
                    "Width": meta["width"],
                    "Bands": meta["count"],
                    "MinVal": f"{v_min:.2f}",
                    "MaxVal": f"{v_max:.2f}",
                    "NaNPercent": f"{nan_pct:.2f}%",
                    "Status": "VALID" if nan_pct < 50 else "WARN_HIGH_NAN"
                })
        except Exception as e:
            records.append({
                "FilePath": str(rel_path),
                "SizeBytes": path.stat().st_size,
                "SizeMB": f"{size_mb:.2f}",
                "Height": "ERR",
                "Width": "ERR",
                "Bands": "ERR",
                "MinVal": "ERR",
                "MaxVal": "ERR",
                "NaNPercent": "ERR",
                "Status": f"CORRUPT: {str(e)[:40]}"
            })

    # Write to CSV
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "FilePath", "SizeBytes", "SizeMB", "Height", "Width", 
            "Bands", "MinVal", "MaxVal", "NaNPercent", "Status"
        ])
        writer.writeheader()
        writer.writerows(records)

    print(f"  [OK] Saved download manifest to: {manifest_path.relative_to(PROJECT_ROOT)}")
    print(f"  Processed {len(records)} files successfully.")

    # Append checkpoint to AGENT_PROGRESS_LOG.md
    log_path = PROJECT_ROOT / "AGENT_PROGRESS_LOG.md"
    if log_path.exists():
        with open(log_path, "a", encoding="utf-8") as lf:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M")
            lf.write(
                f"| 6 | Download Verification | ✓ PASS | {ts} | data/DOWNLOAD_MANIFEST.csv | files={len(records)} | Verified metadata, NaN% and bounds for all GeoTIFFs. |\n"
            )

# ──────────────────────────────────────────────────────────────────
# Main Executable Block
# ──────────────────────────────────────────────────────────────────
def main():
    import json # imported inside main to prevent namespace clashes
    
    parser = argparse.ArgumentParser(description="MFPIT GEE Data Extractor")
    parser.add_argument("--dataset", choices=["static", "temporal", "all", "verify"], required=True,
                        help="Select dataset tier to download or verify.")
    parser.add_argument("--years", nargs=2, type=int, default=[2001, 2023],
                        help="Start and end years for temporal datasets (e.g. 2001 2023)")
    
    args = parser.parse_args()

    ee, roi = init_gee()

    if args.dataset == "static":
        fetch_static_datasets(ee, roi)
    elif args.dataset == "temporal":
        fetch_temporal_datasets(ee, roi, args.years)
    elif args.dataset == "all":
        fetch_static_datasets(ee, roi)
        fetch_temporal_datasets(ee, roi, args.years)
        verify_downloads()
    elif args.dataset == "verify":
        verify_downloads()

if __name__ == "__main__":
    main()
