"""
MFPIT Pipeline — Step 3: GEE Authentication Check
==================================================
Tests Google Earth Engine connectivity using a Service Account JSON key.

CORRECTION vs original agent plan:
  - Added required `scopes` param for modern earthengine-api (≥0.1.370)
  - Added project ID extraction from JSON key (no hardcoding needed)
  - Added fallback to interactive auth if service account fails
  - Added a live GEE test query to confirm the connection is real

Usage:
    python gee/gee_auth_check.py

[CHECKPOINT ✓] printed on success.
[CHECKPOINT ✗] printed on failure with clear action steps.

SERVICE ACCOUNT JSON PATH — set below:
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

# ──────────────────────────────────────────────────────────────────
# USER MUST SET: Path to your service account JSON key file
# ──────────────────────────────────────────────────────────────────
# Search for JSON key robustly
SERVICE_ACCOUNT_JSON = Path("ee-muzzamil12-cbfdee54b77f.json")
if not SERVICE_ACCOUNT_JSON.exists():
    SERVICE_ACCOUNT_JSON = Path("../ee-muzzamil12-cbfdee54b77f.json")
if not SERVICE_ACCOUNT_JSON.exists():
    SERVICE_ACCOUNT_JSON = Path("MFPIT_Project/ee-muzzamil12-cbfdee54b77f.json")

# GEE OAuth scopes required by earthengine-api >= 0.1.370
# These MUST be present or authentication silently falls back to
# legacy credentials that fail on batch exports.
REQUIRED_SCOPES = [
    "https://www.googleapis.com/auth/earthengine",
    "https://www.googleapis.com/auth/devstorage.full_control",
    "https://www.googleapis.com/auth/drive",
]

STUDY_AREA_TEST = [67.0, 23.5, 69.5, 28.0]   # Quick sanity query

# ──────────────────────────────────────────────────────────────────

def load_service_account_info(json_path: Path) -> dict:
    """Load and validate the service account JSON key file."""
    if not json_path.exists():
        raise FileNotFoundError(
            f"Service account key not found at: {json_path.resolve()}\n"
            "ACTION: Copy your .json key file to the project root, or update\n"
            "        SERVICE_ACCOUNT_JSON path in this script."
        )
    with open(json_path) as f:
        info = json.load(f)

    required_fields = ["client_email", "private_key", "project_id"]
    for field in required_fields:
        if field not in info:
            raise ValueError(
                f"Service account JSON is missing field: '{field}'\n"
                "ACTION: Re-download the key from GCP Console → IAM → Service Accounts."
            )
    return info

def authenticate_service_account(sa_info: dict):
    """Attempt service account authentication with required scopes."""
    import ee
    from google.oauth2 import service_account

    credentials = service_account.Credentials.from_service_account_info(
        sa_info,
        scopes=REQUIRED_SCOPES,
    )
    # Initialize without the explicit project argument if it's causing 'not registered' issues
    try:
        ee.Initialize(credentials=credentials, project=sa_info["project_id"])
    except Exception as e:
        print(f"  [WARN] Project-specific initialization failed ({e}), falling back to legacy initialization...")
        ee.Initialize(credentials=credentials)
    return sa_info["project_id"]

def fallback_interactive_auth():
    """Fallback: interactive browser-based authentication."""
    import ee
    print("  [FALLBACK] Attempting interactive authentication...")
    print("  A browser window will open. Log in with your Google account.")
    ee.Authenticate()
    ee.Initialize()
    print("  [FALLBACK] Interactive auth succeeded.")

def run_live_test():
    """Run a real GEE query to prove the connection is functional."""
    import ee
    print("  Running live GEE test query on study area...")
    roi = ee.Geometry.Rectangle(STUDY_AREA_TEST)

    # Fetch a single CHIRPS value — lightweight, fast
    chirps = (
        ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
        .filterDate("2022-08-01", "2022-08-07")
        .filterBounds(roi)
        .first()
    )
    result = chirps.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=roi,
        scale=5566,
        maxPixels=1e8,
    ).getInfo()

    if not result or "precipitation" not in result:
        raise RuntimeError(
            "Live GEE test returned no data. "
            "Check that CHIRPS collection is accessible from your account."
        )

    precip_val = result["precipitation"]
    print(f"  Live test result: CHIRPS mean precipitation (Aug 1–7 2022) = {precip_val:.2f} mm")
    return precip_val

def update_progress_log(status: str, note: str):
    """Append a checkpoint entry to AGENT_PROGRESS_LOG.md."""
    log_path = Path("..") / "AGENT_PROGRESS_LOG.md"  # relative from gee/
    if not log_path.exists():
        log_path = Path("AGENT_PROGRESS_LOG.md")      # fallback if run from root
    if log_path.exists():
        with open(log_path, "a", encoding="utf-8") as f:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M")
            f.write(
                f"| 3 | GEE authentication | {status} | {ts} | — | — | {note} |\n"
            )

# ──────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("MFPIT Step 3: GEE Authentication Check")
    print("=" * 60)

    # 1. Load service account key
    try:
        sa_info = load_service_account_info(SERVICE_ACCOUNT_JSON)
        print(f"  [OK] Service account JSON loaded: {sa_info['client_email']}")
        print(f"  [OK] GCP project: {sa_info['project_id']}")
    except (FileNotFoundError, ValueError) as e:
        print(f"\n[CHECKPOINT ✗] GEE Auth FAILED at JSON loading.\nReason: {e}")
        sys.exit(1)

    # 2. Authenticate
    try:
        project_id = authenticate_service_account(sa_info)
        print(f"  [OK] Service account authenticated for project: {project_id}")
        auth_method = "service_account"
    except Exception as e:
        print(f"  [WARN] Service account auth failed: {e}")
        try:
            fallback_interactive_auth()
            auth_method = "interactive_fallback"
        except Exception as e2:
            print(f"\n[CHECKPOINT ✗] GEE Auth FAILED (both methods).\nReason: {e2}")
            print(
                "\nACTION REQUIRED:\n"
                "  1. Run: earthengine authenticate\n"
                "  2. Verify your service account has the 'Earth Engine Resource Viewer' role in GCP.\n"
                "  3. Check that the API is enabled: https://console.cloud.google.com/apis/library/earthengine.googleapis.com\n"
            )
            update_progress_log("✗ FAIL", "Both auth methods failed")
            sys.exit(1)

    # 3. Live query test
    try:
        precip = run_live_test()
        print(f"  [OK] Live query confirmed: {precip:.2f} mm")
    except Exception as e:
        print(f"\n[CHECKPOINT ✗] GEE authenticated but live query failed.\nReason: {e}")
        print(
            "  This usually means the Earth Engine API is not enabled for your project.\n"
            "  ACTION: Go to https://code.earthengine.google.com and register your project."
        )
        update_progress_log("✗ FAIL", f"Live query failed: {e}")
        sys.exit(1)

    # 4. Success
    update_progress_log("✓ PASS", f"auth={auth_method}, live_precip={precip:.2f}mm")

    print()
    print("[CHECKPOINT ✓] Step 3 completed. GEE is authenticated and live.")
    print(f"               Auth method : {auth_method}")
    print(f"               Project ID  : {sa_info['project_id']}")
    print(f"               Live test   : CHIRPS Aug 2022 = {precip:.2f} mm")
    print()
    print("NEXT STEP: Proceed to Step 4 (static dataset downloads).")
    print("           Run: python gee/gee_data_extractor.py --dataset static")

if __name__ == "__main__":
    # Ensure we can import from project root regardless of where script is run
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    main()
