# MFPIT Central Configuration

# Channel mapping ensures consistency across data extraction, statistics, and model prediction
CHANNEL_MAP = {
    "precip": 2,
    "et": 3,
    "soil": 6,
    "runoff": 7,
    "jrc_occurrence": 12 # Used for explicit leakage prevention
}
