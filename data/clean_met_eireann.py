
from pathlib import Path
import csv
import re

import numpy as np
import pandas as pd


# ============================================================
# 1. Configuration
# ============================================================

DATA_DIR = Path(
    r"D:\STUDY\CODE\PYTHON\AI FOR WEATHER AND CLIMATE"
    r"\AIFW-PRESENTATION-4\DATA"
)

OUTPUT_DIR = DATA_DIR / "processed"

# Inclusive UTC time range
START = "2022-01-01 00:00"
END = "2025-12-31 23:00"

# GNN input features
FEATURES = ["temp", "msl", "rhum", "rain"]

# "mask_111": treat flagged measurements as missing
# "keep_111": retain them but report quality warnings
QUALITY_MODE = "keep_111"


# ============================================================
# 2. Extract station metadata
# ============================================================

def get_metadata(lines, filename):

    preamble = "".join(lines)

    def match(pattern):
        result = re.search(
            pattern,
            preamble,
            flags=re.IGNORECASE
        )
        return result.group(1).strip() if result else None

    def as_float(value):
        return float(value) if value is not None else np.nan

    return {
        "station": filename.stem,

        "official_name": match(
            r"Station\s+Name\s*:\s*([^\r\n]+)"
        ),

        "latitude": as_float(
            match(r"Latitude\s*:\s*([-+]?\d+(?:\.\d+)?)")
        ),

        "longitude": as_float(
            match(r"Longitude\s*:\s*([-+]?\d+(?:\.\d+)?)")
        ),

        "height_m": as_float(
            match(r"Station\s+Height\s*:\s*([-+]?\d+(?:\.\d+)?)")
        )
    }


# ============================================================
# 3. Read one Met Eireann station CSV
# ============================================================

def load_csv(path):

    with path.open(
        "r",
        encoding="utf-8-sig",
        errors="replace"
    ) as fh:
        lines = fh.readlines()

    # Find the actual CSV header
    header_at = next(
        (
            i for i, line in enumerate(lines)
            if re.match(
                r"^\s*date\s*,",
                line,
                flags=re.IGNORECASE
            )
        ),
        None
    )

    if header_at is None:
        raise ValueError(
            f"{path.name}: cannot locate CSV header"
        )

    # Read column names
    raw_header = [
        x.strip().lower()
        for x in next(csv.reader([lines[header_at]]))
    ]

    # Rename repeated indicator columns
    # Example:
    # ind,rain,ind,temp
    # becomes:
    # ind_rain,rain,ind_temp,temp

    columns = []

    for i, col in enumerate(raw_header):

        if col == "ind":
            next_field = (
                raw_header[i + 1]
                if i + 1 < len(raw_header)
                else str(i)
            )
            col = f"ind_{next_field}"

        columns.append(col)

    if len(columns) != len(set(columns)):
        raise ValueError(
            f"{path.name}: duplicate columns"
        )

    # Load observations
    df = pd.read_csv(
        path,
        skiprows=header_at,
        header=0,
        names=columns,
        dtype=str,
        skipinitialspace=True,
        keep_default_na=False,
        encoding="utf-8-sig",
        encoding_errors="replace",
        low_memory=False
    )

    # Verify necessary variables
    required = ["date"] + FEATURES
    missing = sorted(set(required) - set(df.columns))

    if missing:
        raise ValueError(
            f"{path.name}: missing fields {missing}"
        )

    # --------------------------------------------------------
    # Parse UTC timestamps
    # --------------------------------------------------------

    raw_dates = df["date"].astype(str).str.strip()

    parsed = pd.to_datetime(
        raw_dates,
        format="%d-%b-%Y %H:%M",
        utc=True,
        errors="coerce"
    )

    if parsed.isna().any():

        fallback = pd.to_datetime(
            raw_dates,
            dayfirst=True,
            utc=True,
            errors="coerce"
        )

        parsed = parsed.fillna(fallback)

    if parsed.isna().any():
        samples = raw_dates.loc[
            parsed.isna()
        ].head(3).tolist()

        raise ValueError(
            f"{path.name}: invalid dates {samples}"
        )

    # Check duplicate timestamps
    if parsed.duplicated().any():
        raise ValueError(
            f"{path.name}: duplicate timestamps"
        )

    # Check hourly timestamps
    if (
        (parsed.dt.minute != 0).any()
        or (parsed.dt.second != 0).any()
    ):
        raise ValueError(
            f"{path.name}: non-hourly timestamps"
        )

    df["date"] = parsed

    # --------------------------------------------------------
    # Convert meteorological variables to numeric
    # --------------------------------------------------------

    numeric_fields = FEATURES + [
        "ind_rain",
        "ind_temp"
    ]

    for field in numeric_fields:

        if field in df.columns:

            df[field] = pd.to_numeric(
                df[field]
                .astype(str)
                .str.strip()
                .replace("", np.nan),
                errors="coerce"
            )

        else:
            df[field] = np.nan

    # Extract geographic information
    metadata = get_metadata(
        lines[:header_at],
        path
    )

    # Keep selected variables and quality indicators
    df = df[
        [
            "date",
            *FEATURES,
            "ind_rain",
            "ind_temp"
        ]
    ]

    df = df.set_index("date").sort_index()

    return df, metadata


# ============================================================
# 4. Batch process and align all stations
# ============================================================

def main():

    if QUALITY_MODE not in {"mask_111", "keep_111"}:
        raise ValueError("Invalid QUALITY_MODE")

    if not DATA_DIR.exists():
        raise FileNotFoundError(DATA_DIR)

    OUTPUT_DIR.mkdir(exist_ok=True)

    # Read only CSV files in DATA directory
    csv_files = sorted(DATA_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError("No CSV files found")

    # --------------------------------------------------------
    # Unified hourly UTC timeline
    # --------------------------------------------------------

    hours = pd.date_range(
        pd.Timestamp(START, tz="UTC"),
        pd.Timestamp(END, tz="UTC"),
        freq="h",
        name="date"
    )

    station_ids = []

    all_clean = []
    all_raw = []

    metadata_list = []
    quality_reports = []

    # --------------------------------------------------------
    # Process each station
    # --------------------------------------------------------

    for file in csv_files:

        print(f"\nProcessing: {file.name}")

        original, metadata = load_csv(file)

        station = file.stem

        station_ids.append(station)
        metadata_list.append(metadata)

        # Select required time period
        in_period = original.loc[
            (original.index >= hours[0])
            & (original.index <= hours[-1])
        ]

        # Align to the same hourly timeline
        aligned = in_period.reindex(hours)

        raw = aligned.copy()
        clean = aligned.copy()

        # ----------------------------------------------------
        # Quality indicator handling
        # ----------------------------------------------------

        # Temperature:
        # ind_temp = 4 means unavailable
        clean.loc[
            clean["ind_temp"] == 4,
            "temp"
        ] = np.nan

        # Rainfall:
        # ind_rain = 8 means unavailable
        clean.loc[
            clean["ind_rain"] == 8,
            "rain"
        ] = np.nan

        # 111 means investigation required
        if QUALITY_MODE == "mask_111":

            clean.loc[
                clean["ind_temp"] == 111,
                "temp"
            ] = np.nan

            clean.loc[
                clean["ind_rain"] == 111,
                "rain"
            ] = np.nan

        # ----------------------------------------------------
        # Physical validity checks
        # ----------------------------------------------------

        # Relative humidity should be between 0 and 100
        invalid_rhum = (
            ~clean["rhum"].between(0, 100)
            & clean["rhum"].notna()
        )

        clean.loc[
            invalid_rhum,
            "rhum"
        ] = np.nan

        # Rainfall cannot be negative
        invalid_rain = (
            (clean["rain"] < 0)
            & clean["rain"].notna()
        )

        clean.loc[
            invalid_rain,
            "rain"
        ] = np.nan

        # ----------------------------------------------------
        # Data quality report
        # ----------------------------------------------------

        report = {
            "station": station,

            "observed_hours": len(in_period),

            "expected_hours": len(hours),

            "missing_timestamp_hours": (
                len(hours) - len(in_period)
            ),

            "temp_ind_111_hours": int(
                (raw["ind_temp"] == 111).sum()
            ),

            "rain_ind_111_hours": int(
                (raw["ind_rain"] == 111).sum()
            )
        }

        for variable in FEATURES:

            report[
                f"{variable}_raw_missing_pct"
            ] = round(
                100 * raw[variable].isna().mean(),
                2
            )

            report[
                f"{variable}_clean_missing_pct"
            ] = round(
                100 * clean[variable].isna().mean(),
                2
            )

        quality_reports.append(report)

        # ----------------------------------------------------
        # Add station name and save to lists
        # ----------------------------------------------------

        for frame, target_list in [
            (raw, all_raw),
            (clean, all_clean)
        ]:

            one = frame.reset_index()

            one.insert(
                1,
                "station",
                station
            )

            target_list.append(one)

        print(
            f"Rows: {len(in_period)} | "
            f"Temp missing: "
            f"{report['temp_clean_missing_pct']:.2f}%"
        )

        if report["temp_clean_missing_pct"] >= 30:
            print(
                "WARNING: High temperature missing rate!"
            )

        if (
            np.isnan(metadata["latitude"])
            or np.isnan(metadata["longitude"])
        ):
            print(
                "WARNING: Geographic coordinates missing!"
            )

    # ========================================================
    # 5. Merge data into Long Format
    # ========================================================

    long_raw = pd.concat(
        all_raw,
        ignore_index=True
    ).sort_values(
        ["date", "station"]
    )

    long_clean = pd.concat(
        all_clean,
        ignore_index=True
    ).sort_values(
        ["date", "station"]
    )

    long_raw.to_csv(
        OUTPUT_DIR / "weather_long_with_flags.csv",
        index=False
    )

    long_clean.to_csv(
        OUTPUT_DIR / "weather_long.csv",
        index=False
    )

    # ========================================================
    # 6. Convert to Wide Format
    # ========================================================

    wide = long_clean.pivot(
        index="date",
        columns="station",
        values=FEATURES
    )

    # Guarantee consistent column order
    target_cols = pd.MultiIndex.from_product(
        [FEATURES, station_ids]
    )

    wide = wide.reindex(
        index=hours,
        columns=target_cols
    )

    # Flatten MultiIndex columns
    # Example: Dublin Airport 532__temp

    wide.columns = [
        f"{station}__{feature}"
        for feature, station in wide.columns
    ]

    wide.to_csv(
        OUTPUT_DIR / "weather_wide.csv",
        index_label="date"
    )

    # ========================================================
    # 7. Construct GNN Tensor
    # ========================================================

    # Shape:
    # [time, stations, features]

    tensor = np.stack(
        [
            wide[
                [f"{s}__{f}" for f in FEATURES]
            ].to_numpy(dtype=np.float32)

            for s in station_ids
        ],
        axis=1
    )

    np.save(
        OUTPUT_DIR / "weather_tensor.npy",
        tensor
    )

    # ========================================================
    # 8. Save timestamps and station ordering
    # ========================================================

    pd.DataFrame(
        {"date": hours}
    ).to_csv(
        OUTPUT_DIR / "timestamps.csv",
        index=False
    )

    pd.DataFrame(
        {
            "station_idx": range(len(station_ids)),
            "station": station_ids
        }
    ).to_csv(
        OUTPUT_DIR / "station_order.csv",
        index=False
    )

    # ========================================================
    # 9. Save metadata and quality report
    # ========================================================

    pd.DataFrame(
        metadata_list
    ).to_csv(
        OUTPUT_DIR / "station_metadata.csv",
        index=False
    )

    pd.DataFrame(
        quality_reports
    ).to_csv(
        OUTPUT_DIR / "quality_report.csv",
        index=False
    )

    # ========================================================
    # 10. Summary
    # ========================================================

    print("\n====================================")
    print("Processing complete")
    print("====================================")

    print(f"Stations: {len(station_ids)}")
    print(f"Hours: {len(hours)}")
    print(f"Features: {len(FEATURES)}")
    print(f"Tensor shape: {tensor.shape}")

    print(f"\nQuality mode: {QUALITY_MODE}")
    print(f"Output directory: {OUTPUT_DIR}")

    print(
        "\nNaN values are retained intentionally."
        "\nDo not train GNN before checking missing data."
    )


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    main()
