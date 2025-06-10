"""Procesamiento de datos con segmentacion y ruido.

Esta funcion lee los archivos originales, los divide en segmentos,
introduce ruido gaussiano y devuelve un DataFrame con una unica
observacion por intervalo de tiempo.
"""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd


def preparar_datos(
    ruido: float,
    frecuencia_observacion: float,
    periodo_orbital: float,
    numero_segmentos: int,
) -> pd.DataFrame:
    """Devuelve un DataFrame con una observacion por intervalo."""

    raw_files = [
        "data/original/mdot_series_q010_e000.txt",
        "data/original/mdot_series_q025_e000.txt",
        "data/original/mdot_series_q050_e000.txt",
        "data/original/mdot_series_q100_e000.txt",
        "data/original/mdot_series_q100_e050.txt",
        "data/original/mdot_series_q100_e040.txt",
        "data/original/mdot_series_q100_e030.txt",
        "data/original/mdot_series_q100_e020.txt",
        "data/original/mdot_series_q100_e010.txt",
        "data/original/mdot_series_q100_e060.txt",
        "data/original/mdot_series_q100_e070.txt",
    ]

    segments_dir = Path("data/segments")
    segments_dir.mkdir(parents=True, exist_ok=True)

    float_fmt = "%.18e"
    for raw in raw_files:
        raw_path = Path(raw)
        if not raw_path.exists():
            continue
        df_raw = pd.read_csv(raw_path, delim_whitespace=True, header=None)
        rows_per_seg = len(df_raw) // numero_segmentos
        for i in range(numero_segmentos):
            start = i * rows_per_seg
            end = (i + 1) * rows_per_seg if i < numero_segmentos - 1 else len(df_raw)
            seg_df = df_raw.iloc[start:end]
            seg_name = raw_path.stem + f"_seg{i+1:02d}" + raw_path.suffix
            seg_path = segments_dir / seg_name
            if not seg_path.exists():
                seg_df.to_csv(seg_path, sep=" ", header=False, index=False, float_format=float_fmt)

    pattern = "*_seg??.txt"
    col_time = "time"
    col_primary = "acre_rate_primary"
    col_second = "acre_rate_secondary"
    col_mw = "acre_rate_mass_weighted"

    def _parse(fname: str) -> tuple[float | None, float | None, int | None]:
        m = re.search(r"q(\d+)_e(\d+)_seg(\d+)", fname)
        if not m:
            return None, None, None
        q = int(m.group(1)) / 100.0
        e = int(m.group(2)) / 100.0
        s = int(m.group(3))
        return q, e, s

    frames: list[pd.DataFrame] = []
    for f in sorted(segments_dir.glob(pattern)):
        q_val, e_val, seg_id = _parse(f.name)
        if q_val is None:
            continue
        df = pd.read_csv(f, delim_whitespace=True, header=None, names=[col_time, col_primary, col_second])
        df["q"], df["e"], df["seg"] = q_val, e_val, seg_id
        w1 = 1 / (1 + q_val)
        w2 = q_val / (1 + q_val)
        df[col_mw] = w1 * df[col_primary].abs() + w2 * df[col_second].abs()
        q_tag = f"{int(round(q_val * 100)):03d}"
        e_tag = f"{int(round(e_val * 100)):03d}"
        seg_tag = f"{seg_id:02d}"
        df["id"] = f"{q_tag}_{e_tag}_{seg_tag}"
        frames.append(df)

    all_data = pd.concat(frames, ignore_index=True)
    all_data = all_data[~all_data["seg"].isin({1, 2})]

    rng = np.random.default_rng()
    sigma = all_data[col_mw].std(ddof=1)
    all_data[f"{col_mw}_clean"] = all_data[col_mw]
    noise = rng.normal(0.0, ruido * sigma, size=len(all_data))
    all_data[col_mw] = all_data[col_mw] + noise

    all_data["time_days"] = all_data[col_time] * periodo_orbital
    all_data["bucket"] = np.floor((all_data["time_days"] + 1e-9) / frecuencia_observacion).astype(int)

    unique_df = (
        all_data
        .groupby(["id", "bucket"], as_index=False)
        .median(numeric_only=True)
        .sort_values(["id", "time_days"])
        .reset_index(drop=True)
        .drop(columns="bucket")
    )

    return unique_df
