from __future__ import annotations

"""Strict leave-one-recording-out BiLSTM diagnostic for missing-value holes.

This script is validation-only. It uses test-mask geometry to create synthetic
gaps in other recordings, removes every held-out target from its release table,
and never writes a submission candidate.
"""

from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
if (ROOT / "data" / "raw" / "train.csv").is_file():
    DATA = ROOT / "data" / "raw"
else:
    DATA = ROOT / "data_madinah_trial"
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

MAPPING = {
    "20260918_1100": "20260911_1100",
    "20260919_1100": "20260912_1100",
}
KEYS = ["recording", "unit", "minute_of_day"]
CONTEXT = 18
SEQ_LEN = 56
MAX_GAP = SEQ_LEN - 2 * CONTEXT
N_SYNTHETIC_PER_SERIES = 60
EPOCHS = 20
BATCH_SIZE = 64
SEED = 3401


def normalize_keys(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["recording"] = frame["recording"].astype(str)
    frame["unit"] = frame["unit"].astype(str)
    frame["minute_of_day"] = pd.to_numeric(
        frame["minute_of_day"], errors="raise"
    ).astype(int)
    return frame


def contiguous_runs(values: np.ndarray) -> list[tuple[int, int]]:
    values = sorted(set(int(x) for x in values))
    if not values:
        return []
    result: list[tuple[int, int]] = []
    start = previous = values[0]
    for value in values[1:]:
        if value != previous + 1:
            result.append((start, previous))
            start = value
        previous = value
    result.append((start, previous))
    return result


def prepare(train: pd.DataFrame, test: pd.DataFrame):
    train = normalize_keys(train)
    test = normalize_keys(test)
    train["people"] = pd.to_numeric(train["people"], errors="raise")
    test["people"] = pd.to_numeric(test["people"], errors="coerce")

    mapped = test.copy()
    mapped["recording"] = mapped["recording"].map(MAPPING)
    if mapped["recording"].isna().any():
        raise ValueError("Found a test recording without a validation mapping")

    target_rows = mapped.loc[mapped["id"].notna() & mapped["people"].isna()].copy()
    target_keys = set(map(tuple, target_rows[KEYS].to_numpy()))
    visible_keys = set(
        map(tuple, mapped.loc[mapped["people"].notna(), KEYS].to_numpy())
    )
    in_validation = train["recording"].isin(MAPPING.values()).to_numpy()
    train_keys = list(map(tuple, train[KEYS].to_numpy()))
    keep = ~in_validation | np.asarray([key in visible_keys for key in train_keys])
    release = train.loc[keep].copy().reset_index(drop=True)
    release_keys = set(map(tuple, release[KEYS].to_numpy()))
    if target_keys & release_keys:
        raise ValueError("A hidden validation target remains in the release table")

    missing_targets = target_rows.loc[target_rows["kind"].astype(str).eq("missing"), KEYS]
    gold = missing_targets.merge(
        train[KEYS + ["people"]], on=KEYS, how="left", validate="one_to_one"
    )
    gold = gold.dropna(subset=["people"]).rename(columns={"people": "truth"})
    gold = gold.reset_index(drop=True)
    if len(gold) != 358:
        raise ValueError(f"Expected 358 mapped missing labels, found {len(gold)}")

    templates: dict[str, list[tuple[int, int, int]]] = {}
    run_lookup: dict[tuple[str, str, int], tuple[int, int]] = {}
    for (rec, unit), group in missing_targets.groupby(["recording", "unit"], sort=True):
        minutes = group["minute_of_day"].to_numpy(int)
        for start, end in contiguous_runs(minutes):
            length = end - start + 1
            for minute in minutes[(minutes >= start) & (minutes <= end)]:
                run_lookup[(str(rec), str(unit), int(minute))] = (start, end)
                templates.setdefault(str(unit), []).append(
                    (int(minute), int(length), int(minute - start))
                )
    return train, release, gold, templates, run_lookup


class RecordingGrid:
    def __init__(self, frame: pd.DataFrame, units: list[str]):
        self.recording = str(frame["recording"].iloc[0])
        self.units = units
        self.unit_index = {unit: i for i, unit in enumerate(units)}
        self.times = np.sort(frame["minute_of_day"].unique().astype(int))
        self.first = int(self.times[0])
        self.last = int(self.times[-1])
        self.values = np.full((self.last - self.first + 1, len(units)), np.nan)
        for row in frame.itertuples(index=False):
            unit = str(row.unit)
            if unit in self.unit_index and np.isfinite(float(row.people)):
                self.values[int(row.minute_of_day) - self.first, self.unit_index[unit]] = float(row.people)

    def at(self, minute: int) -> np.ndarray:
        if minute < self.first or minute > self.last:
            return np.full(len(self.units), np.nan)
        return self.values[minute - self.first].copy()

    def target_bounds(self, unit: str, start: int, end: int):
        j = self.unit_index[unit]
        left_stop = max(0, start - self.first)
        left_candidates = np.flatnonzero(np.isfinite(self.values[:left_stop, j]))
        right_start = max(0, end + 1 - self.first)
        right_candidates = np.flatnonzero(np.isfinite(self.values[right_start:, j]))
        if not len(left_candidates) or not len(right_candidates):
            return None
        li = int(left_candidates[-1])
        ri = int(right_start + right_candidates[0])
        lt, rt = self.first + li, self.first + ri
        if not (lt < start <= end < rt) or rt - lt > 60:
            return None
        return lt, rt, float(self.values[li, j]), float(self.values[ri, j])


def make_window(
    grid: RecordingGrid,
    unit: str,
    start: int,
    end: int,
    scales: np.ndarray,
):
    length = end - start + 1
    if length < 1 or length > MAX_GAP:
        return None
    bounds = grid.target_bounds(unit, start, end)
    if bounds is None:
        return None
    left_minute, right_minute, left_value, right_value = bounds
    target_idx = grid.unit_index[unit]

    minutes = np.arange(start - CONTEXT, start - CONTEXT + SEQ_LEN, dtype=int)
    raw = np.vstack([grid.at(int(minute)) for minute in minutes])
    observed = np.isfinite(raw)
    gap_positions = np.arange(CONTEXT, CONTEXT + length)
    gap_minutes = minutes[gap_positions]
    baseline = np.interp(
        gap_minutes,
        [left_minute, right_minute],
        [left_value, right_value],
    )
    raw[gap_positions, target_idx] = baseline
    observed[gap_positions, target_idx] = False

    # Per-camera robust scaling keeps large views from dominating small ones.
    safe_raw = np.where(np.isfinite(raw), np.maximum(raw, 0.0), 0.0)
    normalized = np.arcsinh(safe_raw / scales[None, :])
    # Missing peers use zero level; the observation mask carries their status.
    normalized[~np.isfinite(raw)] = 0.0

    tod = minutes.astype(float) % 1440
    time_features = np.column_stack(
        [np.sin(2 * np.pi * tod / 1440), np.cos(2 * np.pi * tod / 1440)]
    )
    gap_flag = np.zeros(SEQ_LEN, dtype=float)
    gap_flag[gap_positions] = 1.0
    relative_position = np.zeros(SEQ_LEN, dtype=float)
    relative_position[gap_positions] = (np.arange(length) + 0.5) / length
    unit_onehot = np.zeros((SEQ_LEN, len(grid.units)), dtype=float)
    unit_onehot[:, target_idx] = 1.0
    x = np.concatenate(
        [normalized, observed.astype(float), time_features,
         gap_flag[:, None], relative_position[:, None], unit_onehot],
        axis=1,
    ).astype(np.float32)
    return x, gap_positions, baseline, raw


class BiLSTMResidual(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 32):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        h, _ = self.rnn(x)
        return self.head(h).squeeze(-1)


def synthetic_examples(
    release: pd.DataFrame,
    holdout: str,
    templates: dict[str, list[tuple[int, int, int]]],
    units: list[str],
    scales: np.ndarray,
    seed: int,
):
    rng = np.random.default_rng(seed)
    examples_x, examples_y, examples_mask = [], [], []
    for rec, frame in release.groupby("recording", sort=True):
        rec = str(rec)
        if rec == holdout:
            continue
        grid = RecordingGrid(frame, units)
        for unit in units:
            options = templates.get(unit, [])
            if not options or unit not in grid.unit_index:
                continue
            source_values = grid.values[:, grid.unit_index[unit]]
            candidates = np.flatnonzero(np.isfinite(source_values))
            if len(candidates) < 40:
                continue
            accepted = 0
            for _ in range(N_SYNTHETIC_PER_SERIES * 8):
                if accepted >= N_SYNTHETIC_PER_SERIES:
                    break
                target_minute, gap_length, offset = options[int(rng.integers(len(options)))]
                nearest = int(np.searchsorted(grid.times, target_minute))
                possible = [i for i in (nearest - 1, nearest) if 0 <= i < len(grid.times)]
                if not possible:
                    continue
                minute = int(grid.times[min(possible, key=lambda i: abs(grid.times[i] - target_minute))])
                if abs(minute - target_minute) > 15:
                    continue
                start = minute - offset
                end = start + gap_length - 1
                made = make_window(grid, unit, start, end, scales)
                if made is None:
                    continue
                x, positions, baseline, _ = made
                minutes = np.arange(start, end + 1)
                truth = np.asarray([grid.at(int(t))[grid.unit_index[unit]] for t in minutes])
                if not np.isfinite(truth).all():
                    continue
                target = np.zeros(SEQ_LEN, dtype=np.float32)
                mask = np.zeros(SEQ_LEN, dtype=bool)
                target[positions] = np.clip((truth - baseline) / scales[grid.unit_index[unit]], -2.0, 2.0)
                mask[positions] = True
                examples_x.append(x)
                examples_y.append(target)
                examples_mask.append(mask)
                accepted += 1
    if len(examples_x) < 80:
        raise ValueError(f"Only {len(examples_x)} synthetic sequences for {holdout}")
    return (
        np.stack(examples_x),
        np.stack(examples_y),
        np.stack(examples_mask),
    )


def fit_model(x: np.ndarray, y: np.ndarray, mask: np.ndarray, seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.set_num_threads(1)
    device = torch.device("cpu")
    model = BiLSTMResidual(x.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0015, weight_decay=1e-4)
    tx = torch.from_numpy(x).to(device)
    ty = torch.from_numpy(y).to(device)
    tm = torch.from_numpy(mask).to(device)
    rng = np.random.default_rng(seed)
    model.train()
    for epoch in range(EPOCHS):
        order = rng.permutation(len(tx))
        for start in range(0, len(order), BATCH_SIZE):
            idx = torch.as_tensor(order[start : start + BATCH_SIZE], dtype=torch.long)
            pred = model(tx[idx])
            active = tm[idx]
            loss = torch.abs(pred[active] - ty[idx][active]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return model.eval()


def main():
    torch.set_num_threads(1)
    train_path, test_path = DATA / "train.csv", DATA / "test.csv"
    if not train_path.is_file() or not test_path.is_file():
        raise SystemExit(f"Missing input files under {DATA}")
    train, release, gold, templates, run_lookup = prepare(
        pd.read_csv(train_path), pd.read_csv(test_path)
    )
    units = sorted(set(train.unit.astype(str)) & set(release.unit.astype(str)))
    if not set(gold.unit.astype(str)).issubset(units):
        raise ValueError("Validation has a camera without released training history")
    baseline_report = pd.read_csv(REPORTS / "v31_pchip_missing_oof.csv")
    aligned = gold.merge(
        baseline_report[KEYS + ["truth", "linear", "pchip"]],
        on=KEYS,
        how="left",
        suffixes=("", "_saved"),
        validate="one_to_one",
    )
    if aligned[["truth_saved", "linear", "pchip"]].isna().any().any():
        raise ValueError("Could not align exact-mask linear/PCHIP baselines")
    if not np.allclose(aligned.truth, aligned.truth_saved, rtol=0, atol=1e-6):
        raise ValueError("Saved missing OOF truths do not match this dataset")
    out = aligned[KEYS + ["truth", "linear", "pchip"]].copy()
    for name in ("bilstm", "blend_010", "blend_025", "blend_050"):
        out[name] = np.nan

    fold_metadata = []
    for heldout in sorted(out.recording.astype(str).unique()):
        fold = out.recording.astype(str).eq(heldout).to_numpy()
        fit_rows = release.loc[release.recording.astype(str).ne(heldout)]
        scale = np.ones(len(units), dtype=float)
        global_med = max(10.0, float(fit_rows.people.median()))
        for j, unit in enumerate(units):
            values = fit_rows.loc[fit_rows.unit.astype(str).eq(unit), "people"].to_numpy(float)
            scale[j] = max(10.0, float(np.median(values))) if len(values) else global_med
        sx, sy, sm = synthetic_examples(
            release, heldout, templates, units, scale, SEED + int(heldout[-4:])
        )
        model = fit_model(sx, sy, sm, SEED + int(heldout[-4:]))
        grid = RecordingGrid(release.loc[release.recording.astype(str).eq(heldout)], units)
        fold_rows = out.loc[fold]
        run_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for idx, row in fold_rows.iterrows():
            rec, unit, minute = str(row.recording), str(row.unit), int(row.minute_of_day)
            start, end = run_lookup[(rec, unit, minute)]
            key = (unit, str(start))
            if key not in run_cache:
                made = make_window(grid, unit, start, end, scale)
                if made is None:
                    raise ValueError(f"No valid validation context for {rec}/{unit}/{start}-{end}")
                x, positions, base, _ = made
                with torch.no_grad():
                    residual = model(torch.from_numpy(x[None, ...])).numpy()[0, positions]
                prediction = np.maximum(
                    0.0,
                    base + residual * scale[grid.unit_index[unit]],
                )
                run_cache[key] = (positions, base, prediction)
            positions, base, prediction = run_cache[key]
            offset = minute - start
            local = int(np.flatnonzero(positions == CONTEXT + offset)[0])
            p = float(prediction[local])
            base_value = float(base[offset])
            out.loc[idx, "bilstm"] = p
            out.loc[idx, "blend_010"] = max(0.0, base_value + 0.10 * (p - base_value))
            out.loc[idx, "blend_025"] = max(0.0, base_value + 0.25 * (p - base_value))
            out.loc[idx, "blend_050"] = max(0.0, base_value + 0.50 * (p - base_value))
        fold_metadata.append((heldout, len(sx)))
        print(f"Fold {heldout}: synthetic_sequences={len(sx)}; validation_rows={int(fold.sum())}")

    if out[["bilstm", "blend_010", "blend_025", "blend_050"]].isna().any().any():
        raise ValueError("Some OOF predictions were not filled")
    out["block"] = (out.minute_of_day // 180).astype(int)
    candidates = ["linear", "pchip", "bilstm", "blend_010", "blend_025", "blend_050"]
    rows = []
    for name in candidates:
        row = {"candidate": name, "rows": len(out), "mae": float(np.mean(np.abs(out.truth - out[name])))}
        for rec, group in out.groupby("recording"):
            row[f"mae_{rec}"] = float(np.mean(np.abs(group.truth - group[name])))
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values("mae")
    blocks = out.groupby(["recording", "block"])[["truth", "pchip", "bilstm", "blend_010", "blend_025", "blend_050"]].apply(
        lambda g: pd.Series({c: float(np.mean(np.abs(g.truth - g[c]))) for c in ["pchip", "bilstm", "blend_010", "blend_025", "blend_050"]})
    )

    REPORTS.mkdir(parents=True, exist_ok=True)
    prediction_path = REPORTS / "v34_bilstm_missing_oof.csv"
    summary_path = REPORTS / "v34_bilstm_missing_summary.csv"
    out.to_csv(prediction_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("\n=== V34 BiLSTM MISSING OOF ===")
    print(summary.round(5).to_string(index=False))
    print("\n3-hour block wins vs PCHIP:")
    for name in ("bilstm", "blend_010", "blend_025", "blend_050"):
        wins = int((blocks[name] <= blocks["pchip"]).sum())
        print(f"{name}: {wins}/{len(blocks)}")
    print("\nSaved diagnostics:", prediction_path, summary_path)
    print("STRICT MISSING OOF ONLY — NO TEST CANDIDATE OR KAGGLE UPLOAD")


if __name__ == "__main__":
    main()
