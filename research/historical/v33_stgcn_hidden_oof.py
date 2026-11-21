from __future__ import annotations

"""Low-capacity graph-temporal neural baseline for the hidden-camera task.

Uses leave-one-recording-out validation. The target camera is masked from every
input window, matching the test condition where hidden cameras have no released
same-recording values. It writes OOF diagnostics only; no submission is created.
Requires PyTorch, NumPy and pandas.
"""

from pathlib import Path
import random
import warnings

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw"
if not DATA.exists():
    DATA = ROOT / "data_madinah_trial"
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)
OUT = REPORTS / "v33_stgcn_hidden_oof.csv"

TEST_TO_VALIDATION = {
    "20260918_1100": "20260911_1100",
    "20260919_1100": "20260912_1100",
}
TARGET_UNITS = {
    "20260911_1100": {"cam_B", "cam_G"},
    "20260912_1100": {"cam_D"},
}
UNITS = [f"cam_{c}" for c in "ABCDEFGH"]
WINDOW = 7
HALF_WINDOW = WINDOW // 2
EPOCHS = 22
BATCH_SIZE = 256
SEED = 42

# Current hidden OOF benchmark from the project's strict cross-fitted run.
REFERENCE_MAE = {
    "20260911_1100": 4.3616,
    "20260912_1100": 4.9494,
}


def _load() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    for frame in (train, test):
        frame["recording"] = frame["recording"].astype(str)
        frame["unit"] = frame["unit"].astype(str)
        frame["minute_of_day"] = pd.to_numeric(
            frame["minute_of_day"], errors="coerce"
        )
    train["people"] = pd.to_numeric(train["people"], errors="coerce")
    return train, test


def _panel_map(train: pd.DataFrame) -> dict[str, pd.DataFrame]:
    panels: dict[str, pd.DataFrame] = {}
    for rec, group in train.groupby("recording", sort=True):
        panel = group.pivot_table(
            index="minute_of_day", columns="unit", values="people", aggfunc="median"
        )
        panels[str(rec)] = panel.reindex(columns=UNITS)
    return panels


def _adjacency(train: pd.DataFrame, panels: dict[str, pd.DataFrame], fit_recs: list[str]) -> np.ndarray:
    """Estimate a sparse, positive residual-correlation graph from fit recordings."""
    residual_panels = []
    for rec in fit_recs:
        panel = panels[rec].copy()
        bins = (panel.index.to_numpy(int) // 30) * 30
        for unit in UNITS:
            values = panel[unit]
            baseline = values.groupby(bins).transform("median")
            panel[unit] = np.log1p(values) - baseline
        panel["_recording"] = rec
        residual_panels.append(
            panel.reset_index().set_index(["_recording", "minute_of_day"])
        )
    joined = pd.concat(residual_panels)
    corr = joined[UNITS].corr(min_periods=25).reindex(index=UNITS, columns=UNITS)
    corr = corr.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)

    adj = np.zeros_like(corr)
    for i in range(len(UNITS)):
        weights = np.maximum(corr[i], 0.0)
        weights[i] = 0.0
        top = np.argsort(weights)[-3:]
        weights = np.square(weights)
        selected = weights[top]
        if selected.sum() <= 1e-12:
            # A uniform peer fallback keeps the graph defined in sparse folds.
            peers = [j for j in range(len(UNITS)) if j != i]
            adj[i, peers] = 1.0 / len(peers)
        else:
            adj[i, top] = selected / selected.sum()
    return adj.astype(np.float32)


def _camera_scaling(train: pd.DataFrame, fit_recs: list[str]) -> tuple[np.ndarray, np.ndarray]:
    fit = train.loc[train["recording"].isin(fit_recs) & train["unit"].isin(UNITS)].copy()
    fit["log_people"] = np.log1p(fit["people"].clip(lower=0))
    medians = np.zeros(len(UNITS), dtype=np.float32)
    scales = np.ones(len(UNITS), dtype=np.float32)
    for i, unit in enumerate(UNITS):
        values = fit.loc[fit["unit"].eq(unit), "log_people"].dropna().to_numpy(float)
        if len(values):
            medians[i] = float(np.median(values))
            q25, q75 = np.quantile(values, [0.25, 0.75])
            scales[i] = max(float((q75 - q25) / 1.349), 0.30)
    return medians, scales


def _make_examples(
    train: pd.DataFrame,
    panels: dict[str, pd.DataFrame],
    recs: list[str],
    medians: np.ndarray,
    scales: np.ndarray,
    keep_keys: set[tuple[str, str, int]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[str, str, int, float]]]:
    """Create temporal windows; the label camera is always removed from inputs."""
    xs: list[np.ndarray] = []
    ys: list[float] = []
    cams: list[int] = []
    meta: list[tuple[str, str, int, float]] = []
    minute_grid = np.arange(0, 1441, dtype=int)
    for rec in recs:
        panel = panels[rec].reindex(minute_grid).reindex(columns=UNITS)
        raw = panel.to_numpy(float)
        log_values = np.log1p(np.maximum(raw, 0.0))
        normalized = (log_values - medians[None, :]) / scales[None, :]
        available = np.isfinite(raw)
        normalized = np.where(available, np.clip(normalized, -5.0, 5.0), 0.0)

        for cam_idx, unit in enumerate(UNITS):
            y_values = raw[:, cam_idx]
            indices = np.flatnonzero(np.isfinite(y_values))
            if keep_keys is not None:
                indices = np.asarray(
                    [m for m in indices if (rec, unit, int(m)) in keep_keys],
                    dtype=int,
                )
            if not len(indices):
                continue

            # [samples, window, cameras, (normalized_count, observed_flag)]
            batch = np.zeros(
                (len(indices), WINDOW, len(UNITS), 2), dtype=np.float32
            )
            for sample_i, minute in enumerate(indices):
                for step_i, delta in enumerate(range(-HALF_WINDOW, HALF_WINDOW + 1)):
                    source_minute = int(minute + delta)
                    if source_minute < 0 or source_minute >= len(minute_grid):
                        continue
                    mask = available[source_minute].copy()
                    # Exact hidden-camera policy: never expose the target camera.
                    mask[cam_idx] = False
                    batch[sample_i, step_i, :, 0] = np.where(
                        mask, normalized[source_minute], 0.0
                    )
                    batch[sample_i, step_i, :, 1] = mask.astype(np.float32)
            xs.append(batch)
            target_log = np.log1p(np.maximum(y_values[indices], 0.0))
            ys.extend(((target_log - medians[cam_idx]) / scales[cam_idx]).tolist())
            cams.extend([cam_idx] * len(indices))
            meta.extend(
                (rec, unit, int(m), float(y_values[m])) for m in indices
            )

    if not xs:
        return (
            np.empty((0, WINDOW, len(UNITS), 2), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int64),
            [],
        )
    return np.concatenate(xs), np.asarray(ys, np.float32), np.asarray(cams, np.int64), meta


def _fit_and_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    cam_train: np.ndarray,
    x_valid: np.ndarray,
    cam_valid: np.ndarray,
    time_valid: np.ndarray,
    adjacency: np.ndarray,
) -> np.ndarray:
    import torch
    from torch import nn

    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    class SmallSTGCN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.camera_embedding = nn.Embedding(len(UNITS), 8)
            self.graph_projection = nn.Linear(4, 24)
            self.graph_projection2 = nn.Linear(24, 16)
            self.head = nn.Sequential(
                nn.Linear(WINDOW * 16 + 8 + 2, 48),
                nn.ReLU(),
                nn.Dropout(0.15),
                nn.Linear(48, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
            )

        def forward(self, x, camera_idx, clock):
            # x: [batch, time, camera, 2]; adjacency aggregates spatial peers.
            msg = torch.einsum("ij,bwjd->bwid", adjacency_t, x)
            h = torch.relu(self.graph_projection(torch.cat([x, msg], dim=-1)))
            h = torch.relu(self.graph_projection2(h))
            batch_idx = torch.arange(len(camera_idx), device=camera_idx.device)
            node_first = h.permute(0, 2, 1, 3)
            target_nodes = node_first[batch_idx, camera_idx].reshape(len(camera_idx), -1)
            emb = self.camera_embedding(camera_idx)
            return self.head(torch.cat([target_nodes, emb, clock], dim=1)).squeeze(1)

    adjacency_t = torch.as_tensor(adjacency, dtype=torch.float32)
    device = torch.device("cpu")
    model = SmallSTGCN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0015, weight_decay=0.01)
    loss_fn = nn.L1Loss()

    x_t = torch.as_tensor(x_train, dtype=torch.float32, device=device)
    y_t = torch.as_tensor(y_train, dtype=torch.float32, device=device)
    c_t = torch.as_tensor(cam_train, dtype=torch.long, device=device)
    order = np.arange(len(y_train))

    for epoch in range(EPOCHS):
        rng.shuffle(order)
        model.train()
        for start in range(0, len(order), BATCH_SIZE):
            idx = order[start : start + BATCH_SIZE]
            pred = model(x_t[idx], c_t[idx], torch.zeros((len(idx), 2), device=device))
            loss = loss_fn(pred, y_t[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    model.eval()
    with torch.no_grad():
        xx = torch.as_tensor(x_valid, dtype=torch.float32, device=device)
        cc = torch.as_tensor(cam_valid, dtype=torch.long, device=device)
        tt = torch.as_tensor(time_valid, dtype=torch.float32, device=device)
        return model(xx, cc, tt).cpu().numpy()


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    random.seed(SEED)
    np.random.seed(SEED)
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise SystemExit("PyTorch is required; use the project's .venv-v31 interpreter") from exc

    train, test = _load()
    train = train.dropna(subset=["recording", "unit", "minute_of_day", "people"]).copy()
    train["minute_of_day"] = train["minute_of_day"].astype(int)
    panels = _panel_map(train)
    hidden = test.loc[test["kind"].eq("hidden")].copy()
    hidden["validation_recording"] = hidden["recording"].map(TEST_TO_VALIDATION)
    hidden["minute_of_day"] = pd.to_numeric(hidden["minute_of_day"], errors="coerce")
    hidden = hidden.dropna(subset=["validation_recording", "minute_of_day"])
    hidden["minute_of_day"] = hidden["minute_of_day"].astype(int)

    label_map = train.set_index(["recording", "unit", "minute_of_day"])["people"]
    keep_by_recording: dict[str, set[tuple[str, str, int]]] = {}
    for row in hidden.itertuples(index=False):
        rec = str(row.validation_recording)
        unit = str(row.unit)
        minute = int(row.minute_of_day)
        if (rec, unit, minute) in label_map.index:
            keep_by_recording.setdefault(rec, set()).add((rec, unit, minute))
    if sum(map(len, keep_by_recording.values())) != 843:
        raise SystemExit(
            "Expected exactly 843 mapped hidden validation targets; got "
            f"{sum(map(len, keep_by_recording.values()))}"
        )

    oof: list[pd.DataFrame] = []
    for heldout in sorted(keep_by_recording):
        fit_recs = [rec for rec in panels if rec != heldout]
        medians, scales = _camera_scaling(train, fit_recs)
        adj = _adjacency(train, panels, fit_recs)
        xtr, ytr, ctr, _ = _make_examples(
            train, panels, fit_recs, medians, scales
        )
        xva, yva, cva, mva = _make_examples(
            train, panels, [heldout], medians, scales, keep_by_recording[heldout]
        )
        if len(yva) == 0:
            raise SystemExit(f"No mapped hidden validation rows for {heldout}")
        minutes = np.asarray([m[2] for m in mva], dtype=float)
        clock = np.column_stack(
            [np.sin(2 * np.pi * minutes / 1440), np.cos(2 * np.pi * minutes / 1440)]
        ).astype(np.float32)
        prediction_scaled = _fit_and_predict(xtr, ytr, ctr, xva, cva, clock, adj)
        truth = np.asarray([m[3] for m in mva], dtype=float)
        camera_idx = cva
        pred_log = prediction_scaled * scales[camera_idx] + medians[camera_idx]
        pred = np.maximum(0.0, np.expm1(pred_log))
        result = pd.DataFrame(
            {
                "recording": heldout,
                "unit": [m[1] for m in mva],
                "minute_of_day": minutes.astype(int),
                "truth": truth,
                "stgcn_prediction": pred,
                "reference_mae": REFERENCE_MAE[heldout],
            }
        )
        oof.append(result)
        print(
            f"Fold {heldout}: train_rows={len(ytr)}; validation_rows={len(result)}; "
            f"ST-GCN MAE={np.mean(np.abs(truth - pred)):.4f}; "
            f"current reference MAE={REFERENCE_MAE[heldout]:.4f}"
        )

    out = pd.concat(oof, ignore_index=True)
    out.to_csv(OUT, index=False)
    print("\n=== SUMMARY ===")
    for rec, group in out.groupby("recording"):
        mae = float(np.mean(np.abs(group["truth"] - group["stgcn_prediction"])))
        print(f"{rec}: rows={len(group)} MAE={mae:.4f} reference={REFERENCE_MAE[rec]:.4f}")
    print("Overall MAE:", round(float(np.mean(np.abs(out["truth"] - out["stgcn_prediction"]))), 4))
    print("Saved diagnostic:", OUT)
    print("OOF VALIDATION ONLY — NO TEST CANDIDATE OR SUBMISSION CREATED")


if __name__ == "__main__":
    main()
