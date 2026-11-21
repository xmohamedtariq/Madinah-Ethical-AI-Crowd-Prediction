"""Conservative Missing reconstruction using released within-camera endpoints."""
import numpy as np
import pandas as pd


def predict_missing(camera_rows: pd.DataFrame, target_minutes):
    """Interpolate missing positions on the camera's observed minute index."""
    series = camera_rows.drop_duplicates("minute_of_day").set_index("minute_of_day")["people"].sort_index()
    series = series.reindex(sorted(set(series.index).union(target_minutes)))
    return series.interpolate(method="linear", limit_area="inside").reindex(target_minutes).clip(lower=0)
