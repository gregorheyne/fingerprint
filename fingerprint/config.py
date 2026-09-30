from __future__ import annotations
import itertools
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterator

import yaml


# ---------------------------------------------------------------------------
# Algorithm hyperparameters, centralized.
#
# Frozen so a base AnalysisConfig can be reused as the starting point for many
# sweep variants (see `sweep_configs`) without one variant's `replace` call
# mutating another. Defaults here match what used to be hardcoded across
# embedding.py/clustering.py/fingerprint.py, so switching to this config has
# no effect on behavior until a value is actually overridden.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PCAConfig:
    n_components: int = 10
    standardize: bool = True
    random_state: int = 0


@dataclass(frozen=True)
class UMAPConfig:
    n_components: int = 10
    n_neighbors: int = 15
    min_dist: float = 0.1
    random_state: int = 0


@dataclass(frozen=True)
class HDBSCANConfig:
    min_cluster_size: int = 5
    min_samples: int | None = 15


@dataclass(frozen=True)
class AnalysisConfig:
    pca: PCAConfig = field(default_factory=PCAConfig)
    umap: UMAPConfig = field(default_factory=UMAPConfig)
    hdbscan: HDBSCANConfig = field(default_factory=HDBSCANConfig)


def load_config(path: str | Path | None = None) -> AnalysisConfig:
    """Load an `AnalysisConfig`, optionally overridden from a YAML file.

    Falls back to defaults if `path` is None or doesn't exist, so this is
    always safe to call. A YAML file only needs to set the fields it wants to
    change -- e.g. just `hdbscan: {min_cluster_size: 10}` -- everything else
    keeps its dataclass default.
    """
    config = AnalysisConfig()
    if path is None:
        return config
    path = Path(path)
    if not path.exists():
        return config
    raw = yaml.safe_load(path.read_text()) or {}
    return AnalysisConfig(
        pca=replace(config.pca, **raw.get("pca", {})),
        umap=replace(config.umap, **raw.get("umap", {})),
        hdbscan=replace(config.hdbscan, **raw.get("hdbscan", {})),
    )


def sweep_configs(base: AnalysisConfig, **grid: dict[str, list]) -> Iterator[AnalysisConfig]:
    """Yield one `AnalysisConfig` per combination in the cartesian product of
    the given per-section value grids, e.g.

        sweep_configs(base, pca={"n_components": [5, 10, 15]},
                            hdbscan={"min_cluster_size": [5, 10]})

    yields 6 configs (3 x 2), each `base` with one (section, field) combo
    overridden via `dataclasses.replace`. Sections not named in `grid` are
    left untouched.
    """
    keys = [(section, name) for section, fields in grid.items() for name in fields]
    value_lists = [grid[section][name] for section, name in keys]
    for combo in itertools.product(*value_lists):
        overrides: dict[str, dict] = {}
        for (section, name), value in zip(keys, combo):
            overrides.setdefault(section, {})[name] = value
        yield AnalysisConfig(
            pca=replace(base.pca, **overrides.get("pca", {})),
            umap=replace(base.umap, **overrides.get("umap", {})),
            hdbscan=replace(base.hdbscan, **overrides.get("hdbscan", {})),
        )
