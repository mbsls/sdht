"""Stock--Watson style static principal-component factor for the soft block.

`SoftFactor` standardizes the input variables (z-score per column) and
extracts the leading principal component. The factor's sign is fixed by
correlation with a reference variable (default Michigan Consumer Sentiment)
so that, across recursive refits, increases in the factor consistently mean
the same thing economically.

Why recursive refit
-------------------
In a pseudo-real-time exercise, each forecast origin gets a freshly fit
factor using only data the forecaster could have seen by that origin.
Otherwise the factor encodes future information through the PCA
loadings, defeating the point of the harness.

Missing data
------------
PCA needs complete cases. Variables with later sample starts (e.g.
Dallas Fed manufacturing from 2004; in-vintage from 2016) restrict the
effective sample to their intersection. The caller is expected to pass
a variable list that gives a non-trivial intersection for the
origins of interest.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class SoftFactor:
    def __init__(
        self,
        variables: list[str],
        n_components: int = 1,
        sign_reference: Optional[str] = "consumer_sentiment",
    ):
        self.variables = list(variables)
        self.n_components = n_components
        self.sign_reference = sign_reference
        self.scaler_: Optional[StandardScaler] = None
        self.pca_: Optional[PCA] = None
        self.sign_: Optional[np.ndarray] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None
        self.loadings_: Optional[pd.DataFrame] = None
        self.fit_index_: Optional[pd.DatetimeIndex] = None

    def fit(self, df: pd.DataFrame) -> "SoftFactor":
        clean = df[self.variables].dropna(how="any")
        if len(clean) < self.n_components + 5:
            raise ValueError(
                f"Too few complete-case rows ({len(clean)}) for {self.n_components} "
                f"components on variables {self.variables}."
            )
        self.fit_index_ = clean.index
        self.scaler_ = StandardScaler().fit(clean.values)
        z = self.scaler_.transform(clean.values)
        self.pca_ = PCA(n_components=self.n_components).fit(z)
        self.explained_variance_ratio_ = self.pca_.explained_variance_ratio_
        self.loadings_ = pd.DataFrame(
            self.pca_.components_.T,
            index=self.variables,
            columns=[f"f{i+1}" for i in range(self.n_components)],
        )

        # Sign convention: factor i is sign-flipped so it correlates
        # positively with the reference variable on the fitted sample.
        scores = self.pca_.transform(z)
        signs = np.ones(self.n_components)
        if self.sign_reference and self.sign_reference in clean.columns:
            ref = clean[self.sign_reference].values
            for i in range(self.n_components):
                corr = np.corrcoef(scores[:, i], ref)[0, 1]
                if corr < 0:
                    signs[i] = -1.0
        self.sign_ = signs
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.pca_ is None:
            raise RuntimeError("Call .fit() before .transform().")
        clean = df[self.variables].dropna(how="any")
        if clean.empty:
            return pd.DataFrame(
                index=df.index,
                columns=[f"soft_f{i+1}" for i in range(self.n_components)],
                dtype=float,
            )
        z = self.scaler_.transform(clean.values)
        scores = self.pca_.transform(z) * self.sign_
        out = pd.DataFrame(
            scores,
            index=clean.index,
            columns=[f"soft_f{i+1}" for i in range(self.n_components)],
        )
        # Re-index to the input frame's index, leaving NaN where soft vars
        # weren't complete; the harness's dropna will drop those rows.
        return out.reindex(df.index)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def __repr__(self) -> str:
        if self.pca_ is None:
            return f"SoftFactor(unfit, vars={self.variables})"
        evr = self.explained_variance_ratio_
        return (
            f"SoftFactor(n={self.n_components}, vars={self.variables}, "
            f"explained_var_ratio={evr.round(3).tolist()})"
        )
