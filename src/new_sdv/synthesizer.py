import numpy as np
import pandas as pd
import torch

from .distributions import DistributionFitter
from .constraints import Constraint


class GaussianCopulaSynthesizer:
    """
    Synthesizer a copula gaussiana:
      - Stima una distribuzione marginale per ogni colonna (beta/gamma/norm/categorical)
      - Trasforma in spazio latente Z ~ N(0,1) via PIT
      - Stima covarianza/correlazione e campiona da MVN
      - Inverte le trasformazioni per ottenere dati sintetici nello spazio originale
    """

    def __init__(self, constraints=None, random_state=None):
        self.constraints = constraints or []
        # Gestione seed per riproducibilità
        if isinstance(random_state, int):
            np.random.seed(random_state)
            torch.manual_seed(random_state)
            self._np_random = np.random.RandomState(random_state)
        else:
            self._np_random = np.random
        self.random_state = random_state
        self.fitted = False

    # ------------------------
    #  FIT
    # ------------------------
    def fit(self, data: pd.DataFrame):
        self.columns = list(data.columns)
        self.n_columns = len(self.columns)
        self.distributions = {}

        # Scelta distribuzione per colonna
        for col in self.columns:
            series = data[col].dropna()
            # Se la colonna è tutta NaN, non possiamo stimare: creiamo distribuzione categorica "NaN"
            if series.empty:
                self.distributions[col] = ("categorical", {"probs": {"<NA>": 1.0}})
                continue

            if pd.api.types.is_numeric_dtype(series):
                # Integers -> float per la stima
                if pd.api.types.is_integer_dtype(series):
                    series = series.astype(float)
                min_val, max_val = float(series.min()), float(series.max())

                # Heuristica per la scelta della marginale
                if min_val >= 0.0 and max_val <= 1.0:
                    dist_name = "beta"
                    params = DistributionFitter.fit_beta(series.values)
                elif min_val >= 0.0:
                    # Dati positivi: se molto skewed, gamma; altrimenti normale
                    skewness = float(pd.Series(series).skew())
                    if skewness > 1.0:
                        dist_name = "gamma"
                        params = DistributionFitter.fit_gamma(series.values)
                    else:
                        dist_name = "norm"
                        params = DistributionFitter.fit_normal(series.values)
                else:
                    dist_name = "norm"
                    params = DistributionFitter.fit_normal(series.values)
            else:
                dist_name = "categorical"
                params = DistributionFitter.fit_categorical(series)

            self.distributions[col] = (dist_name, params)

        # Trasformazione nel latente Z (cdf -> icdf normale)
        latent_cols = []
        for col in self.columns:
            dist_name, params = self.distributions[col]
            # Fill NaN robusto: ffill+bfill per evitare NaN in testa/coda
            series_clean = data[col].fillna(method="ffill").fillna(method="bfill")
            if dist_name == "categorical":
                u = DistributionFitter.cdf_categorical(series_clean, params, rng=self._np_random)
            else:
                u = DistributionFitter.cdf(dist_name, series_clean, params)
            # Clip per evitare 0/1 esatti
            u = np.clip(u, 1e-6, 1 - 1e-6)
            z = torch.distributions.Normal(0, 1).icdf(torch.tensor(u, dtype=torch.float32))
            latent_cols.append(z)

        latent = torch.vstack(latent_cols).T  # shape: (n_rows, n_cols)

        # Covarianza robusta + jitter per PD
        try:
            cov = torch.cov(latent.T)
        except AttributeError:
            cov = torch.from_numpy(np.cov(latent.T.numpy()))
        # Jitter diagonale
        cov = cov + 1e-6 * torch.eye(self.n_columns)
        # Fattorizzazione di Cholesky (se fallisce, aumenta jitter)
        try:
            L = torch.linalg.cholesky(cov)
        except RuntimeError:
            cov = cov + 1e-3 * torch.eye(self.n_columns)
            L = torch.linalg.cholesky(cov)

        self.covariance = cov
        self.cholesky_L = L
        self.fitted = True

    # ------------------------
    #  SAMPLE
    # ------------------------
    def sample(self, num_rows: int) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("Eseguire fit() prima di sample().")

        if isinstance(self.random_state, int):
            torch.manual_seed(self.random_state)

        # Campiona da MVN(0, cov) via Cholesky
        z = torch.randn(num_rows, self.n_columns)
        z_corr = z @ self.cholesky_L.T  # (num_rows, n_cols)
        z_corr_np = z_corr.numpy()

        out = {}
        std_normal = torch.distributions.Normal(0, 1)
        for j, col in enumerate(self.columns):
            dist_name, params = self.distributions[col]
            u = std_normal.cdf(torch.tensor(z_corr_np[:, j])).numpy()
            # Inversione marginale
            if dist_name == "categorical":
                col_vals = DistributionFitter.inv_cdf_categorical(u, params)
            else:
                col_vals = DistributionFitter.inv_cdf(dist_name, u, params)
            out[col] = col_vals

        df = pd.DataFrame(out)

        # Applica vincoli (post-processing)
        for c in self.constraints:
            if isinstance(c, Constraint):
                df = c.ensure(df)

        return df
