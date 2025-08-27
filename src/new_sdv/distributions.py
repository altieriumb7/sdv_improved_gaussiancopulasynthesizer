import numpy as np
import pandas as pd
import scipy.stats as st


class DistributionFitter:
    """
    Utility per stimare distribuzioni marginali e calcolare CDF/PPF
    (inversa della CDF) con accorgimenti robusti per casi ai bordi.
    """

    # ------------------------
    #  FIT DELLE DISTRIBUZIONI
    # ------------------------
    @staticmethod
    def fit_beta(values):
        """
        Stima i parametri (alpha, beta) di una Beta su dati in [0,1].
        Consente 0 e 1 spostandoli leggermente dentro (0,1) per evitare FitDataError.
        """
        vals = np.asarray(values, dtype=float)
        # Riporta eventuali piccoli errori numerici dentro [0,1]
        vals = np.clip(vals, 0.0, 1.0)
        # Sposta 0 e 1 dentro l'intervallo aperto per la MLE della Beta
        eps = 1e-6
        vals = np.clip(vals, eps, 1.0 - eps)

        # Forza loc=0 e scale=1 (supporto in [0,1])
        a, b, loc, scale = st.beta.fit(vals, floc=0, fscale=1)
        return {"a": a, "b": b}

    @staticmethod
    def fit_gamma(values):
        """
        Stima i parametri (shape=a, loc, scale) della Gamma su dati >= 0.
        Sposta eventuali zeri molto piccoli a un epsilon per robustezza.
        """
        vals = np.asarray(values, dtype=float)
        # Forza non negatività e sostituisce zeri con epsilon per evitare degenerazioni
        eps = 1e-8
        vals = np.clip(vals, 0.0, None)
        vals[vals < eps] = eps

        # Vincola loc=0 per dati non negativi
        a, loc, scale = st.gamma.fit(vals, floc=0)
        return {"a": a, "loc": loc, "scale": scale}

    @staticmethod
    def fit_normal(values):
        """
        Stima mu e sigma per la Normale. Impone sigma minimo per colonne quasi costanti.
        """
        vals = np.asarray(values, dtype=float)
        mu = float(np.mean(vals))
        sigma = float(np.std(vals, ddof=0))
        if sigma < 1e-8:
            sigma = 1e-8
        return {"mu": mu, "sigma": sigma}

    @staticmethod
    def fit_categorical(values):
        """
        Distribuzione empirica per variabile categorica.
        Ritorna dict { 'probs': {categoria: p, ...} }.
        """
        counts = pd.value_counts(values, normalize=True)
        # Se una categoria ha prob 0 (non dovrebbe), la rimuoviamo
        probs = {k: float(v) for k, v in counts.to_dict().items() if v > 0}
        return {"probs": probs}

    # ------------------------
    #  CDF / PPF CONTINUI
    # ------------------------
    @staticmethod
    def cdf(dist_name, values, params):
        """Calcola la CDF per una distribuzione continua supportata."""
        vals = np.asarray(values, dtype=float)
        if dist_name == "beta":
            # Dati fuori [0,1] vengono tagliati per robustezza
            vals = np.clip(vals, 0.0, 1.0)
            return st.beta.cdf(vals, params["a"], params["b"], loc=0, scale=1)
        elif dist_name == "gamma":
            vals = np.clip(vals, 0.0, None)
            return st.gamma.cdf(vals, params["a"], loc=params["loc"], scale=params["scale"])
        elif dist_name == "norm":
            return st.norm.cdf(vals, loc=params["mu"], scale=params["sigma"])
        else:
            raise ValueError(f"Distribuzione {dist_name} non supportata in cdf().")

    @staticmethod
    def inv_cdf(dist_name, u_values, params):
        """Calcola l'inversa della CDF (PPF). Si attende u in (0,1)."""
        u = np.asarray(u_values, dtype=float)
        # Clip per sicurezza numerica (evita -inf/+inf in ppf)
        u = np.clip(u, 1e-12, 1 - 1e-12)
        if dist_name == "beta":
            return st.beta.ppf(u, params["a"], params["b"], loc=0, scale=1)
        elif dist_name == "gamma":
            return st.gamma.ppf(u, params["a"], loc=params["loc"], scale=params["scale"])
        elif dist_name == "norm":
            return st.norm.ppf(u, loc=params["mu"], scale=params["sigma"])
        else:
            raise ValueError(f"Distribuzione {dist_name} non supportata in inv_cdf().")

    # ------------------------
    #  CATEGORICAL <-> UNIFORM
    # ------------------------
    @staticmethod
    def cdf_categorical(values, params, rng=None):
        """
        Mappa categorie -> U(0,1) assegnando a ciascuna categoria un intervallo proporzionale alla sua probabilità.
        Ritorna per ogni valore una posizione random uniforme nel suo intervallo.
        """
        if rng is None:
            rng = np.random
        probs = params["probs"]
        # Ordine deterministico per stabilità
        categories = sorted(probs.keys())
        # Costruisce intervalli cumulativi
        cum_intervals = {}
        cum = 0.0
        for cat in categories:
            p = probs[cat]
            cum_intervals[cat] = (cum, cum + p)
            cum += p
        # Normalizza eventuali piccoli errori di somma
        if abs(cum - 1.0) > 1e-9 and cum > 0:
            # Riscalare intervalli per farli finire a 1.0
            scale = 1.0 / cum
            for cat in categories:
                low, high = cum_intervals[cat]
                cum_intervals[cat] = (low * scale, high * scale)

        # Assegna U nel rispettivo intervallo
        u_values = []
        for val in values:
            if val in cum_intervals:
                low, high = cum_intervals[val]
                u = rng.uniform(low, high) if high > low else low
            else:
                # Categoria unseen: assegna un valore molto piccolo (quasi 0)
                u = 1e-12
            u_values.append(u)
        return np.asarray(u_values, dtype=float)

    @staticmethod
    def inv_cdf_categorical(u_values, params):
        """
        Mappa U(0,1) -> categorie in base alle soglie cumulative.
        """
        probs = params["probs"]
        categories = sorted(probs.keys())
        thresholds = []
        cum = 0.0
        for cat in categories:
            cum += probs[cat]
            thresholds.append(cum)
        # Normalizza ultima soglia a 1
        if thresholds:
            thresholds[-1] = 1.0

        inv_values = []
        for u in np.asarray(u_values, dtype=float):
            u = float(np.clip(u, 0.0, 1.0))
            # Trova la prima soglia >= u
            chosen = categories[-1]  # fallback
            for cat, thr in zip(categories, thresholds):
                if u <= thr:
                    chosen = cat
                    break
            inv_values.append(chosen)
        return np.asarray(inv_values, dtype=object)
