# new_sdv: Generazione di dati sintetici con Gaussian Copula

**new_sdv** è una libreria Python per la generazione di dati sintetici tabellari utilizzando un modello a *copula gaussiana*. Questo approccio permette di preservare le distribuzioni marginali di ogni colonna e le correlazioni tra colonne.

## Installazione
```bash
git clone <URL_del_repository>
cd new_sdv
pip install -e .
```

## Esempio di utilizzo
```python
import pandas as pd
from new_sdv import GaussianCopulaSynthesizer, RangeConstraint

dati_reali = pd.DataFrame({
    'reddito': [50000, 60000, 55000, 52000, 58000],
    'eta': [25, 45, 32, 40, 52],
    'città': ['Roma', 'Milano', 'Roma', 'Napoli', 'Milano']
})

synth = GaussianCopulaSynthesizer(
    constraints=[RangeConstraint('eta', min_val=18, max_val=90)],
    random_state=0
)

synth.fit(dati_reali)
dati_sintetici = synth.sample(10)
print(dati_sintetici)
```
