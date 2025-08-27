import pandas as pd
from new_sdv import GaussianCopulaSynthesizer, RangeConstraint, UniqueConstraint

dati_reali = pd.DataFrame({
    'prezzo': [10.5, 12.0, 9.8, 11.3, 10.0],
    'quantita': [100, 150, 80, 120, 90],
    'categoria': ['TipoA', 'TipoB', 'TipoA', 'TipoC', 'TipoB']
})

vincoli = [
    RangeConstraint('prezzo', min_val=0, max_val=100),
    UniqueConstraint('categoria')
]
synth = GaussianCopulaSynthesizer(constraints=vincoli, random_state=0)

synth.fit(dati_reali)
dati_sintetici = synth.sample(10)

print("Dati sintetici generati:")
print(dati_sintetici)
