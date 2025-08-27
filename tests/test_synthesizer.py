import pandas as pd
import numpy as np
from new_sdv import GaussianCopulaSynthesizer, RangeConstraint

def test_synthesizer_basic():
    np.random.seed(42)
    df = pd.DataFrame({
        'valore': np.random.rand(100) * 10,
        'categoria': np.random.choice(['A', 'B', 'C'], size=100)
    })
    synth = GaussianCopulaSynthesizer(
        constraints=[RangeConstraint('valore', min_val=0, max_val=10)],
        random_state=42
    )
    synth.fit(df)
    synth_data = synth.sample(50)
    assert synth_data.shape == (50, 2)
    assert set(synth_data.columns) == {'valore', 'categoria'}
    assert (synth_data['valore'] >= 0).all() and (synth_data['valore'] <= 10).all()
    assert pd.api.types.is_float_dtype(synth_data['valore'])
    assert set(synth_data['categoria']).issubset({'A', 'B', 'C'})
