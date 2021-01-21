"""Download the datasets needed for this work."""

import pandas as pd
import os

# QM9: For this, we use the Pandas-friendly version
# from: https://github.com/globus-labs/g4mp2-atomization-energy
data = pd.read_json('https://github.com/globus-labs/g4mp2-atomization-energy/raw/master/data/output/g4mp2_data.json.gz', lines=True)
data = data.sample(25000, random_state=1)
data.to_json(os.path.join('datasets', 'qm9.json.gz'), orient='records', lines=True)
