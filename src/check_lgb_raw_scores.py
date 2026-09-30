from importlib import import_module
from sklearn.model_selection import train_test_split
m = import_module("14_synthetic_drift_lightgbm")
X, y = m.load_and_clean("UDP", m.SAMPLE_SIZE)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=m.SEED, stratify=y)
model = m.train_lgb(Xtr, ytr)
raw = model.predict(Xtr.values, raw_score=True)
print("raw score min/max:", raw.min(), raw.max())