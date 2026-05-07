import pandas as pd

df = pd.read_csv("datasets/analcatdata_authorship.csv")

df['target'] = df['target'].map({'N': 0, 'P': 1})

df.to_csv("datasets/analcatdata_authorship.csv", index=False)

print("Fixed target column ✅")