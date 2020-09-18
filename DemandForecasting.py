import numpy as np
import pandas as pd
from tpot import TPOTRegressor
from sklearn.model_selection import train_test_split

train = pd.read_csv('/home/noren/Desktop/AV_Jantahack_DemandForcasting/dataset/train_0irEZ2H.csv', parse_dates=['week'])
#test = pd.read_csv('/home/noren/Desktop/AV_Jantahack_DemandForcasting/dataset/test_nfaJ3J5.csv', parse_dates=['week']) 
#sample = pd.read_csv('/home/noren/Desktop/AV_Jantahack_DemandForcasting/dataset/sample_submission_pzljTaX.csv')

target = train['units_sold']
train = train.drop(columns=['week', 'units_sold'])
#test = test.drop(columns=['week'])
X_train, X_test, y_train, y_test = train_test_split(train, target,
                                                    train_size=0.75, test_size=0.25, random_state=42)

tpot = TPOTRegressor(generations=5, population_size=50, verbosity=2, random_state=42)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_Demand_pipeline.py')


