

import statsmodels.api as sm
import numpy as np
import pandas as pd

from sklearn import datasets ## imports datasets from scikit-learn
from sklearn import linear_model



data = datasets.load_boston() ## loads Boston dataset from datasets library



# define the data/predictors as the pre-set feature names
df = pd.DataFrame(data.data, columns=data.feature_names)

# Put the target (housing value -- MEDV) in another DataFrame
target = pd.DataFrame(data.target, columns=["MEDV"])


X = df["RM"]
y = target["MEDV"]

# Note the difference in argument order
model = sm.OLS(y, X).fit()
predictions = model.predict(X) # make the predictions by the model

# Print out the statistics
model.summary()



X = df[["RM", "LSTAT"]]
y = target["MEDV"]
model = sm.OLS(y, X).fit()
predictions = model.predict(X)
model.summary()



# define the data/predictors as the pre-set feature names
df = pd.DataFrame(data.data, columns=data.feature_names)

# Put the target (housing value -- MEDV) in another DataFrame
target = pd.DataFrame(data.target, columns=["MEDV"])


X = df
y = target["MEDV"]


lm = linear_model.LinearRegression()
model = lm.fit(X,y)


predictions = lm.predict(X)
print(predictions)[0:5]


lm.score(X,y)
lm.coef_
lm.intercept_