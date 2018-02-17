import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.feature_selection import SelectFromModel, SelectFwe, f_regression
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import Normalizer
from tpot.builtins import StackingEstimator

# from tpot import TPOTRegressor

# tpot = TPOTRegressor(
#     generations=100,
#     population_size=100,
#     scoring=mape_scorer,
#     cv=5,
#     n_jobs=-1,
#     random_state=RANDOM_SEED,
#     verbosity=2)
# tpot.fit(X_train, y_train)
# print(tpot.score(X_test, y_test))
# tpot.export('tpot_pipeline.py')

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=42)

# Score on the training set was:-0.2606822011378389
exported_pipeline = make_pipeline(
    Normalizer(norm="max"),
    SelectFwe(score_func=f_regression, alpha=0.015),
    SelectFromModel(estimator=ExtraTreesRegressor(max_features=0.8, n_estimators=100), threshold=0.25),
    StackingEstimator(estimator=RidgeCV()),
    GradientBoostingRegressor(alpha=0.85, learning_rate=0.1, loss="ls", max_depth=10, max_features=1.0, min_samples_leaf=1, min_samples_split=2, n_estimators=100, subsample=0.55)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
