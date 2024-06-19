import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from catboost import CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFECV
from sklearn.svm import SVR
import warnings
warnings.filterwarnings('ignore')

# Load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Combine train and test for uniform preprocessing
train_v1 = pd.concat([train, test], sort=False)

# Fill missing values
train_v1['LotFrontage'] = train_v1['LotFrontage'].fillna(train_v1['LotFrontage'].median())
train_v1.drop(['Alley', 'MasVnrType', 'MasVnrArea'], axis=1, inplace=True)
for col in ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']:
    train_v1[col] = train_v1[col].fillna('Not applicable')
for col in ['GarageYrBlt', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'GarageArea', 'GarageCars']:
    train_v1[col] = train_v1[col].fillna(0)
for col in ['Electrical', 'MSZoning', 'Utilities', 'Exterior1st', 'Exterior2nd', 'KitchenQual', 'Functional', 'SaleType']:
    train_v1[col] = train_v1[col].fillna(train_v1[col].mode()[0])

# Feature Engineering
train_v1['TotalSF'] = train_v1['TotalBsmtSF'] + train_v1['1stFlrSF'] + train_v1['2ndFlrSF']
train_v1['TotalBathrooms'] = train_v1['FullBath'] + train_v1['BsmtFullBath'] + 0.5 * (train_v1['HalfBath'] + train_v1['BsmtHalfBath'])
train_v1['TotalPorchSF'] = train_v1['OpenPorchSF'] + train_v1['3SsnPorch'] + train_v1['EnclosedPorch'] + train_v1['ScreenPorch'] + train_v1['WoodDeckSF']
train_v1['HasPool'] = np.where(train_v1['PoolArea'] > 0, 1, 0)
train_v1['Has2ndFloor'] = np.where(train_v1['2ndFlrSF'] > 0, 1, 0)
train_v1['HasGarage'] = np.where(train_v1['GarageArea'] > 0, 1, 0)
train_v1['HasBsmt'] = np.where(train_v1['TotalBsmtSF'] > 0, 1, 0)
train_v1['HasFireplace'] = np.where(train_v1['Fireplaces'] > 0, 1, 0)

# One-hot encoding
train_encoded = pd.get_dummies(train_v1, columns=['MSSubClass', 'MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',
                                                  'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 
                                                  'Exterior1st', 'Exterior2nd', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 
                                                  'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 
                                                  'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 
                                                  'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition'], dtype=int)

# Split train and test
train_processed = train_encoded[train_encoded['SalePrice'].notna()].copy()
test_processed = train_encoded[train_encoded['SalePrice'].isna()].copy()

Y_train = train_processed['SalePrice']
train_processed.drop(['SalePrice', 'Id'], axis=1, inplace=True)
X_train = train_processed

X_test = test_processed.drop(['SalePrice', 'Id'], axis=1)

# Train-test split for validation
X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

# Feature Selection with RFECV
rfecv = RFECV(estimator=CatBoostRegressor(verbose=0), step=1, cv=5, scoring='neg_mean_squared_error')
rfecv.fit(X_train, Y_train)

X_train_selected = rfecv.transform(X_train)
X_valid_selected = rfecv.transform(X_valid)
X_test_selected = rfecv.transform(X_test)

# Model Building with CatBoost, Gradient Boosting, Ridge, Lasso, ElasticNet, and Stacking
catboost_model = CatBoostRegressor(verbose=0, iterations=1000, depth=8, learning_rate=0.05, l2_leaf_reg=3)
gbr_model = GradientBoostingRegressor(n_estimators=1000, max_depth=8, learning_rate=0.05)
ridge_model = Ridge(alpha=1.0)
lasso_model = Lasso(alpha=0.01)
elastic_model = ElasticNet(alpha=0.01, l1_ratio=0.5)

# Train individual models
catboost_model.fit(X_train_selected, Y_train)
gbr_model.fit(X_train_selected, Y_train)
ridge_model.fit(X_train_selected, Y_train)
lasso_model.fit(X_train_selected, Y_train)
elastic_model.fit(X_train_selected, Y_train)

# Stacking Regressor
estimators = [
    ('catboost', catboost_model),
    ('gbr', gbr_model),
    ('ridge', ridge_model),
    ('lasso', lasso_model),
    ('elastic', elastic_model)
]
stacking_model = StackingRegressor(estimators=estimators, final_estimator=CatBoostRegressor(verbose=0, iterations=1000, depth=8, learning_rate=0.05, l2_leaf_reg=3))

stacking_model.fit(X_train_selected, Y_train)

# Evaluate
valid_preds_stacking = stacking_model.predict(X_valid_selected)
rmse_stacking = mean_squared_error(Y_valid, valid_preds_stacking, squared=False)
mae_stacking = mean_absolute_error(Y_valid, valid_preds_stacking)
print(f'Stacking Model RMSE: {rmse_stacking}')
print(f'Stacking Model MAE: {mae_stacking}')

# Predict on test set
final_preds_stacking = stacking_model.predict(X_test_selected)

# Submission
submission = pd.DataFrame({'Id': test['Id'], 'SalePrice': final_preds_stacking})
submission.to_csv('submission.csv', index=False)
