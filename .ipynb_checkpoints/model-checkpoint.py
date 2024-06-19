import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import StackingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV

# Load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
submission = pd.read_csv('sample_submission.csv')

# Separate target variable
y = train['SalePrice']
train.drop(['SalePrice'], axis=1, inplace=True)

# Combine train and test data for preprocessing
all_data = pd.concat([train, test], sort=False)

# Identify numerical and categorical columns
numeric_features = all_data.select_dtypes(include=['int64', 'float64']).columns
categorical_features = all_data.select_dtypes(include=['object']).columns

# Handle missing values for numerical features
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Handle missing values and encode categorical features
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combine transformations
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Preprocess the data
train_preprocessed = preprocessor.fit_transform(train)
test_preprocessed = preprocessor.transform(test)

# Feature Engineering
additional_train_feature = (train['TotalBsmtSF'] + train['1stFlrSF'] + train['2ndFlrSF']).values.reshape(-1, 1)
additional_test_feature = (test['TotalBsmtSF'] + test['1stFlrSF'] + test['2ndFlrSF']).values.reshape(-1, 1)

train_preprocessed = np.hstack((train_preprocessed, additional_train_feature))
test_preprocessed = np.hstack((test_preprocessed, additional_test_feature))


# Train/Test split for validation
X_train, X_valid, y_train, y_valid = train_test_split(train_preprocessed, y, test_size=0.2, random_state=42)

# Basic model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_valid)
rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
print(f'Validation RMSE: {rmse}')

# Hyperparameter tuning with GridSearchCV or RandomizedSearchCV
param_dist = {
    'n_estimators': [100, 200, 500],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

rf_random = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)
rf_random.fit(X_train, y_train)
best_model = rf_random.best_estimator_


# Define base models
base_models = [
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
    ('xgb', XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42))
]

# Define stacking model
stacking_model = StackingRegressor(
    estimators=base_models,
    final_estimator=RidgeCV()
)

stacking_model.fit(X_train, y_train)

# Predictions
stacking_preds = stacking_model.predict(X_valid)
stacking_rmse = np.sqrt(mean_squared_error(y_valid, stacking_preds))
print(f'Stacking Model Validation RMSE: {stacking_rmse}')


# Final predictions on test set
final_predictions = stacking_model.predict(test_preprocessed)

# Create submission file
submission['SalePrice'] = final_predictions
submission.to_csv('submission.csv', index=False)


