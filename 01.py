

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset

df = pd.read_csv('D:\insurance.csv')
print(df.head())

print(df.info())

print(df.describe())

print(df.isnull().sum())

print(df.duplicated().sum())

print(df.isna().sum())

print(df.shape)

# Exploratory Data Analysis (EDA)

categorial_cols = df.select_dtypes(include=['object']).columns

for col in categorial_cols:
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, x=col)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    sns.countplot(data=df, x=col, palette=["#FF9999","#66B2FF","#99FF99","#FFD966"])
    plt.show()


numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

for col in numerical_cols:

    plt.figure(figsize=(8, 4))
    sns.histplot(data=df, x=col, kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()

for col in numerical_cols:
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=df, x=col)
    plt.title(f'Boxplot of {col}')
    plt.xlabel(col)
    plt.ylabel('Value')
    plt.show()

for col in numerical_cols:
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=df, x='smoker', y='charges',palette=["#E5707A","#0A4D8F"])
    plt.title(f'Boxplot of {col} by Smoker Status')
    plt.xlabel('Smoker status')    
    plt.ylabel('charges')
    plt.show()

for col in numerical_cols:
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=df, x='region', y='charges')
    # sns.boxplot(data=df, x=col, palette=["#FF9999","#66B2FF","#99FF99","#FFD966"])
    plt.title('Boxplot of charges by Region')
    plt.xlabel('Region')    
    plt.ylabel('charges')       
    plt.show()

    plt.figure(figsize=(8, 5))
    sns.pairplot(df)
    plt.suptitle('Pairplot of Numerical Features',y=1.02)
    plt.show()

# Correlation analysis

df_encoded = pd.get_dummies(df,columns=['sex','smoker','region'],drop_first=True)
print(df_encoded.head())

plt.figure(figsize=(8,5))
sns.heatmap(df_encoded.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')    
plt.show()

# Prepare the data  for train test split

X=df_encoded.drop('charges',axis=1)
y=df_encoded['charges']     

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

# prepare the model

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train) 

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Evaluate the model on training set

mae = mean_absolute_error(y_train,y_train_pred)
mse = mean_squared_error(y_train, y_train_pred)
rmse = mean_squared_error(y_train, y_train_pred)
r2 = r2_score(y_train, y_train_pred)

print('mean_absolute_error:', mae)
print('mean_squared_error:', mse)
print('root_mean_squared_error:', rmse)
print('r2_score:', r2)

# Evaluate the model on test set

mae = mean_absolute_error(y_test,y_test_pred)
mse = mean_squared_error(y_test, y_test_pred)
rmse = mean_squared_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)
print('mean_absolute_error:', mae)
print('mean_squared_error:', mse)
print('root_mean_squared_error:', rmse)
print('r2_score:', r2)

# using optuna

import optuna
def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 200)
    max_depth = trial.suggest_int('max_depth', 2, 32)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
    max_features = trial.suggest_uniform('max_features', 0.1, 1.0)

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse

# create optuna study

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)
print('Best hyperparameters:', study.best_params)

# train the model with best hyperparameters

best_params = study.best_params
model = RandomForestRegressor(random_state=42, **best_params)
model.fit(X_train, y_train)

y_train_pred_optimized = model.predict(X_train)
y_test_pred_optimized = model.predict(X_test)

# Evaluate the optimized model on training set

mae = mean_absolute_error(y_train,y_train_pred_optimized)
mse = mean_squared_error(y_train, y_train_pred_optimized)
rmse = mean_squared_error(y_train, y_train_pred_optimized)
r2 = r2_score(y_train, y_train_pred_optimized)
print('mean_absolute_error:', mae)
print('mean_squared_error:', mse)
print('root_mean_squared_error:', rmse)
print('r2_score:', r2)

# Evaluate the optimized model on test set

mae = mean_absolute_error(y_test,y_test_pred_optimized)
mse = mean_squared_error(y_test, y_test_pred_optimized)
rmse = mean_squared_error(y_test, y_test_pred_optimized)
r2 = r2_score(y_test, y_test_pred_optimized)
print('mean_absolute_error:', mae)
print('mean_squared_error:', mse)
print('root_mean_squared_error:', rmse)
print('r2_score:', r2)