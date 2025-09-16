import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

wine = load_wine()
X = wine.data
y = wine.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA()),
    ('clf', RandomForestClassifier())
])

param_grid = { 'pca__n_components': [2, 5, 10],
    'clf__n_estimators': [50, 100],
    'clf__max_depth': [3, 5, None]
               }
grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

print("Best parameters found: ", grid_search.best_params_)
best_model = grid_search.best_estimator_
test_accuracy = best_model.score(X_test, y_test)
print(f"Accuracy on test data: {test_accuracy:.4f}")

with open('wine_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

