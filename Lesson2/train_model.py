# train_model.py (local only)
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

X, y = load_iris(return_X_y=True)
clf = RandomForestClassifier(n_estimators=10, random_state=42)
clf.fit(X, y)
joblib.dump(clf, "app/model.pkl")
print("Saved model to app/model.pkl")
