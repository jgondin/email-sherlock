from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.externals import joblib


if __name__ == "__main__":
    # Load Iris Data
    iris_data = load_iris()
    features = iris_data.data
    feature_names = iris_data.feature_names
    target = iris_data.target
    target_names = iris_data.target_names

    knn = KNeighborsClassifier(n_neighbors=3)  # replace with your own ML model here
    knn.fit(features, target)

    joblib.dump(knn, 'models/iris_model.pkl')
