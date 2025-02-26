# Creating Numerical attribute Pipeline
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer, OneHotEncoder
import numpy as np

# pipeline to process numerical attributes
num_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy = "median")),
    ("standardize", StandardScaler())
])

# pipeline to process categorical attributes
cat_pipeline = Pipeline([
    ("impute_cat", SimpleImputer(strategy = "most_frequent")),
    ("1hot_encode", OneHotEncoder(handle_unknown = "ignore"))
])

# pipeline to produce ratio of attributes
def column_ratio(X):
    return X[:, [0]] / X[: , [1]]

# FunctionTransformer will call this with two arguments
def ratio_name(function_transformer, feature_names_in):
    return ["ratio"] # feature names out

def ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy = "median"),
        FunctionTransformer(column_ratio, feature_names_out = ratio_name),
        StandardScaler()
    )

# pipeline to handel heavy tail in the data using Log of the feature
log_pipeline = make_pipeline(
    SimpleImputer(strategy = "median"),
    FunctionTransformer(np.log, feature_names_out = "one-to-one"),
    StandardScaler()
)


# need to sub-class BaseEstimator , TransfomerMixin
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import rbf_kernel

class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters = 10, gamma = 1.0,
                 random_state = None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y = None, sample_weight = None): # y is required even if not used
        
        # Features that is learned during fit is followed by underscore(kmeans_)
        self.kmeans_ = KMeans(self.n_clusters, n_init = 10, 
                             random_state = self.random_state)

        # fitting using KMeans transformer not our implementation
        self.kmeans_.fit(X, sample_weight = sample_weight)
        return self # always return self

    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, #comes after KMeans is performed
                          gamma = self.gamma)

    def get_feature_names_out(self, names = None):
        # for each cluster name will be Cluster1 similarity, Cluster2 similarity etc
        return [f"Cluster{i} similarity" for i in range(self.n_clusters)]

cluster_simil = ClusterSimilarity(n_clusters = 10, gamma = 1., random_state = 42)



# Creating ColumnTransformer to apply the transformations on different columns
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector


# final preprocessing pipeline
preprocessing = ColumnTransformer([
    ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
    ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
    ("people_per_house", ratio_pipeline(), ["population", "households"]),
    ("log", log_pipeline, ["total_bedrooms","total_rooms", "population",
                           "households", "median_income"]),
    ("geo", cluster_simil,["latitude","longitude"]),
    ("cat", cat_pipeline, make_column_selector(dtype_include = object)),
], remainder = num_pipeline) # all remaining columns will use this
preprocessing