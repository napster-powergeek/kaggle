# kaggle
# check skewness
# Work done on kaggle 

from scipy.stats import skew  # for some statistics
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax



numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerics2 = []
for i in features.columns:
    if features[i].dtype in numeric_dtypes:
        numerics2.append(i)
skew_features = features[numerics2].apply(lambda x: skew(x)).sort_values(ascending=False)


high_skew = skew_features[skew_features > 0.5]
skew_index = high_skew.index


for i in skew_index:
    features[i] = boxcox1p(features[i], boxcox_normmax(features[i] + 1))
