# Boruta-Shap
BorutaShap is a wrapper feature selection method which combines both the Boruta feature selection algorithm with shapley values. This combination has proven to out perform the original Permutation Importance method in both speed, and the quality of the feature subset produced. Not only does this algorithm provide a better subset of features, but it can also simultaneously provide the most accurate and consistent global feature rankings which can be used for model inference too. Unlike the orginal R package, which limits the user to a Random Forest model, BorutaShap allows the user to choose any Tree Based learner as the base model in the feature selection process.

Despite BorutaShap's runtime improvments the SHAP TreeExplainer scales linearly with the number of observations making it's use cumbersome for large datasets. To combat this, BorutaShap includes a sampling procedure were a new sub sample of the data can be specifed at each iteration of the algorithm. From experiments, any sub sample greater than 20% with over 30 iterations produced an identical feature subset when compared to using the entire data set. Even with these improvments the user still might want a faster solution so BorutaShap has included an option to use the mean decrease in gini impurity. This importance measure is independent of the size dataset as it uses the tree's structure to compute a global feature ranking making it much faster than SHAP at larger datasets. Although this metric returns somewhat comparable feature subsets, it is not a reliable measure of global feature importance in spite of it's wide spread use. Thus, I would recommend to using the SHAP metric whenever possible.

### Algorithm

1. Start by creating new copies of all the features in the data set and name them shadow + feature_name, shuffle these newly added features to remove their correlations with the response variable.

2. Run a classifier on the extended data with the random shadow features included. Then rank the features using a feature importance metric the original algorithm used permutation importance as it's metric of choice.

3. Create a threshold using the maximum importance score from the shadow features. Then assign a hit to any feature that had exceeded this threshold.

4. For every unassigned feature preform a two sided T-test of equality.

5. Attributes which have an importance significantly lower than the threshold are deemed 'unimportant' and are removed them from process. Deem the attributes which have importance significantly higher than than the threshold as 'important'.

6. Remove all shadow attributes and repeat the procedure until an importance has been assigned for each feature, or the algorithm has reached the previously set limit of runs.

If the algorithm has reached its set limit of runs and an importance has not been assigned to each feature the user has two choices. Either increase the number of runs or use the tentative rough fix function which compares the median importance values between unassigned features and the maximum shadow feature to make the decision.



## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

```bash
pip install BorutaShap
```


## Usage
### Using Shap and Basic Random Forest 
```python
from sklearn.datasets import load_breast_cancer
import BorutaShap

def load_data():
  cancer = load_breast_cancer()
  X = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns = np.append(cancer['feature_names'], ['target']))
  y = X.pop('target')
  return X, y
  
X, y = load_data()

# no model selected default is Random Forest, if classification is False it is a Regression problem
Feature_Selector = BorutaShap(importance_measure='shap',
                              classification=True)

Feature_Selector.fit(X=X, y=y, n_trials=50, random_state=0)

# Returns Boxplot of features
Feature_Selector.plot(which_features='rejected')
# If not all features have been accepted or rejected this function makes a
# decision by comparing median values of the max shadow feature and each tentative feature
Feature_Selector.TentativeRoughFix()
# Returns a subset of the original data with the selected features
subset = Feature_Selector.Subset()
```



## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
