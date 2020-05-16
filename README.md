# Boruta-Shap
A model agnostic feature selection tool which combines both the Boruta feature selection algorithm with shapley values.  


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

cancer = load_breast_cancer()
X = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns = np.append(cancer['feature_names'], ['target']))
y = X.pop('target')

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
