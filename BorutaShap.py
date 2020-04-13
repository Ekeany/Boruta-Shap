from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import shap


class BorutaShap:


    def __init__(self, model=None, Shap=True):

        self.Shap = Shap

        if isinstance(model, RandomForestClassifier):
            self.model = model

        else:
            self.model = RandomForestClassifier()

    
    def fit(self, X, y, n_trials = 20):
        
        self.X = X
        self.y = y
        
        self.check_X()
        self.x_boruta = self.create_shadow_features()
        self.model.fit(self.x_boruta, self.y)


        pass

    
    def create_shadow_features(self):

        if self.x_type == "pandas":

            self.X_shadow = self.X.apply(np.random.permutation)
            self.X_shadow.columns = ['shadow_' + feature for feature in self.X.columns]
            X_boruta = pd.concat([self.X, X_shadow], axis = 1)

        else:
            pass

        return X_boruta
    

    def check_X(self):

        if isinstance(self.X, pd.DataFrame):
            self.x_type == "pandas"

        elif isinstance(self.X, np.array):
            self.x_type == "numpy"

        else:
            raise AttributeError('X must be a pandas Dataframe or Numpy array')


    def feature_importance(self):

        if self.Shap:

            shap_values = self.explainer()
            
            vals = np.abs(shap_values).mean(0)
            feature_importance = pd.DataFrame(list(zip(self.x_boruta.columns,vals)), columns=['col_name','feature_importance_vals'])
            X_feature_import = feature_importance[self.X.columns]
            Shadow_feature_import = feature_importance[self.X_shadow.columns]

        else:
            X_feature_import = self.model.feature_importances_[:len(self.X.columns)]
            Shadow_feature_import = self.model.feature_importances_[len(self.X.columns):]

        return X_feature_import, Shadow_feature_import


    def explainer(self):
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(self.x_boruta)
        return shap_values





