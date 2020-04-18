from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import pandas as pd
import numpy as np
import shap
import os

import warnings
warnings.filterwarnings("ignore")

class BorutaShap:

    def __init__(self, model=None, Shap=True, model_type = 'tree', classification = True):
        
        self.Shap = Shap
        self.classification = classification
        self.model = model
        self.model_type = model_type
        self.check_model()


    def check_model(self):

        check_fit = hasattr(self.model, 'fit')
        check_predict_proba = hasattr(self.model, 'predict')

        if self.model is None:
            if self.classification:
                self.model = RandomForestClassifier()
            else:
                self.model = RandomForestRegressor()
        
        elif check_fit is False and check_predict_proba is False:
            raise AttributeError('Model must contain both the fit() and predict() methods')

        else:
            pass


    def check_X(self):

        if isinstance(self.X, pd.DataFrame) is False:
            raise AttributeError('X must be a pandas Dataframe')

        else:
            pass


    def check_missing_values(self):

        X_missing = self.X.isnull().any().any()
        Y_missing = self.y.isnull().any().any()

        if X_missing or Y_missing:
            raise ValueError('There are missing values in your Data')
        
        else:
            pass


    def fit(self, X, y, n_trials = 20):
        
        self.X = X
        self.y = y
        
        self.check_X()
        self.check_missing_values()
        self.create_shadow_features()
        self.model.fit(self.X_boruta, self.y)
        X_feature_import, Shadow_feature_import = self.feature_importance()
        print(X_feature_import > Shadow_feature_import.max())


    
    def create_shadow_features(self):

            self.X_shadow = self.X.apply(np.random.permutation)
            self.X_shadow.columns = ['shadow_' + feature for feature in self.X.columns]
            self.X_boruta = pd.concat([self.X, self.X_shadow], axis = 1)


    def feature_importance(self):

        if self.Shap:

            self.explain()
            vals = np.abs(self.shap_values).mean(0)

            X_feature_import = vals[:len(self.X.columns)]
            Shadow_feature_import = vals[len(self.X_shadow.columns):]

        else:

            X_feature_import = self.model.feature_importances_[:len(self.X.columns)]
            Shadow_feature_import = self.model.feature_importances_[len(self.X.columns):]

        return X_feature_import, Shadow_feature_import


    def explain(self):

        if self.model_type == 'tree':
            explainer = shap.TreeExplainer(self.model)
            
            if self.classification:
                # for some reason shap returns values wraped in a list of length 1
                self.shap_values = explainer.shap_values(self.X_boruta)[0]
            else:
                self.shap_values = explainer.shap_values(self.X_boruta)

        elif self.model_type == 'linear':
            explainer = shap.LinearExplainer(self.model, self.X_boruta, feature_dependence="independent")
            self.shap_values = explainer.shap_values(self.X_boruta)

        else:
            raise AttributeError("Model Type has not been Selected (linear or tree)")
            


if __name__ == "__main__":
    
    np.random.seed(56)
    current_directory = os.getcwd()

    X = pd.read_csv(current_directory + '\\Datasets\\Ozone.csv')
    y = X.pop('V4')

    feature_selector = BorutaShap(Shap=False, classification=False)
    feature_selector.fit(X, y)

    feature_selector = BorutaShap(Shap=True, classification=False)
    feature_selector.fit(X, y)


