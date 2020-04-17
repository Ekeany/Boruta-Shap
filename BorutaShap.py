from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import shap


class BorutaShap:

    def __init__(self, model=None, Shap=True, model_type = 'tree'):

        self.Shap = Shap
        self.model_type = model_type
        self.check_model()


    def check_model(self):

        check_fit = hasattr(self.model, 'fit')
        check_predict_proba = hasattr(self.model, 'predict_proba')

        if self.model is None:
            self.model = RandomForestClassifier()
        
        elif check_fit is False and check_predict_proba is False:
            raise AttributeError('Model must contain both the fit() and predict_proba() methods')

        else:
            self.model = model


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


    
    def create_shadow_features(self):

            self.X_shadow = self.X.apply(np.random.permutation)
            self.X_shadow.columns = ['shadow_' + feature for feature in self.X.columns]
            self.X_boruta = pd.concat([self.X, self.X_shadow], axis = 1)


    def feature_importance(self):

        if self.Shap:

            self.explainer()
            
            vals = np.abs(self.shap_values).mean(0)
            feature_importance = pd.DataFrame(list(zip(self.x_boruta.columns,vals)),
                                              columns=['col_name','feature_importance_vals'])
            X_feature_import = feature_importance[self.X.columns]
            Shadow_feature_import = feature_importance[self.X_shadow.columns]

        else:
            X_feature_import = self.model.feature_importances_[:len(self.X.columns)]
            Shadow_feature_import = self.model.feature_importances_[len(self.X.columns):]

        return X_feature_import, Shadow_feature_import


    def explainer(self):

        if self.model_type == 'tree':
            explainer = shap.TreeExplainer(self.model)
            self.shap_values = explainer.shap_values(self.x_boruta)
        elif
        


if __name__ == "__main__":

    forest = RandomForestClassifier()
    BorutaShap(model=forest)


