from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from statsmodels.stats.multitest import multipletests
from scipy.stats import binom_test
from tqdm import tqdm
import pandas as pd
import numpy as np
import shap
import os

import warnings
warnings.filterwarnings("ignore")

class BorutaShap:

    def __init__(self, model=None, importance_measure='Shap', model_type = 'tree',
                classification = True, percentile = 100, pvalue=0.05):
        
        self.importance_measure = importance_measure.lower()
        self.percentile = percentile
        self.pvalue = pvalue
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
        self.X_feature_import, self.Shadow_feature_import = self.feature_importance()
        return self.calculate_hits()


    def calculate_hits(self):
        shadow_threshold = np.percentile(self.Shadow_feature_import,
                                        self.percentile)
        return self.X_feature_import > shadow_threshold


    def create_shadow_features(self):

            self.X_shadow = self.X.apply(np.random.permutation)
            self.X_shadow.columns = ['shadow_' + feature for feature in self.X.columns]
            self.X_boruta = pd.concat([self.X, self.X_shadow], axis = 1)


    @staticmethod
    def calculate_Zscore(array):
        return [(element - np.mean(array)/np.std(array)) for element in array]


    def feature_importance(self):

        if self.importance_measure == 'shap':

            self.explain()
            vals = np.abs(self.shap_values).mean(0)

            vals = self.calculate_Zscore(vals)
            X_feature_import = vals[:len(self.X.columns)]
            Shadow_feature_import = vals[len(self.X_shadow.columns):]

        elif self.importance_measure == 'permutation':

            permuation_importnace_ = permutation_importance(estimator=self.model, X=self.X_boruta, y=self.y)
            permuation_importnace_ = self.calculate_Zscore(permuation_importnace_.importances_mean)
            X_feature_import = permuation_importnace_[:len(self.X.columns)]
            Shadow_feature_import = permuation_importnace_[len(self.X.columns):]
            
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


    @staticmethod
    def binomial_H0_test(array, n, p, alternative):
        return [binom_test(x, n=n, p=p, alternative=alternative) for x in array]

    
    def test_features(self, hits, iteration):

        acceptance_p_values = self.binomial_H0_test(hits,
                                                    n=iteration,
                                                    p=0.5,
                                                    alternative='greater')
        regect_p_values = self.binomial_H0_test(hits,
                                                n=iteration,
                                                p=0.5,
                                                alternative='less')
        
        modified_acceptance_p_values = multipletests(acceptance_p_values,
                                                    alpha=0.05,
                                                    method='bonferroni')

        modified_regect_p_values = multipletests(regect_p_values,
                                                alpha=0.05,
                                                method='bonferroni')

        np.array(modified_regect_p_values) < self.pvalue
        np.array(acceptance_p_values) < self.pvalue

        

        

        



        


def averageOfList(numOfList):
       avg = sum(numOfList) / len(numOfList)
       return avg

if __name__ == "__main__":
    
    current_directory = os.getcwd()

    X = pd.read_csv(current_directory + '\\Datasets\\Ozone.csv')
    y = X.pop('V4')

    hits_natty = np.zeros((len(X.columns)))
    hits_shap   = np.zeros((len(X.columns)))

    history_shap_shadow = np.zeros(len(X.columns))
    history_shap_x = np.zeros(len(X.columns))

    history_shadow = np.zeros(len(X.columns))
    history_x = np.zeros(len(X.columns))
    for trial in tqdm(range(20)):

        np.random.seed(trial+1)
        
        feature_selector = BorutaShap(importance_measure='permutation',
                                      classification=False)
        hits_natty += feature_selector.fit(X, y)

        history_shadow = np.vstack((history_shadow, feature_selector.Shadow_feature_import))
        history_x = np.vstack((history_x, feature_selector.X_feature_import))

        
        feature_selector = BorutaShap(importance_measure='Shap',
                                      classification=False)
        hits_shap += feature_selector.fit(X, y)

        history_shap_shadow = np.vstack((history_shap_shadow, feature_selector.Shadow_feature_import))
        history_shap_x = np.vstack((history_shap_x, feature_selector.X_feature_import))

   

    history_x = pd.DataFrame(data=history_x,
                            columns=X.columns)
    history_x['Max_Shadow'] =  [max(i) for i in history_shadow]
    history_x['Min_Shadow'] =  [min(i) for i in history_shadow]
    history_x['Mean_Shadow'] =  [averageOfList(i) for i in history_shadow]
    history_x.iloc[1:].to_csv('features_ozone.csv', index=False)

    history_shap_x = pd.DataFrame(data=history_shap_x,
                                columns=X.columns)
    history_shap_x['Max_Shadow'] =  [max(i) for i in history_shap_shadow]
    history_shap_x['Min_Shadow'] =  [min(i) for i in history_shap_shadow]
    history_shap_x['Mean_Shadow'] =  [averageOfList(i) for i in history_shap_shadow]
    history_shap_x.iloc[1:].to_csv('Shap_features_ozone.csv', index=False)

    print(hits_natty)
    print(hits_shap)



