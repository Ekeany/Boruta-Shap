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


"""
todo 
when doing the tests as the hits still are giving zero for certain features if once accepted are
now being rejected.

update tehe hits by making it smaller

remove z scores
create and way to store all of the shap scores 

"""




class BorutaShap:

    def __init__(self, model=None, importance_measure='Shap', model_type='tree',
                classification=True, percentile=100, pvalue=0.05):

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
    

    def fit(self, X, y, n_trials = 20, random_state=0, remove_feature_when_done = True):
        
        np.random.seed(random_state)
        self.X = X
        self.y = y
        self.n_trials = n_trials
        self.ncols = self.X.shape[1]
        self.remove_feature_when_done = remove_feature_when_done
        self.all_columns = self.X.columns.to_numpy()
        self.rejected_columns = []
        self.accepted_columns = []
        
        self.check_X()
        self.check_missing_values()

        self.features_to_remove = []
        self.hits  = np.zeros(self.ncols)
        self.order = self.create_mapping_between_cols_and_indices()
        self.create_importance_history()
        for trial in tqdm(range(self.n_trials)):
            
            if self.remove_feature_when_done:
                self.remove_features_if_rejected_or_accepted()
            else:
                pass

            self.columns = self.X.columns.to_numpy()
            self.create_shadow_features()

            if self.X.shape[1] == 0:
                break

            else:
                self.model.fit(self.X_boruta, self.y)
                self.X_feature_import, self.Shadow_feature_import = self.feature_importance()
                self.update_importance_history()
                self.hits += self.calculate_hits()
                self.test_features(iteration=trial+1)

        print(self.calculate_rejected_accepted_tentative())


    def calculate_rejected_accepted_tentative(self):

        rejected  = list(set(self.flatten_list(self.rejected_columns))-set(self.flatten_list(self.accepted_columns)))
        accepted  = list(set(self.flatten_list(self.accepted_columns)))
        tentative = list(set(self.all_columns) - set(rejected+accepted))

        return accepted, rejected, tentative


    def create_importance_history(self):

        self.history_shadow = np.zeros(self.ncols)
        self.history_x = np.zeros(self.ncols)

    
    def update_importance_history(self):

        padded_history_shadow  = np.full((self.ncols), np.NaN)
        padded_history_x = np.full((self.ncols), np.NaN)

        for (index, col) in enumerate(self.columns):
            map_index = self.order[col]
            padded_history_shadow[map_index] = self.Shadow_feature_import[index]
            padded_history_x[map_index] = self.X_feature_import[index]

        self.history_shadow = np.vstack((self.history_shadow, padded_history_shadow))
        self.history_x = np.vstack((self.history_x, padded_history_x))


    def results_to_csv(self, filename='feature_importance'):

        history_x = pd.DataFrame(data=self.history_x,
                                 columns=self.all_columns)

        
        history_x['Max_Shadow'] =  [max(i) for i in self.history_shadow]
        history_x['Min_Shadow'] =  [min(i) for i in self.history_shadow]
        history_x['Mean_Shadow'] =  [self.average_of_list(i) for i in self.history_shadow]
        

        features = pd.DataFrame(data={'Features':history_x.iloc[1:].columns.values,
        'Average Feature Importance':history_x.iloc[1:].mean(axis=0).values,
        'Standard Deviation Importance':history_x.iloc[1:].std(axis=0).values}).sort_values(by='Average Feature Importance',
                                                                                            ascending=False)

        features.to_csv(filename + '.csv', index=False)


    def remove_features_if_rejected_or_accepted(self):

        if len(self.features_to_remove) != 0:
            for feature in self.features_to_remove:
                try:
                    self.X.drop(feature, axis = 1, inplace=True)
                except:
                    pass
        
        else:
            pass
    

    @staticmethod
    def average_of_list(lst):
        return sum(lst) / len(lst) 

    @staticmethod
    def flatten_list(array):
        return [item for sublist in array for item in sublist]


    def create_mapping_between_cols_and_indices(self):
        return dict(zip(self.X.columns.to_list(), np.arange(self.X.shape[1])))


    def calculate_hits(self):
        shadow_threshold = np.percentile(self.Shadow_feature_import,
                                        self.percentile)
        
        padded_hits = np.zeros(self.ncols)
        hits = self.X_feature_import > shadow_threshold

        for (index, col) in enumerate(self.columns):
            map_index = self.order[col]
            padded_hits[map_index] += hits[index]

        return padded_hits


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

            if not self.remove_feature_when_done:
                vals = self.calculate_Zscore(vals)
            else:
                pass

            X_feature_import = vals[:len(self.X.columns)]
            Shadow_feature_import = vals[len(self.X_shadow.columns):]

        elif self.importance_measure == 'permutation':

            permuation_importnace_ = permutation_importance(estimator=self.model, X=self.X_boruta, y=self.y)
            
            if not self.remove_feature_when_done:
                permuation_importnace_ = self.calculate_Zscore(permuation_importnace_.importances_mean)
            else:
                permuation_importnace_ = permuation_importnace_.importances_mean
            
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


    @staticmethod
    def find_index_of_true_in_array(array):
        length = len(array)
        return list(filter(lambda x: array[x], range(length)))
    

    @staticmethod
    def bonferoni_corrections(pvals, alpha=0.05, n_tests=None):

        pvals = np.array(pvals)
        
        if n_tests is None:
            n_tests = len(pvals)
        else:
            pass
        
        alphacBon = alpha / float(n_tests)
        reject = pvals <= alphacBon
        pvals_corrected = pvals * float(n_tests)
        return reject, pvals_corrected


    @staticmethod
    def FDR_correction(self):
        pass


    def test_features(self, iteration):

        acceptance_p_values = self.binomial_H0_test(self.hits,
                                                    n=iteration,
                                                    p=0.5,
                                                    alternative='greater')
                                                    
        regect_p_values = self.binomial_H0_test(self.hits,
                                                n=iteration,
                                                p=0.5,
                                                alternative='less')
        
        # [1] as function returns a tuple 
        modified_acceptance_p_values = self.bonferoni_corrections(acceptance_p_values,
                                                                  alpha=0.05,
                                                                  n_tests=len(self.columns))[1]

        modified_regect_p_values = self.bonferoni_corrections(regect_p_values,
                                                              alpha=0.05,
                                                              n_tests=len(self.columns))[1]

        # Take the inverse as we want true to keep featrues
        rejected_columns = np.array(modified_regect_p_values) < self.pvalue
        accepted_columns = np.array(modified_acceptance_p_values) < self.pvalue

        rejected_indices = self.find_index_of_true_in_array(rejected_columns)
        accepted_indices = self.find_index_of_true_in_array(accepted_columns)

        rejected_features = self.all_columns[rejected_indices]
        accepted_features = self.all_columns[accepted_indices]

        self.features_to_remove = np.concatenate([rejected_features,
                                                  accepted_features])


        self.rejected_columns.append(rejected_features)
        self.accepted_columns.append(accepted_features)




if __name__ == "__main__":
    
    current_directory = os.getcwd()

    X = pd.read_csv(current_directory + '\\Datasets\\Ozone.csv')
    y = X.pop('V4')

    Feature_Selector = BorutaShap(model=None, importance_measure='Shap',
                model_type='tree', classification=False, percentile=100,
                pvalue=0.05)

    Feature_Selector.fit(X=X, y=y, n_trials=40, random_state=0, remove_feature_when_done=False)

    Feature_Selector.results_to_csv(filename='Shap_importance')
    print(Feature_Selector.hits)



    


