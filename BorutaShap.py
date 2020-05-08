from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from statsmodels.stats.multitest import multipletests
from sklearn.model_selection import cross_val_score
from scipy.stats import binom_test
from tqdm import tqdm
import random
import pandas as pd
import numpy as np
import shap
import os

import warnings
warnings.filterwarnings("ignore")


"""
todo ajust the max min and median shadow features into percentiles if required

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
    

    def fit(self, X, y, n_trials = 20, random_state=0, remove_feature_when_done = True, sample=False, sample_fraction=0.2):
        
        np.random.seed(random_state)
        self.X = X.copy()
        self.y = y.copy()
        self.n_trials = n_trials
        self.sample = sample
        self.fraction = sample_fraction
        self.random_state = random_state
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

        self.store_feature_importance()
        self.calculate_rejected_accepted_tentative()


    def calculate_rejected_accepted_tentative(self):

        self.rejected  = list(set(self.flatten_list(self.rejected_columns))-set(self.flatten_list(self.accepted_columns)))
        self.accepted  = list(set(self.flatten_list(self.accepted_columns)))
        self.tentative = list(set(self.all_columns) - set(self.rejected + self.accepted))

        print(str(len(self.accepted))  + ' attributes confirmed important: ' + str(self.accepted))
        print(str(len(self.rejected))  + ' attributes confirmed unimportant: ' + str(self.rejected))
        print(str(len(self.tentative)) + ' tentative attributes remains: ' + str(self.tentative))



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



    def store_feature_importance(self):

        self.history_x = pd.DataFrame(data=self.history_x,
                                 columns=self.all_columns)
        

        self.history_x['Max_Shadow']    =  [max(i) for i in self.history_shadow]
        self.history_x['Min_Shadow']    =  [min(i) for i in self.history_shadow]
        self.history_x['Mean_Shadow']   =  [np.nanmean(i) for i in self.history_shadow]
        self.history_x['Median_Shadow'] =  [np.nanmedian(i) for i in self.history_shadow]


    def results_to_csv(self, filename='feature_importance'):
    
        self.history_x.dropna(axis=0,inplace=True)
        features = pd.DataFrame(data={'Features':self.history_x.iloc[1:].columns.values,
        'Average Feature Importance':self.history_x.iloc[1:].mean(axis=0).values,
        'Standard Deviation Importance':self.history_x.iloc[1:].std(axis=0).values}).sort_values(by='Average Feature Importance',
                                                                                            ascending=False)
        self.history_x.iloc[1:].to_csv(filename + 'xxx.csv', index=False)
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
        mean_value = np.mean(array)
        std_value  = np.std(array)
        return [(element-mean_value)/std_value for element in array]


    def feature_importance(self):

        if self.importance_measure == 'shap':

            self.explain()
            vals = np.abs(self.shap_values).mean(0)
            vals = self.calculate_Zscore(vals)

            X_feature_import = vals[:len(self.X.columns)]
            Shadow_feature_import = vals[len(self.X_shadow.columns):]

        elif self.importance_measure == 'permutation':
            
            X_train, X_test, y_train, y_test = train_test_split(self.X_boruta, self.y, test_size=0.33, random_state=self.random_state)
            self.model.fit(X=X_train, y=y_train)
            permuation_importnace_ = permutation_importance(estimator=self.model, X=X_test, y=y_test)

            permuation_importnace_ = self.calculate_Zscore(np.abs(permuation_importnace_.importances_mean))
            X_feature_import = permuation_importnace_[:len(self.X.columns)]
            Shadow_feature_import = permuation_importnace_[len(self.X.columns):]

        elif self.importance_measure == 'gini':
            
                feature_importances_ = self.calculate_Zscore(np.abs(self.model.feature_importances_))
                X_feature_import = feature_importances_[:len(self.X.columns)]
                Shadow_feature_import = feature_importances_[len(self.X.columns):]

        else:

            raise ValueError('No Importance_measure was specified select one of (shap, gini, permutation)')

        return X_feature_import, Shadow_feature_import


    def get_sample(self):

        if self.fraction > 1 or self.fraction < 0:
            raise ValueError('Frac must be between 0-1')

        else:
            return self.X_boruta.sample(frac=self.fraction, replace=False, random_state=self.random_state)


    def explain(self):

        if self.model_type == 'tree':
            explainer = shap.TreeExplainer(self.model, approximate=True)
            
            if self.sample:
                if self.classification:
                    # for some reason shap returns values wraped in a list of length 1
                    self.shap_values = explainer.shap_values(self.get_sample())[0]
                else:
                    self.shap_values = explainer.shap_values(self.get_sample())
            else:

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
    def symetric_difference_between_two_arrays(array_one, array_two):
        set_one = set(array_one)
        set_two = set(array_two)
        return np.array(list(set_one.symmetric_difference(set_two)))


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

        #self.features_to_remove = np.concatenate([rejected_features,
                                                  #accepted_features])
        self.features_to_remove = rejected_features


        self.rejected_columns.append(rejected_features)
        self.accepted_columns.append(accepted_features)


    def TentativeRoughFix(self):
        
        median_tentaive_values = self.history_x[self.tentative].median(axis=0).values
        median_max_shadow = self.history_x['Max_Shadow'].median(axis=0)
        

        filtered = median_tentaive_values > median_max_shadow

        self.tentative = np.array(self.tentative)
        newly_accepted = self.tentative[filtered]
        
        if len(newly_accepted) < 1:
            newly_rejected = self.tentative

        else:
            newly_rejected = self.symetric_difference_between_two_arrays(newly_accepted, self.tentative)

        print(str(len(newly_accepted)) + ' tentative features are now accepted: ' + str(newly_accepted))
        print(str(len(newly_rejected)) + ' tentative features are now rejected: ' + str(newly_rejected))

        self.rejected = self.rejected + newly_rejected.tolist()
        self.accepted = self.accepted + newly_accepted.tolist()


    def Subset(self, X):
        X = X.copy()
        return X[self.accepted]



if __name__ == "__main__":
    
    from sklearn.model_selection import train_test_split

    current_directory = os.getcwd()

    X = pd.read_csv(current_directory + '\\Datasets\\Madelon.csv')
    #y = X.pop('V4')
    #print(X.columns)
    y = X.pop('decision')


    Feature_Selector = BorutaShap(model=None, importance_measure='shap',
                model_type='tree', classification=True, percentile=100,
                pvalue=0.05)

    Feature_Selector.fit(X=X, y=y, n_trials=20, random_state=0, remove_feature_when_done=True,
                        sample_fraction=0.1, sample=True)

    Feature_Selector.TentativeRoughFix()
    Feature_Selector.results_to_csv(filename='perm_importance_madelon_test')
  
    print(Feature_Selector.hits)

    X = Feature_Selector.Subset(X)

    model = RandomForestClassifier()
    scores = cross_val_score(model, X, y, cv=5)
    print(scores)
    print(scores.mean())


    


