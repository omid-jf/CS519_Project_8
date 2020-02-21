# ======================================================================= 
# This file is part of the CS519_Project_8 project.
#
# Author: Omid Jafari - omidjafari.com
# Copyright (c) 2018
#
# For the full copyright and license information, please view the LICENSE
# file that was distributed with this source code.
# =======================================================================

import sys
from time import time
import inspect
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier


class Classifiers(object):
    # Constructor
    def __init__(self, criterion="gini", max_depth=None, n_estimators=25, max_samples=1.0, max_features=1.0,
                 bootstrap=True, bootstrap_features=False, n_jobs=1, learning_rate=0.1,
                 seed=1, x_tr=[], y_tr=[], x_ts=[]):
        self.criterion = criterion
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.n_jobs = n_jobs
        self.learning_rate = learning_rate
        self.seed = seed
        self.x_tr = x_tr
        self.y_tr = y_tr
        self.x_ts = x_ts
        self.__obj = None

    def call(self, method):
        return getattr(self, method)()

    def __fit(self):
        start = int(round(time() * 1000))
        self.__obj.fit(self.x_tr, self.y_tr)
        end = int(round(time() * 1000)) - start
        print(inspect.stack()[1][3].split("_", 1)[1] + " training time: " + str(end) + " ms")

    def __predict(self):
        start = int(round(time() * 1000))
        y_tr_pred = self.__obj.predict(self.x_tr)
        end = int(round(time() * 1000)) - start
        print(inspect.stack()[1][3].split("_", 1)[1] + " prediction time of training data: " + str(end) + " ms")

        start = int(round(time() * 1000))
        y_ts_pred = self.__obj.predict(self.x_ts)
        end = int(round(time() * 1000)) - start
        print(inspect.stack()[1][3].split("_", 1)[1] + " prediction time of testing data: " + str(end) + " ms")

        return y_tr_pred, y_ts_pred

    def run_decisiontree(self):
        self.__obj = DecisionTreeClassifier(criterion=self.criterion, random_state=self.seed, max_depth=self.max_depth)
        self.__fit()
        return self.__predict()

    def run_randforest(self):
        self.__obj = RandomForestClassifier(criterion=self.criterion, n_estimators=self.n_estimators, random_state=self.seed)
        self.__fit()
        return self.__predict()

    def run_bagging(self):
        tree = DecisionTreeClassifier(criterion=self.criterion, random_state=self.seed, max_depth=self.max_depth)
        self.__obj = BaggingClassifier(base_estimator=tree, n_estimators=self.n_estimators,
                                       max_samples=self.max_samples, max_features=self.max_features,
                                       bootstrap=self.bootstrap, bootstrap_features=self.bootstrap_features,
                                       n_jobs=self.n_jobs, random_state=self.seed)
        self.__fit()
        return self.__predict()

    def run_adaboost(self):
        tree = DecisionTreeClassifier(criterion=self.criterion, random_state=self.seed, max_depth=self.max_depth)
        self.__obj = AdaBoostClassifier(base_estimator=tree, n_estimators=self.n_estimators,
                                        learning_rate=self.learning_rate, random_state=self.seed)
        self.__fit()
        return self.__predict()