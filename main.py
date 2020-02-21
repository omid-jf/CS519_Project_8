# ======================================================================= 
# This file is part of the CS519_Project_8 project.
#
# Author: Omid Jafari - omidjafari.com
# Copyright (c) 2018
#
# For the full copyright and license information, please view the LICENSE
# file that was distributed with this source code.
# =======================================================================

import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from classifiers import Classifiers
from sklearn.metrics import accuracy_score


for dataset in ["digits", "mammographic"]:

    # ## Preprocessing ##
    # Digits dataset
    if dataset == "digits":
        print("\n\n************")
        print("Digits dataset")
        print("************")

        # Loading the dataset
        digits = datasets.load_digits()
        x = digits.data
        y = digits.target

    # Mammographic dataset
    elif dataset == "mammographic":
        print("\n\n************")
        print("Mammographic dataset")
        print("************")

        # Loading the dataset
        df = pd.read_csv("mammographic_masses.data", header=None, names=["BIRADS", "Age", "Shape", "Margin", "Density",
                                                                         "Severity"])
        df.replace("?", np.NaN, inplace=True)
        x = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        # Replacing missing values with the mean of the column
        imp = SimpleImputer(missing_values=np.NaN, strategy="mean")
        x = imp.fit_transform(X=x, y=y)

    # Standardizing data
    sc_x = StandardScaler()
    x_std = sc_x.fit_transform(x)

    # Splitting data
    x_std_tr, x_std_ts, y_tr, y_ts = train_test_split(x_std, y, test_size=0.3, random_state=1)

    # Running classifiers
    classifier = Classifiers(criterion="gini", max_depth=None, n_estimators=25, max_samples=1.0, max_features=1.0,
                             bootstrap=True, bootstrap_features=False, n_jobs=1, learning_rate=0.1,
                             seed=1, x_tr=x_std_tr, y_tr=y_tr, x_ts=x_std_ts)

    for name in ["decisiontree", "randforest", "bagging", "adaboost"]:
        print("\n\n" + name)
        y_tr_pred, y_ts_pred = classifier.call("run_" + name)

        train_error = accuracy_score(y_tr, y_tr_pred)
        test_error = accuracy_score(y_ts, y_ts_pred)

        print("%s - train accuracy: %.3f - test accuracy: %.3f" % (name, train_error, test_error))
