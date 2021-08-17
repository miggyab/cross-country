import os
import pandas as pd
from datetime import datetime as dt
import numpy as np
from sklearn.metrics import roc_auc_score, cohen_kappa_score, f1_score, confusion_matrix, accuracy_score, \
    precision_score, recall_score
from statistics import mean
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle
from pathlib import Path


def prepare_data(df, feat_set, inc):
    drop = ['course', 'session', 'user_id', 'session_user_id', 'normal_grade', 'achievement_level',
            'country', 'continent']

    X = df.drop(drop, axis=1)

    if feat_set == 'inc-only':
        inc_drop = [x for x in list(X.columns) if inc != int(x[-1])]
    else:
        inc_drop = [x for x in list(X.columns) if inc < int(x[-1])]
    X = X.drop(inc_drop, axis=1)

    # impute missing data with median of column
    for k in list(X.columns):
        X[k].fillna(X[k].median(), inplace=True)
        X[k] = pd.to_numeric(X[k], downcast='float')

    y = [0 if i == 'none' else 1 for i in df['achievement_level'].tolist()]

    return X, y


def get_best_train_model(filename):
    model_params = dict()
    df = pd.read_csv(filename)
    for index, row in df.iterrows():
        model_params[row['country']] = [row['clf'], row['feat_set'], row['inc'], row['auc_score']]
    return model_params


def pickle_best_models(folder):
    model_params = get_best_train_model('best_of_the_best.csv')
    clfs = [RandomForestClassifier(n_estimators=700), XGBClassifier(use_label_encoder=False, verbosity=0)]

    for f in os.listdir(folder):
        country = f[:-4]
        Path(os.path.join('models', country)).mkdir(parents=True, exist_ok=True)

        clf = clfs[0] if model_params[country][0] == 'RF' else clfs[1]
        feat_set = model_params[country][1]
        inc = model_params[country][2]

        df = pd.read_csv(os.path.join(folder, f), low_memory=False).replace('.', np.nan)
        X, y = prepare_data(df, feat_set, inc)

        skf = StratifiedKFold(n_splits=10, shuffle=True)

        print('pickling {} models at {}'.format(country, dt.now()))

        fold = 1
        for train_index, test_index in skf.split(X, y):
            print('pickling {}-{} at {}'.format(country, fold, dt.now()))

            X_train = X.iloc[train_index]
            y_train = np.array(y)[train_index.astype(int)]

            clf.fit(X_train, y_train)

            filename = os.path.join('models', country, 'model-{}.sav'.format(fold))
            pickle.dump(clf, open(filename, 'wb'))
            fold += 1


def get_cross_auc(folder):
    model_params = get_best_train_model('best_of_the_best.csv')

    outpath = 'cross_country.csv'
    with open(outpath, 'w+') as outfile:
        outfile.write('train_country,test_country,auc_score,distance,kappa,accuracy,f1,precision,recall,specificity\n')

    for train_file in os.listdir(folder):
        train_country = train_file[:-4]

        feat_set = model_params[train_country][1]
        inc = model_params[train_country][2]
        train_auc = model_params[train_country][3]

        for test_file in os.listdir(folder):
            if train_file == test_file:
                continue

            test_country = test_file[:-4]

            print('starting cross-country analysis of {} and {} at {}'.format(train_country, test_country, dt.now()))

            df_test = pd.read_csv(os.path.join(folder, test_file), low_memory=False).replace('.', np.nan)
            X_test, y_test = prepare_data(df_test, feat_set, inc)

            auc_scores = list()
            kappa_scores = list()
            f1_scores = list()
            accuracy_scores = list()
            precision_scores = list()
            recall_scores = list()
            specificity_scores = list()

            for f in os.listdir(os.path.join('models', train_country)):
                print('testing {}-{} on {} at {}'.format(train_country, f, test_country, dt.now()))

                filename = os.path.join('models', train_country, f)
                clf = pickle.load(open(filename, 'rb'))
                predictions_bin = clf.predict(X_test)
                predictions = clf.predict_proba(X_test)[:, 1]

                auc_scores.append(roc_auc_score(y_test, predictions))
                kappa_scores.append(cohen_kappa_score(y_test, predictions_bin))
                accuracy_scores.append(accuracy_score(y_test, predictions_bin))
                f1_scores.append(f1_score(y_test, predictions_bin))
                precision_scores.append(precision_score(y_test, predictions_bin))
                recall_scores.append(recall_score(y_test, predictions_bin))

                tn, fp, fn, tp = confusion_matrix(y_test, predictions_bin).ravel()

                specificity_scores.append(tn / (tn + fp))

            with open(outpath, 'a+') as outfile:
                auc_score = mean(auc_scores)
                distance = abs(train_auc - auc_score)
                outfile.write('{},{},{},{},{},{},{},{},{},{}\n'.format(
                    train_country, test_country, auc_score, distance, mean(kappa_scores), mean(accuracy_scores),
                    mean(f1_scores), mean(precision_scores), mean(recall_scores), mean(specificity_scores)))


folder = 'merged_country'
# pickle_best_models(folder)
get_cross_auc(folder)
