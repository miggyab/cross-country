import pandas as pd
import numpy as np
from datetime import datetime
import os
import random
from sklearn.metrics import roc_auc_score, cohen_kappa_score, f1_score, confusion_matrix, accuracy_score, \
    precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from statistics import mean
from pathlib import Path
import pickle


def split(folder):
    for f in os.listdir(folder):
        print('splitting {} at {}'.format(f[:-4], datetime.now()))
        passed = list()
        failed = list()

        df = pd.read_csv(os.path.join(folder, f), keep_default_na=False)
        for index, row in df.iterrows():
            if row['achievement_level'] == 'none':
                failed.append(row['user_id'])
            else:
                passed.append(row['user_id'])

        # GET 10% HOLDOUT SET
        num_pass_holdout = round(len(passed) * .1)
        num_fail_holdout = round(len(failed) * .1)

        pass_holdout = random.sample(passed, num_pass_holdout)
        fail_holdout = random.sample(failed, num_fail_holdout)

        # PRINT HOLDOUT SET TO FILE
        holdout_index = df[df['user_id'].isin(pass_holdout) | df['user_id'].isin(fail_holdout)].index
        df.iloc[holdout_index].to_csv(os.path.join('holdout', f), index=False)

        # REMOVE HOLDOUT FROM MAIN DATASET
        df.drop(holdout_index, axis=0).to_csv(os.path.join('merged_final', f), index=False)


def standardize():
    folder = 'merged_final'
    for f in os.listdir(folder):
        print('standardizing {} at {}'.format(f[:-4], datetime.now()))
        df = pd.read_csv(os.path.join(folder, f), keep_default_na=False)
        keys = list(df.columns)[8:]
        for k in keys:
            df[k] = (df[k] - df[k].mean()) / df[k].std()
        df.fillna('.').to_csv(os.path.join('merged_std', f), index=False)


def sort_by_country():
    print('starting to split by country')
    folder = 'merged_std'
    output_folder = 'merged_country'

    countries = list()

    for f in os.listdir(folder):
        slug = f[:-4]
        print('splitting by country in {} at {}'.format(slug, datetime.now()))
        with open(os.path.join(folder, f), 'r') as infile:
            header = next(infile)
            for line in infile:
                tokens = line.split(',')
                country = tokens[6]

                if country not in countries:
                    countries.append(country)
                    with open(os.path.join(output_folder, '{}.csv'.format(country)), 'w+') as outfile:
                        outfile.write(header)

                with open(os.path.join(output_folder, '{}.csv'.format(country)), 'a') as outfile:
                    outfile.write(line)


def pull_country_stats():
    folder = 'merged_country'
    with open('country-stats.csv', 'w+') as outfile:
        outfile.write('country_code,num_learners\n')
        for f in os.listdir(folder):
            print('pulling counts from {} at {}'.format(f[:-4], datetime.now()))
            df = pd.read_csv(os.path.join(folder, f), keep_default_na=False, low_memory=False)
            outfile.write('{},{}\n'.format(f[:-4], len(df.index)))


def within_country():
    folder = 'test'
    folder = 'merged_country'
    outfile_path = 'country-auc-scores.csv'

    clf_names = ['CART', 'RF', 'XGB']
    classifiers = [DecisionTreeClassifier(), RandomForestClassifier(n_estimators=700, n_jobs=-1),
                   XGBClassifier(use_label_encoder=False, verbosity=0, n_jobs=-1)]
    feat_sets = ['inc-only', 'appended']

    with open(outfile_path, 'w+') as outfile:
        outfile.write('country,clf,feat_set,increment,N,N_n,N_p,auc_roc,kappa,accuracy,f1,precision,recall,'
                      'specificity,TP,FP,TN,FN\n')

    for f in os.listdir(folder):
        country = f[:-4]
        print('starting within-country analysis of {} at {}'.format(country, datetime.now()))

        drop = ['course', 'session', 'user_id', 'session_user_id', 'normal_grade', 'achievement_level',
                'country', 'continent']
        df = pd.read_csv(os.path.join(folder, f), low_memory=False).replace('.', np.nan)

        X = df.drop(drop, axis=1)
        # impute missing data with median of column
        for k in list(X.columns):
            X[k].fillna(X[k].median(), inplace=True)
            X[k] = pd.to_numeric(X[k], downcast='float')
        y = [0 if i == 'none' else 1 for i in df['achievement_level'].tolist()]

        for i in range(len(clf_names)):
            for k in range(len(feat_sets)):
                for j in range(1, 9):
                    print('running {} {} models for increment {} using the {} feature set at {}'.format(
                        clf_names[i], country, j, feat_sets[k], datetime.now()))

                    if feat_sets[k] == 'inc-only':
                        inc_drop = [x for x in list(X.columns) if j != int(x[-1])]
                    else:
                        inc_drop = [x for x in list(X.columns) if j < int(x[-1])]
                    X_inc = X.drop(inc_drop, axis=1)

                    '''# hyperparameter tuning
                    if clf_names[i] != 'CART':
                        if clf_names[i] == 'RF':
                            clf_t = [RandomForestClassifier(n_estimators=x) for x in range(100, 1000, 200)]
                        elif clf_names[i] == 'XGB':
                            clf_t = [XGBClassifier(use_label_encoder=False, verbosity=0, n_estimators=x) for x
                                     in range(100, 1000, 200)]

                        for x in range(len(clf_t)):
                            metrics = train_model(X_inc, y, groups, clf_t[x])
                            with open('country-auc-scores.csv', 'a+') as outfile:
                                outfile.write('{},{}_{},{},{},,,'.format(country, clf_names[i], str(x * 200 + 100), feat_sets[k], j))
                                for m in metrics:
                                    outfile.write(',{}'.format(m))
                                outfile.write(',,,,\n')
                    else:
                        metrics = train_model(X_inc, y, groups, clf)
                        # N,N_n,N_p,auc_roc,kappa,accuracy,f1,precision,recall,specificity,TP,FP,TN,FN

                        with open('country-auc-scores.csv', 'a+') as outfile:
                            outfile.write('{},{},{},{},,,'.format(country, clf_names[i], feat_sets[k], j))
                            for m in metrics:
                                outfile.write(',{}'.format(m))
                            outfile.write(',,,,\n')'''

                    clf = classifiers[i]
                    skf = StratifiedKFold(n_splits=10, shuffle=True)

                    fold = 0
                    for train_index, test_index in skf.split(X_inc, y):
                        metrics, clf_trained = train_model_fold(X_inc, y, train_index, test_index, clf)
                        with open(outfile_path, 'a+') as outfile:
                            outfile.write('{},{},{},{}_{}'.format(country, clf_names[i], feat_sets[k], j, fold))
                            for m in metrics:
                                outfile.write(',{}'.format(m))
                            outfile.write('\n')

                        Path(os.path.join('models', country, clf_names[i], feat_sets[k], str(j), str(fold))).mkdir(
                            parents=True, exist_ok=True)
                        filename = os.path.join('models', country, clf_names[i], feat_sets[k], str(j), str(fold),
                                                'model.sav')
                        pickle.dump(clf_trained, open(filename, 'wb'))

                        fold += 1


def train_model_fold(X_inc, y, train_index, test_index, clf):
    metrics = list()

    X_train = X_inc.iloc[train_index]
    X_test = X_inc.iloc[test_index]
    y_train = np.array(y)[train_index.astype(int)]
    y_test = np.array(y)[test_index.astype(int)]

    clf.fit(X_train, y_train)
    predictions_bin = clf.predict(X_test)
    predictions = clf.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, predictions_bin).ravel()

    metrics.append(len(y_test))
    metrics.append(np.bincount(y_test)[0])
    metrics.append(np.bincount(y_test)[1])
    metrics.append(roc_auc_score(y_test, predictions))
    metrics.append(cohen_kappa_score(y_test, predictions_bin))
    metrics.append(accuracy_score(y_test, predictions_bin))
    metrics.append(f1_score(y_test, predictions_bin))
    metrics.append(precision_score(y_test, predictions_bin, zero_division=0))
    metrics.append(recall_score(y_test, predictions_bin))
    metrics.append(tn / (tn + fp))
    metrics.append(tp)
    metrics.append(fp)
    metrics.append(tn)
    metrics.append(fn)

    return metrics, clf


def train_model(X_inc, y, groups, clf):
    gkf = StratifiedKFold(n_splits=10)

    auc_scores = list()
    kappa_scores = list()
    f1_scores = list()
    accuracy_scores = list()
    precision_scores = list()
    recall_scores = list()
    specificity_scores = list()

    for train_index, test_index in gkf.split(X_inc, y, groups=groups):
        X_train = X_inc.iloc[train_index]
        X_test = X_inc.iloc[test_index]
        y_train = np.array(y)[train_index.astype(int)]
        y_test = np.array(y)[test_index.astype(int)]

        clf.fit(X_train, y_train)
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

    print(datetime.now())

    metrics = list()
    metrics.append(mean(auc_scores))
    metrics.append(mean(kappa_scores))
    metrics.append(mean(accuracy_scores))
    metrics.append(mean(f1_scores))
    metrics.append(mean(precision_scores))
    metrics.append(mean(recall_scores))
    metrics.append(mean(specificity_scores))

    return metrics


def main():
    # split('merged')
    # standardize()
    # sort_by_country()
    # pull_country_stats()
    within_country()


if __name__ == '__main__':
    main()
