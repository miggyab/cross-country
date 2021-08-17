import math
import scipy.stats
import pandas as pd


def se(auc, n_p, n_n, dp, dn):
    return math.sqrt((auc * (1 - auc) + dp + dn) / (n_p * n_n))


def get_d(auc, n_p, n_n):
    return (n_p - 1) * (auc / (2 - auc) - pow(auc, 2)), \
           (n_n - 1) * ((2 * pow(auc, 2)) / (1 + auc) - pow(auc, 2))


def auc_test(auc1, auc2, n_p, n_n):
    dp1, dn1 = get_d(auc1, n_p, n_n)
    se1 = se(auc1, n_p, n_n, dp1, dn1)

    dp2, dn2 = get_d(auc2, n_p, n_n)
    se2 = se(auc1, n_p, n_n, dp2, dn2)

    z = (auc1 - auc2) / math.sqrt(pow(se1, 2) + pow(se2, 2))
    p = scipy.stats.norm.sf(abs(z)) * 2

    level = 1

    if abs(z) > 3.819:
        level = 0.0001
    elif abs(z) > 3.291:
        level = 0.001
    elif abs(z) > 2.576:
        level = 0.01
    elif abs(z) > 1.96:
        level = 0.05

    is_sig = True if level < 1 else False

    return z, p, is_sig


def commit_to_dict(filename):
    aucs = dict()
    n_s = dict()

    df = pd.read_csv(filename)
    for index, row in df.iterrows():
        country = row['country']
        clf = row['clf']
        feat_set = row['feat_set']
        auc = row['auc_score']

        if country not in aucs:
            aucs[country] = dict()
            n_s[country] = [row['N_p'], row['N_n']]

        if clf not in aucs[country]:
            aucs[country][clf] = dict()

        if feat_set not in aucs[country][clf]:
            aucs[country][clf][feat_set] = list()

        aucs[country][clf][feat_set].append(auc)

    return aucs, n_s


def main():
    filename = 'inc_metrics.csv'
    outfilename = 'best_models.csv'
    compfilename = 'comparisons.csv'
    aucs, n = commit_to_dict(filename)
    best_models = dict()

    with open(outfilename, 'w+') as outfile:
        outfile.write('country,clf,feat_set,inc,auc_score,n_p,n_n\n')

    with open(compfilename, 'w+') as outfile:
        outfile.write('country,clf,feat_set,inc_i,inc_j,auc_i,auc_j,z-score,p-val,n_p,n_n\n')

    for country in aucs:
        n_p = n[country][0]
        n_n = n[country][1]

        # exclude smaller countries
        # if n_n + n_p < 200:
            # continue

        best_models[country] = [0.0, list()]

        for clf in aucs[country]:
            for feat_set in aucs[country][clf]:
                for i in range(8):
                    auc_i = aucs[country][clf][feat_set][i]
                    all_not_sig = True
                    for j in reversed(range(8)):
                        if i == j:
                            break

                        auc_j = aucs[country][clf][feat_set][j]

                        z, p, is_sig = auc_test(auc_i, auc_j, n_p, n_n)

                        with open(compfilename, 'a+') as outfile:
                            outfile.write('{},{},{},{},{},{},{},{},{},{},{}\n'.format(
                                country, clf, feat_set, (i + 1), (j + 1), auc_i, auc_j, z, p, n_n, n_p))

                        if p < 0.05:
                            all_not_sig = False
                            break

                    if all_not_sig:
                        if auc_i > best_models[country][0]:
                            best_models[country] = [auc_i, [clf, feat_set, (i + 1)]]
                        with open(outfilename, 'a+') as outfile:
                            outfile.write('{},{},{},{},{},{},{}\n'.format(country, clf, feat_set, (i + 1), auc_i, n_p, n_n))
                        break

    with open('best_of_the_best.csv', 'w+') as outfile:
        outfile.write('country,clf,feat_set,inc,auc_score\n')
        for country in best_models:
            outfile.write('{}'.format(country))
            for i in range(3):
                outfile.write(',{}'.format(best_models[country][1][i]))
            outfile.write(',{}\n'.format(best_models[country][0]))


if __name__ == '__main__':
    main()
