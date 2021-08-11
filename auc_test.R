library(auctestr)
library(hash)

df = read.csv('inc_metrics.csv', na.strings=".")

auc_scores = hash()
for (row in 1:nrow(df)) {
  country = df[row, 'country']
  clf = df[row, 'clf']
  feat_set = df[row, 'feat_set']
  auc_score = df[row, 'auc_score']
  
  if (!has.key(country, auc_scores)) {
     auc_scores[[country]] = hash()
   }
  
  if (!has.key(clf, auc_scores[[country]])) {
    auc_scores[[country]][[clf]] = hash()
  }
  
  if (!has.key(feat_set, auc_scores[[country]][[clf]])) {
    auc_scores[[country]][[clf]][[feat_set]] = list()
  }
  
  auc_scores[[country]][[clf]][[feat_set]] = append(auc_scores[[country]][[clf]][[feat_set]], auc_score)
}

country_ns = hash()
for (row in 1:nrow(df)) {
  country = df[row, 'country']
  
  if (!has.key(country, country_ns)) {
    country_ns[[country]] = list(df[row, 'N_p'], df[row, 'N_n'])
  }
}

for (country in ls(auc_scores)) {
  if (country == 'US') { next }
  for (clf in ls(auc_scores[[country]])) {
    for (feat_set in ls(auc_scores[[country]][[clf]])) {
      for (i in 1:7) {
        all_not_significant = TRUE
        for (j in rev(seq(1:8))) {
          if (i == j) { break }
          
          auc_i = auc_scores[[country]][[clf]][[feat_set]][[i]]
          auc_j = auc_scores[[country]][[clf]][[feat_set]][[j]]
          
          if (auc_i > auc_j) { next }
          
          z = fbh_test(auc_i, auc_j, country_ns[[country]][[1]], country_ns[[country]][[2]])
          
          if (abs(z) > 3.819) {
            all_not_significant = FALSE
            print(paste('z =', format(z, digits=3), 'p < 0.0001'))
          }
          else if (abs(z) > 3.291) {
            all_not_significant = FALSE
            print(paste('z =', format(z, digits=3), 'p < 0.001'))
          }
          else if (abs(z) > 2.576) {
            all_not_significant = FALSE
            print(paste('z =', format(z, digits=3), 'p < 0.01'))
          }
          else if (abs(z) > 1.96) {
            all_not_significant = FALSE
            print(paste('z =', format(z, digits=3), 'p < 0.05'))
          }
          else {
            print('not significant')
          }
          
          # if significantly different, not this increment
          if (p <= 0.05) {
            all_not_significant = FALSE
            break
          }
        }
        if (all_not_significant) {
          print(paste(country, clf, feat_set, i, auc_scores[[country]][[clf]][[feat_set]][[i]]))
          break
        }
      }
    }
  }
}
