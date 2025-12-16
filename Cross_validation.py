#Evaluation avec les autres indicateurs

from sklearn.metrics import classification_report
print(classification_report(y_test, y_predict))
report_dict = classification_report(
    y_test,
    y_predict,
    output_dict=True
)

df_report = pds.DataFrame(report_dict).transpose()

df_report = df_report.round(6)

df_report

#Cross-validation

from sklearn.model_selection import StratifiedKFold #cross-validation splitter
from sklearn.model_selection import cross_validate #cross-validation evaluation of metrics
scoring = ['accuracy', 'precision_macro', 'precision_weighted', 'recall_macro', 'recall_weighted', 'f1_macro', 'f1_weighted']
cv = StratifiedKFold(n_splits=5, random_state=10, shuffle=True)
scores = cross_validate(knnc, X_normalized, Y , scoring=scoring,
                        cv=cv, return_train_score=False)

scores.keys()

print('Global accuracy over all folds: %0.6f (+/- %0.6f)'
      % (scores['test_accuracy'].mean(), scores['test_accuracy'].std() * 2))

print('For each metric, list the score values on each fold:')
for metric in sorted(scores.keys()):
    print(
        str(['{:.6f}'.format(value) for value in scores[metric]])
        + ' ' + metric
    )

df_scores = pds.DataFrame({
    'Accuracy': scores['test_accuracy'],
    'Precision (weighted)': scores['test_precision_weighted'],
    'Recall (weighted)': scores['test_recall_weighted'],
    'F1 (weighted)': scores['test_f1_weighted']
})

# Nommer les folds
df_scores.index = [f'Fold {i+1}' for i in range(len(df_scores))]

# Arrondir pour affichage / Excel
df_scores = df_scores.round(6)

df_scores

#Copier-coller directement dans canva
