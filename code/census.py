import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import classification_report, \
    ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
plt.rcParams['figure.dpi'] = 300

try:
    os.chdir('240project')
except FileNotFoundError:
    pass

train = pd.read_csv('census-income.data', header=None, na_values=' ?')
test = pd.read_csv('census-income.test', header=None, na_values=' ?')
train.fillna('Unspecified', inplace=True)
test.fillna('Unspecified', inplace=True)
print(train.shape, test.shape)

# 42 columns
train.columns = test.columns = ['age', 'class_of_worker', 'industry_code', 'occupation_code', 'education',
'wage_per_hour', 'enrolled_in_edu_inst_last_wk', 'marital_status', 'major_industry_code',
'major_occupation_code', 'race', 'hispanic_origin', 'sex', 'member_of_a_labor_union',
'reason_for_unemployment', 'full_or_part_time_employment_stat', 'capital_gains',
'capital_losses', 'dividends_from_stocks', 'tax_filer_status', 
'region_of_previous_residence', 'state_of_previous_residence',
'detailed_household_and_family_stat', 'detailed_household_summary_in_household',
'instance_weight', 'migration_code_change_in_msa', 'migration_code_change_in_reg',
'migration_code_move_within_reg', 'live_in_this_house_one_year_ago',
'migration_prev_res_in_sunbelt', 'num_persons_worked_for_employer',
'family_members_under_18', 'country_of_birth_father',
'country_of_birth_mother', 'country_of_birth_self', 'citizenship',
'own_business_or_self_employed', 'fill_inc_questionnaire_for_veterans_admin', 
'veterans_benefits', 'weeks_worked_in_year', 'year', 'label']

# instance weight is ignorable for classification but needed for interpretation
# it is the number of people that each record represents due to sampling
instance_weight_train = train['instance_weight']
instance_weight_test = test['instance_weight']
train.drop(['instance_weight'], axis=1, inplace=True)
test.drop(['instance_weight'], axis=1, inplace=True)

# present in training but not in testing, need to drop
# {'detailed_household_and_family_stat_ Grandchild <18 ever marr not in subfamily'}
# idx = train.columns.get_loc('detailed_household_and_family_stat')
train.drop('detailed_household_and_family_stat', axis=1, inplace=True)
test.drop('detailed_household_and_family_stat', axis=1, inplace=True)

# remap label to 0 and 1
train['label'] = train['label'].map({' - 50000.': 0, ' 50000+.': 1})
test['label'] = test['label'].map({' - 50000.': 0, ' 50000+.': 1})
print(train['label'].value_counts())
# 0    187141
# 1     12382
y_train = train['label']
y_test = test['label']
train.drop(['label'], axis=1, inplace=True)
test.drop(['label'], axis=1, inplace=True)

# set up a copy of the data for BernoulliNB
# train_nb = train.copy()
# test_nb = test.copy()
# numerical vars:
# num_vars = ['age', 'wage_per_hour', 'capital_gains', 'capital_losses', 'dividends_from_stocks',
#             'num_persons_worked_for_employer', 'weeks_worked_in_year']
# for col in num_vars:
#     train_nb[col] = pd.cut(train_nb[col], bins=2, labels=['low', 'high'])
#     test_nb[col] = pd.cut(test_nb[col], bins=2, labels=['low', 'high'])

# see census-income.names for nominal variables
# in addition to cols with dtype object, these variables need to be converted
# ['industry_code', 'occupation_code', 'own_business_or_self_employed', 'veterans_benefits', 'year']
to_enc = train.columns[train.dtypes == 'object'].tolist() + \
    ['industry_code', 'occupation_code', 'own_business_or_self_employed', 'veterans_benefits', 'year']
train_enc = pd.get_dummies(train, columns=to_enc)
test_enc = pd.get_dummies(test, columns=to_enc)
col_names = train_enc.columns

# train_nb_enc = pd.get_dummies(train_nb, columns=train_nb.columns)
# test_nb_enc = pd.get_dummies(test_nb, columns=test_nb.columns)

print(train_enc.shape, test_enc.shape)  # 473 columns
# print(train_nb_enc.shape, test_nb_enc.shape)  # 479 columns

# scale data
train_enc = StandardScaler().fit_transform(train_enc)
test_enc = StandardScaler().fit_transform(test_enc)

estimators = {
    'NB': BernoulliNB(),
    'LR': LogisticRegression(random_state=0),
    'RF': RandomForestClassifier(
        n_estimators=100,
        max_features='sqrt',
        oob_score=True,
        n_jobs=-1,
        random_state=0),
}

for name, estimator in estimators.items():
    estimator.fit(train_enc, y_train)
    print(f"{name} train accuracy: {estimator.score(train_enc, y_train)}")
    print(f"{name} test accuracy: {estimator.score(test_enc, y_test)}")

# NB train accuracy: 0.7418392866987766
# NB test accuracy: 0.7419959503618613
# LR train accuracy: 0.9533337008765906
# LR test accuracy: 0.9533589944066879
# RF train accuracy: 0.9995288763701428
# RF test accuracy: 0.953589543112608

# RF feature importances
importances = pd.Series(estimators.get('RF').feature_importances_, index=col_names)
less = importances[importances > 0.005].sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(12, 9))
bp = sns.barplot(x=less.values, y=less.index, ax=ax)
bp.set_yticklabels(less.index, fontsize=6)
plt.show()

estimators_reduced_data = {
    'NB': BernoulliNB(),
    'LR': LogisticRegression(random_state=0),
    'RF': RandomForestClassifier(
        n_estimators=100,
        max_features='sqrt',
        oob_score=True,
        n_jobs=-1,
        random_state=0)
}

for name, estimator in estimators_reduced_data.items():
    estimator.fit(train_enc[:, less.index], y_train)
    print(f"{name} train accuracy: {estimator.score(train_enc[:, less.index], y_train)}")
    print(f"{name} accuracy: {estimator.score(test_enc[:, less.index], y_test)}")

# NB train accuracy: 0.8318790314901039
# NB accuracy: 0.8334636434714621
# LR train accuracy: 0.9515544573808533
# LR accuracy: 0.9515647240432229
# RF train accuracy: 0.9955894809119751
# RF accuracy: 0.9511036266313827


# metrics
def evaluate_models(estimator_dict, X_test, y_test, fname):
    print(X_test.shape, y_test.shape)
    predictions = [e.predict(X_test) for e in estimator_dict.values()]
    names = [n for n in estimator_dict.keys()]

    for i, name in enumerate(estimators.keys()):
        print(f"NAME: {name}")
        print(classification_report(y_true=y_test, y_pred=predictions[i]))

    figs, axes = plt.subplots(1, 3)
    for ax, p, name in zip(axes.ravel(), predictions, names):
        ConfusionMatrixDisplay.from_predictions(y_test, p, ax=ax, colorbar=False)
        ax.set_title(name)
    plt.tight_layout()
    plt.savefig(f'cm_{fname}', dpi=300)
    # plt.show()

    fig, ax = plt.subplots(1, 1)
    for e in estimator_dict.values():
        PrecisionRecallDisplay.from_estimator(e, X_test, y_test, ax=ax)
    plt.savefig(f'pr_{fname}', dpi=300)
    # plt.show()

    fig, ax = plt.subplots(1, 1)
    for e in estimator_dict.values():
        RocCurveDisplay.from_estimator(e, X_test, y_test, ax=ax)
    plt.savefig(f'roc_{fname}', dpi=300)
    # plt.show()


evaluate_models(estimators, test_enc, y_test, fname='full')
evaluate_models(estimators_reduced_data, test_enc[:, less.index], y_test, fname='red')
