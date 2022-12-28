import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
#-------------------------------------------------------------------------------------
# Data loading and splitting
df = pd.read_csv('MCSDatasetNEXTCONLab.csv')
x = df.iloc[:, 0:12]
y = df.iloc[:, 12]
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)
 
# Models fitting and prediction
def models(model, x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = model.fit(x_train, y_train)
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    accuracy_train = accuracy_score(y_train, y_train_pred)
    accuracy_test = accuracy_score(y_test, y_test_pred)
    report_test = classification_report(y_test, y_test_pred)
    return (model, y_train_pred, y_test_pred, accuracy_train, accuracy_test, report_test)

#---------------------------------------------MAIN------------------------------------

RF, AB, NB = RandomForestClassifier(), AdaBoostClassifier(), GaussianNB()

# First model (Random Forest)
model_RF, y_train_pred_RF, y_test_pred_RF, accuracy_train_RF, accuracy_test_RF, report_RF = models(RF, x, y)
# Second model (Adaboost)
model_AB, y_train_pred_AB, y_test_pred_AB, accuracy_train_AB, accuracy_test_AB, report_AB = models(AB, x, y)
# Third model (Naive Bayes)
model_NB, y_train_pred_NB, y_test_pred_NB, accuracy_train_NB, accuracy_test_NB, report_NB = models(NB, x, y)

# Frist ensemble framework : majority voting-based aggregator
voting = VotingClassifier(estimators=[('RF', model_RF),('AB', model_AB), ('NB', model_NB)], voting='hard')
model_voting, y_train_pred_voting, y_test_pred_voting, accuracy_train_voting, accuracy_test_voting, repot_voting = models(voting,x,y)

# Second ensemble framework : weighted sum aggregation
t = accuracy_train_RF + accuracy_train_AB + accuracy_train_NB
w_RF, w_AB, w_NB = accuracy_train_RF/t, accuracy_train_AB/t, accuracy_train_NB/t
aggregated_output = (w_RF * y_train_pred_RF) + (w_AB * y_train_pred_AB) + (w_NB * y_train_pred_NB)
y_pred_weighted_sum  = []
for i in aggregated_output: 
    if i > 0.5: y_pred_weighted_sum.append(1)
    else: y_pred_weighted_sum.append(0)
accuracy_weighted_sum = accuracy_score(y_train, y_pred_weighted_sum)
report_weighted_sum = classification_report(y_train, y_pred_weighted_sum)

# Bar plots
df = pd.DataFrame({'Model':['Random Forest', 'AdaBoost', 'Naive Bayes', 'Voting', 'Weighted Sum'],
'Accuracy on Training Data':[accuracy_train_RF, accuracy_train_AB,
                             accuracy_train_NB, accuracy_train_voting, accuracy_weighted_sum],
'Accuracy on Testing Data':[accuracy_test_RF, accuracy_test_AB,
                            accuracy_test_NB, accuracy_test_voting, accuracy_weighted_sum] ,
})               

# Bar plot for training data
bar_plot_train = sns.barplot(data = df, x = 'Model', y = 'Accuracy on Training Data')
bar_plot_train.set_ylim(0.8, 1)

# Bar plotfor testing data
bar_plot_test = sns.barplot(data = df, x = 'Model', y = 'Accuracy on Testing Data')
bar_plot_test.set_ylim(0.8, 1)
