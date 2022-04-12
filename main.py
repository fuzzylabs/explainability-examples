import sklearn
import shap
import matplotlib.pyplot as plt

# print the JS visualization code to the notebook
shap.initjs()

# train a SVM classifier
X_train,X_test,Y_train,Y_test = sklearn.model_selection.train_test_split(*shap.datasets.iris(), test_size=0.2, random_state=0)
svm = sklearn.svm.SVC(kernel='rbf', probability=True)
svm.fit(X_train, Y_train)

# use Kernel SHAP to explain test set predictions
explainer = shap.KernelExplainer(svm.predict_proba, X_train, link="logit")
shap_values = explainer.shap_values(X_test, nsamples=100)

# print(shap_values)

# plot the SHAP values for the Setosa output of all instances
# This commented code seems to just save a white image
# fig = shap.force_plot(explainer.expected_value[0], shap_values[0], X_test, link="logit")
# plt.savefig('grafic.png')

fig = shap.summary_plot(shap_values, X_train, show=False)
plt.savefig('shap.png', bbox_inches='tight')

