import scipy.io as io
import preprocessor as pre
from naive_bayes import naiveBayes
from logit import logisitcRegression

Numpyfile = io.loadmat('mnist_data.mat')
trX = Numpyfile['trX']
trY = Numpyfile['trY']
tsX = Numpyfile['tsX']
tsY = Numpyfile['tsY']

#extracting features (average value of pixels, standard deviation of pixels)
trxFeatures = pre.extractFeatures(trX)
tsxFeatures = pre.extractFeatures(tsX)

mean_7,mean_8,sd_7,sd_8,accuracy_bayesian_7,accuracy_bayesian_8,accuracy_bayesian_total=naiveBayes(trxFeatures,trY[0],tsxFeatures,tsY[0])
w,accuracy_logit_7,accuracy_logit_8,accuracy_logit_total = logisitcRegression(trxFeatures,trY[0],tsxFeatures,tsY[0])

print("Accuracy for Naive Bayes:")
print("For Digit 7: ",accuracy_bayesian_7*100)
print("For Digit 8: ",accuracy_bayesian_8*100)
print("Overall: ",accuracy_bayesian_total*100)
print("\n")
print("Accuracy for Logistic Regression: ")
print("For Digit 7: ",accuracy_logit_7*100)
print("For Digit 8: ",accuracy_logit_8*100)
print("Overall: ",accuracy_logit_total*100)