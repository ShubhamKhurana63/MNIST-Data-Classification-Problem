import pandas as pd
import tensorflow as tf
import pickle
import gzip
import numpy as np
from PIL import Image

import os
from sklearn.svm import SVC
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping, TensorBoard
from sklearn.metrics import confusion_matrix  


#from keras.optimizers import optimizers
from sklearn.ensemble import RandomForestClassifier
from scipy import stats

#loading of the MNIST data
filename = 'mnist.pkl.gz'
f = gzip.open(filename, 'rb')
training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
print('Original Train data',training_data[0][:,2:5], np.max(training_data[0]))
f.close()
trainingInputs=training_data[0]#50000*784
trainingTargets=training_data[1]

#loading of the USPS data
def loadUSPSData():
    USPSMat  = []
    USPSTar  = []
    curPath  = 'USPSdata/Numerals'
    savedImg = []

    for j in range(0,10):
        curFolderPath = curPath + '/' + str(j)
        imgs =  os.listdir(curFolderPath)
        for img in imgs:
            curImg = curFolderPath + '/' + img
            if curImg[-3:] == 'png':
                img = Image.open(curImg,'r')
                img = img.resize((28, 28))
                savedImg = img
                imgdata = (255-np.array(img.getdata()))/255
                USPSMat.append(imgdata)
                USPSTar.append(j)
    return np.asarray(USPSMat),np.asarray(USPSTar)

#processing the given target into 10 bit encoded vector 
def processLabels(target):
    one_hot_targets = np.eye(10)[target]
    return np.asarray(one_hot_targets)



#computing the softmax for given weights and data
def calculateSoftMax(data,weights):
    dproduct=np.dot(data, weights)
    y=findSoftMax(dproduct)#calling the custom implementation of softmax method
    return y

#Applying logistic regression, using gradient descent for updation of weights
def logisticRegression(trainingData,targets):
    print(trainingData.shape)
    W_LIST= np.zeros([trainingData.shape[1], 10],dtype=np.float32)
    W=np.asarray(W_LIST)
    print(W.shape)
    for i in range(0,2500):#1500 iterations
        for k in range(0,trainingData.shape[0],100):
            h=calculateSoftMax(trainingData[k:k+100,:], W)
            X=trainingData[k:k+100,:].T        
            encodedHotVector=processLabels(targets[k:k+100])
            differenceList=[h[j]-encodedHotVector[j] for j in range(0,h.shape[0])]
            difference=np.asarray(differenceList)
            delW=np.dot(X,difference)/targets.shape[0]
            W-=.05*delW#.05 as the learning rate
    return W    

#evaluating the accuracy for the logistic regression on the predicted weights
def evaludateLogisticAccuracy(validationData,predictedWeights,targets,dataType):
    output=calculateSoftMax(validationData,predictedWeights); 
    ctr_right=0
    ctr_wrong=0
    outputVector=np.argmax(output,axis=1)
    for z in range(0,outputVector.shape[0]):
        if outputVector[z]==targets[z]:
            ctr_right=ctr_right+1
        else:
            ctr_wrong=ctr_wrong+1
    createConfusionMatrix(targets,outputVector,dataType)
    print('--------ctr right for logistic regression---------',ctr_right)
    print('-------ctr  wrong for logistic regression------',ctr_wrong)
    print('------accuracy on logistic regression '+str(dataType)+'------',(ctr_right/(ctr_right+ctr_wrong)))
    return outputVector
#evaluating the accuracy for the neural network implementation, using the keras model
def evaluateNeuralNetwork(testData,testLabels,model,type):
    #processedTestLabels=processLabels(testLabels)
    rightCounter=0
    wrongCounter=0
    neuralPredictedLabels=[]
    for i,j in zip(testData,testLabels):
        predictedLabel=model.predict(np.asarray(i).reshape(-1,784))
        #print(predictedLabel)
        neuralPredictedLabels.append(predictedLabel.argmax())
        if predictedLabel.argmax()==j:
            rightCounter=rightCounter+1
        else:
            wrongCounter=wrongCounter+1
    createConfusionMatrix(testLabels,np.asarray(neuralPredictedLabels),type)
    accuracy=rightCounter/(rightCounter+wrongCounter)
    print('----------accuracy for '+str(type)+' of neural network----------- ',accuracy)
    return neuralPredictedLabels

#evaluating the accuracy for the random forest
def evaluateRandomForest(testData,testLabels,dataType,clf):
    predictedData=clf.predict_proba(testData)
    rightCounter=0
    wrongCounter=0
    randomForestTargetList=[]
    for i,j in zip(predictedData,testLabels):
        #predictedLabel=model.predict(np.asarray(i).reshape(-1,784))    
        randomForestTargetList.append(i.argmax())
        if i.argmax()==j:
            rightCounter=rightCounter+1
        else:
            wrongCounter=wrongCounter+1
    createConfusionMatrix(testLabels,np.asarray(randomForestTargetList),dataType)
    print('--------ctr right for random forest---------',rightCounter)
    print('-------ctr  wrong for random forest------',wrongCounter)
    print('------accuracy on for random forest '+str(dataType)+'------',(rightCounter/(rightCounter+wrongCounter)))
    return randomForestTargetList

#custom implementation of the softmax function
def findSoftMax(z):
    #max=np.max(z,axis=1).reshape((-1,1))
    #print('------max------',max.shape)
    #-max
    exponential=np.exp(z)#50000 * 10
    #print('-----exp shape-----------',exponential.shape)
    norms=np.sum(exponential,axis=1).reshape((-1,1))    
    #print('---------norms--------',norms.shape)
    return exponential/norms

#method for creation of the confusion matrix
def createConfusionMatrix(y_actual,y_predicted,type):
    print('confusion matrix type: '+str(type))
    print(confusion_matrix(y_actual,y_predicted))

input_size = 784 #input has got 784 features(28 * 28)
drop_out = 0.2
first_dense_layer_nodes  = 512
second_dense_layer_nodes = 10 #out put has to be in the form of   1 * 10 vector 
#configuring the sequential model for DNN using keras(4-layered structure, including the input layer)
def get_model():
    model=Sequential()
    model.add(Dense(first_dense_layer_nodes, input_dim=input_size))
    model.add(Activation('relu'))#activation on first dense layer or the hidden layer
    model.add(Dropout(drop_out));   
    #added another layer to get better accuracy
    model.add(Dense(512))
    model.add(Activation("relu"))#activation on second dense layer or the hidden layer
    model.add(Dropout(drop_out))
    model.add(Dense(second_dense_layer_nodes))
    model.add(Activation('softmax'))#activation on last dense layer or the output layer
    model.summary()
    #using adam optimizer with default learning rate value
    model.compile(optimizer="adam",
                  loss='categorical_crossentropy',#using categorical_crossentropy, since we are dealing with the multiclass regression problem
                  metrics=['accuracy'])
    return model

#training the model, using keras
def fitModel(trainingInputs,trainingTargets,model):
    num_epochs = 150 
    """
    1 epoch=[(Total Data)/batch size] number of iterations
    """
    model_batch_size = 256
    tb_batch_size = 32
    early_patience = 50 #if no decrease in loss is observed, num of epochs with no improvement, training will be stopped
    tensorboard_cb   = TensorBoard(log_dir='logs', batch_size= tb_batch_size, write_graph= True)
    #in case the loss is not decreasing for more than 100 iterations then callback will stop the training.
    earlystopping_cb = EarlyStopping(monitor='val_loss', verbose=1, patience=early_patience, mode='min')
    processedLabels=processLabels(trainingTargets)
    history = model.fit(trainingInputs
                    , processedLabels
                    , epochs=num_epochs
                    , validation_split=0.0
                    , batch_size=model_batch_size
                    , callbacks = [tensorboard_cb,earlystopping_cb]
                   )
    return history



# for setting the labels in the targetBox(being used for ensemble)
def majorityPollingFiller(targets,targetBox,isHotEncoded):   
    if isHotEncoded:   
        outputVector=np.argmax(targets,axis=1)
        targetBox.append(outputVector)
    else:
        targetBox.append(targets)
    

#computing the new targets from  matrix of combined result of classifiers, using hard voting method
def calculateEnsembleAccuracy(targetBox,type,actualTargets):
    predictedTargetMatrix=np.asarray(targetBox)
    predictedMatrix=predictedTargetMatrix.T
    ensembleTargets=[]
    ctr_right=0
    ctr_wrong=0
    for i in predictedMatrix:
        ensembleTargets.append(np.bincount(i).argmax()) #getting the mode from each row 

    for i,j in zip(ensembleTargets,actualTargets):
            if i==j:
                ctr_right=ctr_right+1
            else:
                ctr_wrong=ctr_wrong+1
    createConfusionMatrix(actualTargets,ensembleTargets,type)
    print('----------------------- ensemble for '+str(type)+'-----------------------------')    
    print('--------ctr right for ensemble---------',ctr_right)
    print('-------ctr  wrong for ensemble------',ctr_wrong)
    print('------accuracy on ensemble------',(ctr_right/(ctr_right+ctr_wrong)))


trainingInputs=trainingInputs[0:50000,:]
trainingTargets=trainingTargets[0:50000]
targetBoxValidation=[]
targetBoxTesting=[]
print('--------trainingInputs-------',trainingInputs.shape)
print('--------trainingtargets-------',trainingTargets.shape)

#========================logistic Regression=============================================================#
predictedWeights=logisticRegression(trainingInputs,trainingTargets) #calling method for training using logistic regression
print('--------predicted weights-------',predictedWeights)
uspsSamples,uspsTargets=loadUSPSData()
validation_Targets=validation_data[1]
#evaluating the logistic regression accuracy for validation data
logisticValidationPrediction=evaludateLogisticAccuracy(validation_data[0],predictedWeights,validation_data[1],'validation')
#evaluating the logistic regression accuracy for testing data
logisticTestPrediction=evaludateLogisticAccuracy(test_data[0],predictedWeights,test_data[1],'testing')
#evaluating the logistic regression accuracy for usps data
evaludateLogisticAccuracy(uspsSamples,predictedWeights,uspsTargets,'usps')

#pushing the predicted labels for validation data into the ensemble matrix
majorityPollingFiller(logisticValidationPrediction,targetBoxValidation,False)
#pushing the predicted labels for testing data into the ensemble matrix
majorityPollingFiller(logisticTestPrediction,targetBoxTesting,False)

#==================================SVM=====================================================================#



#using the radial basis fucntion with gamma value as 1(highest value)
# svm_nonlinear_model=SVC(kernel="rbf",gamma=1)
# #training model for SVM
# svm_nonlinear_model.fit(trainingInputs,trainingTargets)
# #predciting label for validation data
# validationDataTargetsForSVM=svm_nonlinear_model.predict(validation_data[0])
# #predicting the label for testing data
# testingDataTargetsForSVM=svm_nonlinear_model.predict(test_data[0])

# #getting the accuracy for the SVM 
# accuracy_validation=svm_nonlinear_model.score(validation_data[0],validation_data[1])
# accuracy_test=svm_nonlinear_model.score(test_data[0],test_data[1])
# accuracy_SVM_USPS=svm_nonlinear_model.score(uspsSamples,uspsTargets)
# print('-----accuracy for SVM rbf for validation---------',accuracy_validation)
# print('-----accuracy for SVM rbf for test---------',accuracy_test)
# print('-----accuracy for SVM rbf for USPS---------',accuracy_SVM_USPS)
# #print(validationDataTargetsForSVM)
# majorityPollingFiller(validationDataTargetsForSVM,targetBoxValidation,False)
# majorityPollingFiller(testingDataTargetsForSVM,targetBoxTesting,False)
# createConfusionMatrix(test_data[1],testingDataTargetsForSVM,'svm')

#using radial basis function as kernel with default configuration
svm_nonlinear_model=SVC(kernel="rbf",gamma='auto')
svm_nonlinear_model.fit(trainingInputs,trainingTargets)
validationDataTargetsForSVM=svm_nonlinear_model.predict(validation_data[0])
testingDataTargetsForSVM=svm_nonlinear_model.predict(test_data[0])
uspsDataTargetsForSVM=svm_nonlinear_model.predict(uspsSamples)

accuracy_validation=svm_nonlinear_model.score(validation_data[0],validation_data[1])
accuracy_test=svm_nonlinear_model.score(test_data[0],test_data[1])
accuracy_SVM_USPS=svm_nonlinear_model.score(uspsSamples,uspsTargets)
print('-----accuracy for rbf for validation---------',accuracy_validation)
print('-----accuracy for rbf for test---------',accuracy_test)
print('-----accuracy for SVM rbf for USPS---------',accuracy_SVM_USPS)
majorityPollingFiller(validationDataTargetsForSVM,targetBoxValidation,False)
majorityPollingFiller(testingDataTargetsForSVM,targetBoxTesting,False)
createConfusionMatrix(test_data[1],testingDataTargetsForSVM,'svm')
createConfusionMatrix(uspsTargets,uspsDataTargetsForSVM,'svm')

# #using linear basis function as kernel, with default configuration
# svm_nonlinear_model=SVC(kernel="linear")
# svm_nonlinear_model.fit(trainingInputs,trainingTargets)
# validationDataTargetsForSVM=svm_nonlinear_model.predict(validation_data[0])
# testingDataTargetsForSVM=svm_nonlinear_model.predict(test_data[0])
# accuracy_validation=svm_nonlinear_model.score(validation_data[0],validation_data[1])
# accuracy_test=svm_nonlinear_model.score(test_data[0],test_data[1])
# accuracy_SVM_USPS=svm_nonlinear_model.score(uspsSamples,uspsTargets)
# print('-----accuracy for rbf for validation---------',accuracy_validation)
# print('-----accuracy for rbf for test---------',accuracy_test)
# print('-----accuracy for SVM rbf for USPS---------',accuracy_SVM_USPS)
# print(validationDataTargetsForSVM)
# majorityPollingFiller(validationDataTargetsForSVM,targetBoxValidation,False)
# majorityPollingFiller(testingDataTargetsForSVM,targetBoxTesting,False)
# createConfusionMatrix(test_data[1],testingDataTargetsForSVM,'svm')
#==============================================Neural Network=================================================#
nnModel=get_model()
history=fitModel(trainingInputs,trainingTargets,nnModel)
#evaluating the valiation data
neuralValidationPrediction=evaluateNeuralNetwork(validation_data[0],validation_data[1],nnModel,'validation')
#evaluating the testing data
neuralTestPrediction=evaluateNeuralNetwork(test_data[0],test_data[1],nnModel,'test')
#evaluating the USPS data on a trained model
evaluateNeuralNetwork(uspsSamples,uspsTargets,nnModel,'usps')
majorityPollingFiller(neuralValidationPrediction,targetBoxValidation,False)
majorityPollingFiller(neuralTestPrediction,targetBoxTesting,False)
#====================================================Random Forest===========================================#
#using sklearn random classifier, with estimators value =1000, for better decison process
clf=RandomForestClassifier(n_estimators=1000,n_jobs=2,random_state=0)
clf.fit(trainingInputs,trainingTargets)        
forestValidationPrediction=evaluateRandomForest(validation_data[0],validation_data[1],'validation',clf)
forestTestPrediction=evaluateRandomForest(test_data[0],test_data[1],'test',clf)
evaluateRandomForest(uspsSamples,uspsTargets,'usps',clf) 
majorityPollingFiller(forestValidationPrediction,targetBoxValidation,False)
majorityPollingFiller(forestTestPrediction,targetBoxTesting,False)

#method call for computation of the accuracy of combined classifier results, for validation data
calculateEnsembleAccuracy(targetBoxValidation,'validation',validation_data[1])
#method call for computation of the accuracy of combined classifier results, for testing data
calculateEnsembleAccuracy(targetBoxTesting,'testing',test_data[1])