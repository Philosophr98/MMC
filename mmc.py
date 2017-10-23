import sys
sys.executable
import glob
import os
import librosa.display as disp
import tensorflow as tf
import librosa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
%matplotlib inline



from sklearn import svm

#######################################  feature extraction      ################################################
def extract_feature(clip_name):
    A, sample_rate = librosa.load(clip_name)
    stft = np.abs(librosa.stft(A))
    mfccs = np.mean(librosa.feature.mfcc(y=A, sr=sample_rate, n_mfcc=40).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(A, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(A),
    sr=sample_rate).T,axis=0)
    return mfccs,chroma,mel,contrast,tonnetz



#######################              parses all the audio files in specified folder and calls the feature extraction function       #####################################################
def parse_through_files(parent_fold,sub_folds,file_ext="*.wav"):
    features, labels = np.empty((0,193)), np.empty(0)
    for label, sub_fold in enumerate(sub_folds):
        for fn in glob.glob(os.path.join(parent_fold, sub_fold, file_ext)):
            print(fn)
            try:
              mfccs, chroma, mel, contrast,tonnetz = extract_feature(fn)
            except Exception as e:
              print ("Error encountered while parsing file: "), fn
              continue
            ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
            features = np.vstack([features,ext_features])
            labels = np.append(labels, fn.split('/')[2].split('-')[1])
    return np.array(features), np.array(labels, dtype = np.int)


parent_fold ='sounds'
train_sub_folds = ["cheering","music","speech"]
test_sub_folds = ["fold3"]
train_features, train_labels = parse_through_files(parent_fold,train_sub_folds)
test_features, test_labels = parse_through_files(parent_fold,test_sub_folds)



###########################################         SVM           ######################################################

clf=svm.SVC(decision_function_shape='ovo')
clf.fit(train_features,train_labels)
print(train_labels)


clf.predict(test_features)
print(test_labels)




##########################################         K-nearest neighbours    ##################################################
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(train_features,train_labels)


knn.predict(test_features)
print(test_labels)
