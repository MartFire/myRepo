import pylab as pl
import matplotlib.pyplot as plt
import time
import numpy as np
from scipy.signal import butter, lfilter, iirfilter
from threading import Thread
import winsound

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable

from mne import create_info, find_events, pick_types
from mne.realtime import FieldTripClient
from mne.decoding import CSP

from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

def execute_bci():
    """"Start online BCI"""

    bads = []
    #Channels names and types
    ch_names = ['Fp1','Fp2','C3','C4','P7','P8','O1','O2']
    ch_types = ['eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg']
    fs = 250 # Sampling rate
    raw_info = create_info(ch_names, fs, ch_types) # Object used by MNE 
    global picks
    picks = pick_types(raw_info, meg=False, eeg=True, eog=False,
                            stim=False, include=[], exclude=bads)
    
    global all_images # Array to store images used during the experiment, except test image
    all_images = []                        
    im1 = pl.imread('online images/arrow red left.png')
    im2 = pl.imread('online images/arrow green left.png')
    all_images.append([im1, im2])
    
    im1 = pl.imread('online images/arrow red right.png')
    im2 = pl.imread('online images/arrow green right.png')
    all_images.append([im1,im2])
    
    global test_image # Image used during testing
    test_image = pl.imread('online images/test.png')
    global csp # CSP object
    global clf # Classifier
    global train_datas #Array to store training data
    train_datas = []
    global train_labels #Array to store training labels
    
    # Client used to get data from the EEG helmet
    with FieldTripClient(host='localhost',info=raw_info, port=1972, wait_max=10) as rt_client:
        for k in range(5):
            train_order = np.random.binomial(1, 0.5, 10) # List of labels for the next training sequence
            
            # Play sound to indicate the start of the training sequence
            winsound.PlaySound('online sound/starting training.wav', winsound.SND_FILENAME) 
            for i, label in enumerate(train_order):
                #Add label to training labels
                if (k, i)  == (0, 0):
                    train_labels = np.array([label])
                else:
                    train_labels = np.concatenate((train_labels, [label]))
                #Display training images and record data
                display_arrow(label, rt_client)
            
            # Play sound to indicate the end of the training sequence
            winsound.PlaySound('online sound/end of training.wav', winsound.SND_FILENAME)
            # Adjust CSP and classifier
            adjuster = Adjuster(train_datas, train_labels)
            adjuster.start()
            adjuster.join()
            csp = adjuster.csp
            clf = adjuster.clf
            
            # Play sound to indicate the start of the testing sequence
            winsound.PlaySound('online sound/starting testing.wav', winsound.SND_FILENAME)    
            for t in range(10):
                tester = Tester(csp, clf, rt_client, picks)
                #Display test image, record data, and indicate predict label
                tester.run()
            # Play sound to indicate the end of the testing sequence
            winsound.PlaySound('online sound/end of testing.wav', winsound.SND_FILENAME)
        #Display the features distribution post CSP
        display_distribution(train_datas, train_labels, csp)
        
class Adjuster(Thread):
    """Thread to ajust CSP and classfier with new training data"""
    
    def __init__(self, datas, labels):
        Thread.__init__(self)
        self.datas = datas
        self.labels = labels
        self.csp = None
        self.clf = None
    
    def run(self):
        self.csp = CSP(reg='ledoit_wolf')
        self.csp.fit(self.datas, self.labels)
        
        self.clf = svm.SVC()
        self.clf.fit(self.csp.transform(self.datas), self.labels)


    
def display_arrow(label, rt_client):
    """Display training images and get data from EEG helmet"""
    global picks, train_datas, train_labels, csp, clf, picks
    
    # Get arrow images for corresponding label (left arrows for label 0 and right arrows for label 1)
    images = all_images[label] 
    
    fig, ax = plt.subplots()
    #Display red arrow to indicate the movement to execute
    im = ax.imshow(images[0]) 
    fig.show()
    #Also, say the movement to execute
    if label == 0:
        winsound.PlaySound('online sound/left_sound.wav', winsound.SND_FILENAME) 
    else:
        winsound.PlaySound('online sound/right_sound.wav', winsound.SND_FILENAME)
    # Wait 1 s
    plt.pause(1)
    #Display a green arrow to indicate to the subject to start his movement
    im.set_data(images[1])
    fig.canvas.draw()
    #Also, Play sound 'go' in addition
    winsound.PlaySound('online sound/go.wav', winsound.SND_FILENAME)
    #Wait 3 s
    plt.pause(3) 
    
    #Get last 500 samples from EEG helmet
    new_data = rt_client.get_data_as_epoch(n_samples=500, picks=picks).get_data()
    #Bandpass filter this samples to keep only 7-35 Hz frequencies
    new_data = butter_bandpass_filter(new_data, 7, 35, 250, order=5)
    #Add data to training data
    if len(train_datas) != 0:
        train_datas = np.concatenate((train_datas, new_data), axis=0)
    else:
        train_datas = new_data
    #Play 'end' sound to indicate to the subject to stop executing his movement
    winsound.PlaySound('online sound/end.wav', winsound.SND_FILENAME)    
    plt.close()

class Tester(Thread):
    """Thread which display testing image, record data and indicate predicted label"""
    def __init__(self, csp, clf, client, picks):
        Thread.__init__(self)
        self.client = client
        self.picks = picks
        self.csp = csp
        self.clf = clf
        
    def run(self):
        fig, ax = plt.subplots()
        #Display test image which indicate to the subject to start executing a left or right movement
        im = ax.imshow(test_image)
        fig.show()
        #Also, say 'go'
        winsound.PlaySound('online sound/go.wav', winsound.SND_FILENAME)
        #Wat 3 s
        plt.pause(3)
        # Get 500 last samples frm EEG helmet
        test_data = self.client.get_data_as_epoch(n_samples=500, picks=self.picks).get_data()
        #Bandpass filter these samples
        test_data = butter_bandpass_filter(test_data, 7, 35, 250, order=5)
        #Apply CSP filter to thses samples
        test_data = self.csp.transform(test_data)
        #Predict label from these samples
        guess = self.clf.predict(test_data)
        
        # Play sound to indicate to the subject the predicted label
        if guess == 0.:
            winsound.PlaySound('online sound/left_sound.wav', winsound.SND_FILENAME)
        else:
            winsound.PlaySound('online sound/right_sound.wav', winsound.SND_FILENAME)
        plt.pause(1)
        plt.close()
        
        
def butter_bandpass(lowcut, highcut, fs, order=5):
    """Return bandpass filter coeffs"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """Apply bandpass filter to data"""
    b, a = butter_bandpass(lowcut, highcut, fs, order=5)
    out = lfilter(b, a, data, axis = 0)
    return out
           
def display_distribution(datas, labels, csp):
    """Plot features distribution post CSP"""
    datas = csp.transform(datas)
    fig = plt.figure()
    fig.set_figheight(24)
    fig.set_figwidth(24)
    combs = [[0,1,2],[1,2,3],[0,2,3],[0,1,3]]
    
    index0 = np.where(labels == 0)
    index1 = np.where(labels == 1)
    left = []
    right = []
    for i in index0[0]:
        left.append(datas[i])
    left = np.array(left)
    
    for i in index1[0]:
        right.append(datas[i])
    right = np.array(right)
    
    sorted_feat = [left, right]
    colors = 'br'
    markers = 'ov'
    for c in range(len(combs)):
        ax = fig.add_subplot(2,2,c+1, projection='3d')
        for k in range(2):
            f = sorted_feat[k]
            ax.scatter(f[:,combs[c][0]], f[:,combs[c][1]], f[:,combs[c][2]], c=colors[k], marker = markers[k])
    plt.show()
        