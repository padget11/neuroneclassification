import scipy.io as spio
import numpy as np
import scipy.special as scisp
import scipy.signal as scisi
import statistics
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

# peakdetection function takes the data and threshold as input and returns the
# peaks from the data, the threshold is used to ignore any peaks that may be
# caused by noise
def peakdetection(data, threshold):

    # bandpass filter the data with highpass 0.01 Wn (125 Hz) and lowpass 0.08 (1000 Hz)
    b, a = scisi.butter(2, [0.01, 0.08], btype='band')
    filtered_data = scisi.filtfilt(b, a, data)

    # set all negative values to 0 so when signal is squared the negative peaks aren't included
    d_z = np.array([x if x > 0 else 0 for x in filtered_data])

    # square data to increase signal and decrease noise
    d_sq = d_z * d_z

    # find peaks by finding change in gradient
    dx = np.diff(d_sq) # gradient
    turning = []
    sign_dx = np.sign(dx) # shows positive or negative gradient
    for i in range(len(sign_dx) - 1):
        # peaks occurs when gradient changes from positive to negative
        if sign_dx[i + 1] < 0 and sign_dx[i] > 0:
            # only keep peaks above a threshold
            if d_sq[i] > threshold:
                turning.append(i)

    # move turning point locations to start of peak to match training data,
    # finds a point in the gradient below a threshold of 0.2
    peaks = []
    for y in turning:
        min_loc = y # set location as peak to begin
        x = 0 # reset x
        z = y
        # need to shift data away from the zero crossing of the gradient so move left until
        # gradient of gradient turns positive
        while np.sign(dx[z] - dx[z - 1]) < 0:
            z = z - 1
            x = z
        x = 0 # reset x
        # keep shifting left until threshold met
        while dx[z - x] > 0.2 and z - x > 0:
            min_loc = z - x
            x = x + 1
        peaks.append(min_loc)

    return peaks

# splitpeaks function takes data, peaks and window as input and splits the data
# into the windows, returns the split data
def splitpeaks(data, peaklocation, window_size):
    
    window_size = int(window_size)
    data_split = []
    for x in range(len(peaklocation)):
        window = data[peaklocation[x]:peaklocation[x] + window_size]
        data_split.append(window)

    return data_split

# plotclasses function take data, peak location, class and window and plots
# each peak according to class
def plotclasses(peaks, data, type, window_size):

    for x in range(len(peaks)):
        if type[x] == 1:
            plt.plot(data[peaks[x]:peaks[x] + window_size], linewidth=0.5, color='b')
        if type[x] == 2:
            plt.plot(data[peaks[x]:peaks[x] + window_size], linewidth=0.5, color='g')
        if type[x] == 3:
            plt.plot(data[peaks[x]:peaks[x] + window_size], linewidth=0.5, color='r')
        if type[x] == 4:
            plt.plot(data[peaks[x]:peaks[x] + window_size], linewidth=0.5, color='y')

# performpca function takes the training and testing data and numper of pricipal components
# as inputs and returns the PCA to be used on the submission data, the PCA will reduce the
# number of inputs needed for the neural network
def performpca(train_d, test_d, components):

    pca = PCA(n_components = components)
    # fit to the training data
    pca.fit(train_d)

    # determine amount of variance explained by components
    print("Total Variance Explained: ", np.sum(pca.explained_variance_ratio_))

    # plot the explained variance
    plt.figure(0)
    plt.plot(pca.explained_variance_ratio_)
    plt.title('Variance Explained by Extracted Componenents')
    plt.ylabel('Variance')
    plt.xlabel('Principal Components')
        
    # extract the principle components from the training data
    train = pca.fit_transform(train_d)
    # transform the test data using the same components
    test = pca.transform(test_d)

    # normalise the data sets
    min_max_scaler = MinMaxScaler()
    train_norm = min_max_scaler.fit_transform(train)
    test_norm = min_max_scaler.fit_transform(test)

    return train_norm, test_norm, pca, min_max_scaler

# neural network class definition used to classify the data
class NeuralNetwork:
    # Init the network, this gets run whenever we make a new instance of this class
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set the number of nodes in each input, hidden and output layer
        self.i_nodes = input_nodes
        self.h_nodes = hidden_nodes
        self.o_nodes = output_nodes

        # Weight matrices, wih (input -> hidden) and who (hidden -> output)
        self.wih = np.random.normal(0.0, pow(self.h_nodes, -0.5), (self.h_nodes, self.i_nodes))
        self.who = np.random.normal(0.0, pow(self.o_nodes, -0.5), (self.o_nodes, self.h_nodes))

        # Set the learning rate
        self.lr = learning_rate

        # Set the activation function, the logistic sigmoid
        self.activation_function = lambda x: scisp.expit(x)

    # Train the network using back-propagation of errors
    def train(self, inputs_list, targets_list):
        # Convert inputs into 2D arrays
        inputs_array = np.array(inputs_list, ndmin=2).T
        targets_array = np.array(targets_list, ndmin=2).T

        # Calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs_array)

        # Calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # Calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)

        # Calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        # Current error is (target - actual)
        output_errors = targets_array - final_outputs

        # Hidden layer errors are the output errors, split by the weights, recombined at hidden nodes
        hidden_errors = np.dot(self.who.T, output_errors)

        # Update the weights for the links between the hidden and output layers
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)),
        np.transpose(hidden_outputs))

        # Update the weights for the links between the input and hidden layers
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
        np.transpose(inputs_array))

    # Query the network
    def query(self, inputs_list):
        # Convert the inputs list into a 2D array
        inputs_array = np.array(inputs_list, ndmin=2).T

        # Calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs_array)

        # Calculate output from the hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # Calculate signals into final layer
        final_inputs = np.dot(self.who, hidden_outputs)

        # Calculate outputs from the final layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


#################################### MAIN CODE ##########################################
# this script loads data of neurones firing to classify them into 4 different types, the
# training data is used with known peaks and classes to train a neural network which then
# classifies data with unknown peaks and classes, the data needs to be filtered before it
# is classified to remove noise, the final output is saved as a .mat file

# load training data 
mat_train = spio.loadmat('training.mat', squeeze_me=True)
d = mat_train['d']
Index = mat_train['Index']
Class = mat_train['Class']

# since the peak Index and Class aren't ordered to the data, sort
# in ascending
idx = np.argsort(Index)
Index = np.array(Index)[idx]
Class = np.array(Class)[idx]

# load submission data
mat_sub = spio.loadmat('submission.mat', squeeze_me=True)
d_sub = mat_sub['d']

# set up neural network with 8 input nodes (reduced from 40 with PCA),
# 20 hidden nodes, 4 output nodes and learning rate of 0.08
window = 40
components = 8
n = NeuralNetwork(components, 20, 4, 0.08)

# find peaks for training data with threshold of 1
peaks = peakdetection(d, 1)

# variables for bandpass filter
freq_hz = 25000 # sampling frequency
low = 20 # low cut-off frequency
high = 5000  # high cut-off frequency
nyquist_freq = freq_hz/2 # nyquist frequency

# apply a 4th order bandpass filter to data
b, a = scisi.butter(4, [low/nyquist_freq, high/nyquist_freq], btype='band')
d_filt = scisi.filtfilt(b, a, d)

# split the data into peaks of size window
all_data = splitpeaks(d_filt, peaks, window) 

# match class with peak
class_train = []
for record in peaks:
    idx = (np.abs(Index - record)).argmin() # find closest peak in Index and match with Class
    class_train.append(Class[idx])

# split the data into training and testing
train_data = all_data[0:3000]
train_label = class_train[0:3000]
test_data = all_data[3001:-1]
test_label = class_train[3001:-1]

# perform PCA
train_ext, test_ext, pca, min_max_scaler = performpca(train_data, test_data, components)

# Calculate percentage of peaks found
print("Peaks found = ", (len(peaks) * 100 / len(Index)), ' % ')

# train neural network on train data
for x in range(3): # train 3 times
    for x, record in enumerate(train_ext):
        targets = np.zeros(4) + 0.01 # set all targets to 0
        targets[int(train_label[x]) - 1] = 0.99 # set  correct class to 0.99
        n.train(record, targets)

# test neural network with test data
scorecard = []
for x, record in enumerate(test_ext):
    correct_label = test_label[x] # get correct class
    outputs = n.query(record)
    label = np.argmax(outputs) + 1 # find most probable class, add one since indexing from 0
    # add 0 or 1 to scorecard depending of classified correctly
    if (label == correct_label):
        scorecard.append(1)
    else:
        scorecard.append(0)
    pass

# filter the submission data with a Savitzt-Golay filter, order 4 and window of 21, 
# since submission data is more noisy than training data, it requires more filtering
d_sub_savgol = scisi.savgol_filter(d_sub, 21, 4)

# find peaks on submission data with threshold of 1
peaks_sub = peakdetection(d_sub_savgol, 1)

# filter with the same coefficients as in training data
d_sub_filt = scisi.filtfilt(b, a, d_sub_savgol)

# split data into peaks of size window
all_data_sub = splitpeaks(d_sub_filt, peaks_sub, window)

# PCA tranform the submission data
all_data_sub_pca = pca.transform(all_data_sub)
all_data_sub_pca_norm = min_max_scaler.fit_transform(all_data_sub_pca)

# classify submission data
class_sub = []
for record in all_data_sub_pca_norm:
    outputs = n.query(record)
    label = np.argmax(outputs) + 1
    class_sub.append(label)
    pass

# print total number found for each class in submission data
print("Number of class 1 = ", class_sub.count(1))
print("Number of class 2 = ", class_sub.count(2))
print("Number of class 3 = ", class_sub.count(3))
print("Number of class 4 = ", class_sub.count(4))

# calculate the performance score, the fraction of correct answers
scorecard_array = np.asarray(scorecard)
print("Performance = ", (scorecard_array.sum() / scorecard_array.size)*100, '%')

# save submission data to mat file
spio.savemat('11230.mat', {'Class': class_sub, 'Index': peaks_sub})

# plot submission data classes
plt.figure(1)
plotclasses(peaks_sub, d_sub_filt, class_sub, window)

plt.show()