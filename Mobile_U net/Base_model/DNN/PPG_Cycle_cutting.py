from scipy.signal import butter, filtfilt
from scipy.signal import argrelextrema

# For Max-Min Normalization 
def min_max_normalize(signal):
    
    min_val = np.min(signal)
    max_val = np.max(signal)
    
    # Perform min-max normalization
    signal_normalized = (signal - min_val) / (max_val - min_val)
    return signal_normalized


#### Butter Bandpass
def butter_bandpass(lowcut, highcut, fs, signal, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band', analog=False)
    # Apply the low pass filter to the PPG signal
    filtered_signal = filtfilt(b, a, abp_signal)
    
    #Reverse the filtered array and do butterworth filter again, Note only one array is reversed
    filter_reverse = filtered_signal[::-1]
    reverse_filtered_signal = filtfilt(b, a, filter_reverse)

    # Final Filtered array
    Final_filtered_array = reverse_filtered_signal[::-1]
    
    return Final_filtered_array


# Define the Butterworth filter
def butter_lowpass(cutoff, fs, signal, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='lowpass', analog=False)
    # Apply the low pass filter to the PPG signal
    filtered_signal = filtfilt(b, a, signal)
    
    #Reverse the filtered array and do butterworth filter again, Note only one array is reversed
    filter_reverse = filtered_signal[::-1]
    reverse_filtered_signal = filtfilt(b, a, filter_reverse)

    # Final Filtered array
    Final_filtered_array = reverse_filtered_signal[::-1]
    
    return Final_filtered_array




# Load the data from the Excel sheet
df = pd.read_excel(r'H:\HTIC\Vasc_Age\ABP.xlsx')

# Extract the PPG signal
abp_signal = df['ABP_RAW'].values
abp_signal =  abp_signal[40:]
#print(abp_signal)

# Define the sampling frequency and the cutoff frequency for the filter
fs = 125  # Hz
cutoff = 5  # Hz

Final_filtered_array_blf = butter_lowpass(cutoff, fs, abp_signal)


###################################################  Cycle cutting ################################################
threshold = 2

# Normalizse the graph
CC_normalizse = min_max_normalize(Final_filtered_array_blf)

# Passing through first BPF
CC_BPF1_signal = butter_bandpass(0.05, 1, fs, CC_normalizse)

#Passing through Second BPF
CC_BPF2_signal = butter_bandpass(0.1250, 2, fs, CC_BPF1_signal)

# Normalizse the graph again after passing through BPF2
CC_normalizse_2 = min_max_normalize(CC_BPF2_signal)

# detemining the valley points 
CC_valley_indices = argrelextrema(CC_normalizse_2, np.less, order = threshold)[0]
print(CC_valley_indices)
print(type(CC_normalizse_2))

#################################  Finding the cycle peak and cycle valley(ABP signal)  #########
cycle_peak = []
cycle_valley = []
for i in range(1, len(CC_valley_indices)):
    one_cycle = Final_filtered_array_blf[CC_valley_indices[i-1]:CC_valley_indices[i]]
    #print(one_cycle)
    peak_index = np.argmax(one_cycle)
    valley_index = np.argmin(one_cycle)
    peak_global_index = CC_valley_indices[i-1] + peak_index
    valley_global_index = CC_valley_indices[i-1] + valley_index
    # Append the global indices to the respective lists
    cycle_peak.append(peak_global_index)
    cycle_valley.append(valley_global_index)  
print(cycle_valley)
print(cycle_peak)



######################################  With 5% offset   ############################
Valley = np.zeros((len(CC_valley_indices) - 1, 2))

for i in range(1, len(CC_valley_indices) - 1):
    # Assuming valley_global_index is defined properly
    size_cycle = cycle_valley[i] - cycle_valley[i-1]
    offset = int(0.05 * size_cycle)
    first_point = cycle_valley[i-1] - offset
    second_point = cycle_valley[i] + offset
    Valley[i][0] = first_point
    Valley[i][1] = second_point
####################################################################
cut_graph = Final_filtered_array_blf[int(Valley[5][0]):int(Valley[5][1])]

plt.figure(figsize=(20, 4))
plt.scatter(range(len(cut_graph)), cut_graph, s=10)  # s is the size of the points
plt.title('Filtered PPG Signal')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.show()


print(len(cut_graph))