from scipy.io.wavfile import read as wav_read
import numpy as np
from scipy.fftpack import fft, ifft
import math
from scipy.interpolate import UnivariateSpline
from scipy.signal import spectrogram
import subprocess as sp
from warnings import warn
import sys

def log(txt):
    print(txt, file=sys.stderr)

def normalize(wave):
    def unbias(wave):
        return wave - wave.mean()
    wave = unbias(wave)
    return wave / np.max(np.abs(wave))

# FILE LOADING ----------------------------------------------------

log('Loading file...')

rate, wave = wav_read('./input.wav')
wave = normalize(wave)
length = wave.shape[0] / rate
cnt = 100

def time_to_wi(time, rate):
    return (time * rate).astype(int)

# SEARCHING SILENCE ---------------------------------------------------------

cnt_per_sec = 10
VOLUME_FOR_SILENCE = 0.03

def est1(seg):
    return np.mean(np.abs(seg))

def get_silence(wave):
    segs_num = int(length * cnt_per_sec)
    is_silence = np.zeros(shape=segs_num)
    for i in range(segs_num):
        s1 = rate * i // cnt_per_sec
        s2 = rate * (i + 1) // cnt_per_sec
        vol = est1(wave[s1:s2])
        is_silence[i] = 1 if vol > VOLUME_FOR_SILENCE else 0
    return is_silence

# ----------

MIN_SEGMENT_LENGTH = 1

def eject_list_of_segments_silenceless(oldwave):
    old_silence_mask = get_silence(wave)
    silence_mask = np.concatenate(([0], old_silence_mask, [0]))
    
    d = silence_mask[1:] - silence_mask[:-1]
   
    start = np.argwhere(d > 0).ravel()
    stop  = np.argwhere(d < 0).ravel()
    
    segments = []
    pauses_lens = [0.5]
    for i in np.arange(start.shape[0]):
        new_part = wave[time_to_wi(start[i] / cnt_per_sec, rate) : time_to_wi(stop[i] / cnt_per_sec, rate)]
        if len(new_part) > MIN_SEGMENT_LENGTH:
            segments.append(new_part)
            pauses_lens.append(0.0)
        if i < len(start) - 1:
            pauses_lens[len(pauses_lens) - 1] += (start[i + 1]  - stop[i]) / cnt_per_sec
    return start / cnt_per_sec, stop / cnt_per_sec, pauses_lens
    
    
seg_starts, seg_stops, pauses_lens = eject_list_of_segments_silenceless(wave)

# AUTOCORRELATION ---------------------------------------------------------

log('Autocorrelation..')

def determine(left, right):
    base = 440
    i1 = math.log(left / base) / math.log(2) * 12
    i2 = math.log(right / base) / math.log(2) * 12
    i1 = math.ceil(i1)
    i2 = math.floor(i2)
    if i1 == i2:
        return i1
    else:
        warn("choose between {}, {}".format(i1, i2))
        return i1

def get_pitch(wave, pos, length):
    # print(wave.shape, pos)
    try:
        wave = wave[pos:]
        wave = wave[:int(length / cnt * rate)]
        # wave = np.hamming(wave.shape[0]) * wave

        nwave = np.zeros(shape=wave.shape[0] * 2)
        nwave[:wave.shape[0]] = wave
        window = np.flipud(nwave)

        corr = ifft(fft(nwave) * fft(window)).real
        n = corr.shape[0]
        corr = np.roll(corr, n)[:n]

        # show(corr)
        # plt.show()

        corr = corr[:corr.shape[0] // 2]
        diff = np.diff(corr)
        maxs = np.argwhere(np.logical_and(diff[:-1] > 0, diff[1:] < 0)).squeeze()
        maxs = maxs[np.argwhere(corr[maxs] > 0.95 * corr[maxs].max()).squeeze()]
        # print(maxs)
        if len(maxs.shape) == 0:
            ind = int(maxs)
        else:
            ind = maxs[0]

        # print(ind)
        if ind == 1:
            show(corr)
        return rate / (ind + 2), rate / (ind if ind != 0 else 1)
    except Exception as e:
        return 1, 10
    
    
res = []
for pos in np.arange(0, length - 1 / cnt, 1 / cnt):
    pitch = get_pitch(wave, int(pos * rate), length)
    det = determine(*pitch)
    res.append(det)

#show_results(wave, rate, res)

# FILTERING ------------------------------------------------------------

log('Filtering...')

def movingaverage(interval, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, 'same')
    
PEAK_FACTOR = 0.5
MAX_PEAK_LENGTH = 50

def filter_lunges_in_segment(olddata, times):
    data = olddata[:]
    smooth_data = movingaverage(data, np.minimum(70, len(data)))
    i = 1
    places = []
    while i < len(data):
        j = 0
        if np.abs(data[i] - smooth_data[i]) > PEAK_FACTOR * np.abs(smooth_data[i]):
            last_sign = data[i] - smooth_data[i-1]
            places.append(i)
            delta = data[i] - data[i-1]
            start_value = data[i-1]
            while j < MAX_PEAK_LENGTH and i + j < len(data) :                
                if (not (np.abs(data[i+j] - smooth_data[i+j]) > PEAK_FACTOR * np.abs(smooth_data[i+j]))) or (data[i+j] - smooth_data[i+j]) * last_sign < 0:
                    break
                data[i+j] -= delta
                j += 1
        i = i + j + 1
    return data
    
SILENCE_CONST = -1108

def filter_all_segments(predict, seg_starts, seg_stops, pauses):
    times = np.arange(0, len(predict) / cnt, 1 / cnt)
    pr_len = len(predict)
    near_notes = np.array(predict[:])
    
    start = (seg_starts * cnt).astype(int)
    stop = (seg_stops * cnt).astype(int)
    
    tmp1 = start[0] - 1 if start[0] > 0 else 0
    tmp2 = (stop[len(stop) - 1] + 1 if start[0] > 0 else 0)
    near_notes[:tmp1] = SILENCE_CONST
    near_notes[tmp2:] = SILENCE_CONST
    for i in np.arange(len(start)):
        l = start[i]
        r = stop[i]
        if (i > 0):
             near_notes[stop[i-1] : start[i]] = SILENCE_CONST  
        near_notes[l:r] = filter_lunges_in_segment(predict[l:r], times[l:r])
    return near_notes
    
corrected_notes = filter_all_segments(res, seg_starts, seg_stops, pauses_lens)
#show_results_filtered(wave, rate, corrected_notes, corrected_notes, np.arange(0, len(corrected_notes) / cnt, 1 / cnt))

# KMEANS -----------------------------------------------------

log('kmeans...')

class DSU:
    def __init__(self, count):
        self.ar = list(range(count))
        self.rank = [0] * count
        self.get = self.get_col
    def get_col(self, index):
        if self.ar[index] == index:
            return index
        self.ar[index] = self.get_col(self.ar[index])
        return self.ar[index]
    def unite(self, a, b):
        a = self.get_col(a)
        b = self.get_col(b)
        if self.rank[a] > self.rank[b]:
            a, b = b, a
        self.ar[a] = b
        if self.rank[a] == self.rank[b]:
            self.rank[b] += 1

def k_means_2(seg, k):
    segs = [(i, i + 1) for i in range(len(seg))]
    for i in range(len(seg) - k + 1):
        mval = 1e10
        pos = -1
        for j in range(len(segs) - 1):
            u1, u2 = segs[j]
            v1, v2 = segs[j + 1]
            m1 = np.median(seg[u1:u2])
            m2 = np.median(seg[v1:v2])
            if mval > abs(m1 - m2):
                mval = abs(m1 - m2)
                pos = j
        segs[pos] = (segs[pos][0], segs[pos + 1][1])
        del segs[pos + 1]
    return segs
    
#  /////
    
MIN_NOTE_LENGTH = 0.1 # secs
NOTES_PER_PHRASE = 20

start = (seg_starts * cnt).astype(int)
stop = (seg_stops * cnt).astype(int)

all_segments = []

for i in np.arange(len(start)):
    seg = corrected_notes[start[i]:stop[i]]
    
    kmeans_notes = k_means_2(seg, k=NOTES_PER_PHRASE)
    

    for u, v in kmeans_notes:
        if (v - u) / cnt > MIN_NOTE_LENGTH:
            all_segments.append((u + start[i], v + start[i], np.median(seg[u:v])))

    
# SOUND GENERATION --------------------------------------

log('Generating sound...')

from synthesis import AudioBuilder

r = 44100
builder = AudioBuilder(length=25.0, rate=r)
lsize = len(all_segments)

for left, right, frq in all_segments:
    builder.add_freq(int(frq), int((left / cnt) * r), dur=int((right-left) / cnt * r))

builder.write('output.wav')
