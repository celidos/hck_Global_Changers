#!/usr/bin/env python3
# coding: utf-8

import numpy as np

from scipy.io import wavfile

from os import listdir
from os.path import isfile, join
import re
import subprocess as sp
import tempfile
import os
import warnings

def normalize(wave):
    def unbias(wave):
        return wave - wave.mean()
    wave = unbias(wave)
    return wave / np.max(np.abs(wave))

def convert_freq(wave, fr_from, fr_to):
    if fr_from == fr_to:
        return wave
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as f1, \
            tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as f2:
        f1.close()
        f2.close()
        wavfile.write(f1.name, fr_from, wave)
        # compatibility with python < 3.5
        sp.check_call(['ffmpeg', '-y', '-loglevel', 'quiet', '-i', f1.name, '-acodec', 'pcm_s16le', '-ac', '1', '-ar', 
                str(fr_to), f2.name])
        out_rate, out_wave = wavfile.read(f2.name)
        os.unlink(f1.name)
        os.unlink(f2.name)
        return out_wave

SOURCE_FOLDER = './samples/piano/'
SHIFT = 36

class AudioBuilder:
    def __init__(self, length, rate, prefix=SOURCE_FOLDER):
        self.rate = rate
        self.length = length
        self.data = np.zeros(shape=int(length * rate))
        self.wavs = []
        names = [f for f in listdir(prefix) if isfile(join(prefix, f))] # ???
        names.sort()
        for filename in names:
            fr, wave = wavfile.read(join(prefix, filename))
            self.wavs.append(convert_freq(wave[:, 0], fr, rate))
    def clear(self):
        self.data[...] = 0
    def add_wave(self, wave, pos):
        wave = normalize(wave)
        self.data[pos:wave.shape[0]] += wave
    def add_freq(self, note, pos, dur=None, amp=1.0):
        note += SHIFT
        if not(0 <= note < len(self.wavs)):
            warnings.warn('filter govno')
            return
        max_dur = self.wavs[note].shape[0]
        if dur is None:
            dur = max_dur
        if dur > max_dur:
            dur = max_dur
        if dur + pos > self.data.shape[0]:
            dur = self.data.shape[0] - pos
        if 0 <= note < len(self.wavs):
            self.data[pos:pos+dur] += self.wavs[note][:dur] * amp
        else:
            warnings.warn('unsupported frequency')
    def build(self):
        return normalize(self.wave)
    def write(self, filename):
        wavfile.write(filename, self.rate, normalize(self.data))

if __name__ == '__main__':
    r = 44100
    builder = AudioBuilder(length=3.0, rate=r)

    builder.add_freq(0, 0, dur=r // 4)
    builder.add_freq(2, r // 2, amp=0.5, dur=r // 4)
    builder.add_freq(4, r, amp=0.25, dur=r // 4)
    builder.write('output.wav')
