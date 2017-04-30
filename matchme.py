#!/bin/env python

import sys
import numpy as np
import soundfile as sf
from soundfile import SoundFile
import math
from multiprocessing.pool import ThreadPool
import itertools
import datetime

class Matcher():
	WIDTH_BASE = 1000 # Milliseconds
	def __init__(self, reffile, freqmin=0, freqmax=20000, freqstep=10, slicewidth=2000, numthreads=8):
		self.freqmin = freqmin
		self.freqmax = freqmax
		self.freqstep = freqstep
		self.slicewidth = slicewidth
		self.pool = ThreadPool(numthreads)
		self._loadref(reffile)

	def _loadref(self, reffile):
		self.channelcnt = reffile.channels
		self.samplerate = reffile.samplerate
		data = reffile.read(always_2d=True)
		channels = [[sample[chan] for sample in data] for chan in range(self.channelcnt)]
		self.channels = [None for _ in range(self.channelcnt)]
		for chan in range(self.channelcnt):
			self.channels[chan] = self._fft(channels[chan], self.samplerate)

	def _fft(self, data, samplerate):
		numparts = math.floor(len(data) * Matcher.WIDTH_BASE / samplerate / self.slicewidth)
		print(numparts)
		partsamples = math.floor(len(data) / numparts)
		samples = [data[partsamples * i:partsamples * (i + 1) - 1] for i in range(numparts)]
		print("Before async")
		results = [self.pool.apply_async(self._partfft, args=(samples[part], samplerate)) for part in range(numparts)]
		return [res.get() for res in results]

	def _partfft(self, data, samplerate):
		begin = datetime.datetime.now()
		print("Starting fft, {} data points".format(len(data)))
		fftdata = np.fft.fft(data)
		freqs = np.fft.fftfreq(len(fftdata))
		bins = [0 for _ in range(math.floor((self.freqmax - self.freqmin) / self.freqstep))]

		maxval = 0
		for i in range(len(fftdata)):
			freq = abs(freqs[i] * samplerate)
			if freq < self.freqmin or freq > self.freqmax:
				continue
			sample = abs(fftdata[i])
			bin = math.floor(freq / self.freqstep)
			bins[bin] += sample
			maxval = max(maxval, bins[bin])

		for i in range(len(bins)):
			bins[i] /= maxval
		end = datetime.datetime.now()
		print("FFT step took {} ms".format((end - begin).total_seconds() * 1000))
		return bins

	def match(self, file):
		data = file.read(always_2d=True)
		channels = [[sample[chan] for sample in data] for chan in range(self.channelcnt)]
		matchthreshold = math.floor((self.freqmax - self.freqmin) / self.freqstep) * 0.1
		print("Matching threshold is {}".format(matchthreshold))
		channeldata = [None for _ in range(self.channelcnt)]
		matches = []
		for chan in range(file.channels):
			channeldata[chan] = self._fft(channels[chan], file.samplerate)
			for match in self._fft_fuzzy_match(self.channels[chan], channeldata[chan], matchthreshold):
				matches.append((chan, match))
		return matches			
	
	def _fft_fuzzy_match(self, reference, sample, matchthreshold):
		matches = []
		for refid in range(len(reference)):
			for sampleid in range(len(sample)):
				likeness = self._fft_match(reference[refid], sample[sampleid])
				if not likeness <= matchthreshold:
					continue
				avg = likeness
				for i in range(1, len(reference) - refid):
					if sampleid + i >= len(sample):
						break
					likeness = self._fft_match(reference[refid + i], sample[sampleid + i])
					if not likeness <= matchthreshold:
						break
					avg += likeness
					if i == len(reference) - refid - 1:
						matches.append(("Match at", datetime.timedelta(seconds=sampleid*self.slicewidth/Matcher.WIDTH_BASE), datetime.timedelta(seconds=refid*self.slicewidth/Matcher.WIDTH_BASE), avg / (len(reference) - refid)))
		return matches


	def _fft_match(self, sample_a, sample_b):
		assert len(sample_a) == len(sample_b)
		diff = 0
		for bin in range(len(sample_a)):
			diff += math.sqrt(abs(sample_a[bin] - sample_b[bin]))
		return diff

rname = 'airplanes_sample.ogg'
fname = 'airplanes.ogg'
matcher = None
with SoundFile(rname) as file:
	matcher = Matcher(file, slicewidth=1000)

print(matcher.channels)

with SoundFile(fname) as file:
	times = {}
	for match in sorted(matcher.match(file), key=lambda entry: entry[1][3]):
		print(match)
		seconds = str(match[1][1] - match[1][2])
		if not seconds in times:
			times[seconds] = 0
		times[seconds] += 1

	timesort = []
	for key in times:
		timesort.append((key, times[key]))

	timesort = sorted(timesort, key=lambda x: x[1])

	for time in timesort:
		print(time)

print(matcher.freqstep)
