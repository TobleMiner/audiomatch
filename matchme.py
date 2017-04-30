#!/bin/env python

import sys
import numpy as np
import soundfile as sf
from soundfile import SoundFile
import math
from multiprocessing.pool import ThreadPool
import itertools
import datetime

def dumpChannelData(channels):
	for chan in channels:
		for slice in chan:
			for bin in slice:
				print(bin)
			

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
		self.channels = self._fft(reffile, self.samplerate)

	def _fft(self, file, samplerate, offset=0):
		lenpart = math.floor(file.samplerate * self.slicewidth / Matcher.WIDTH_BASE)
		channelfft = []
		results = [[] for _ in range(file.channels)]
		file.seek(math.floor(offset * file.samplerate))
		for slice in file.blocks(lenpart, always_2d=True):
			channels = [[sample[chan] for sample in slice] for chan in range(file.channels)]
			for chan in range(file.channels):
				results[chan].append(self.pool.apply_async(self._partfft, args=(channels[chan], samplerate)))
		return [[res.get() for res in results[chan]] for chan in range(file.channels)]

	def _partfft(self, data, samplerate):
		begin = datetime.datetime.now()
#		print("Starting fft, {} data points".format(len(data)))
		fftdata = np.fft.fft(data)
		freqs = np.fft.fftfreq(len(fftdata))
		bins = [0 for _ in range(math.floor((self.freqmax - self.freqmin) / self.freqstep))]

		maxval = 0
		for i in range(len(fftdata)):
			freq = abs(freqs[i] * samplerate)
			if freq < self.freqmin or freq >= self.freqmax:
				continue
			sample = abs(fftdata[i])
			bin = math.floor(freq / self.freqstep)
			bins[bin] += sample
			maxval = max(maxval, bins[bin])

		for i in range(len(bins)):
			bins[i] /= maxval
		end = datetime.datetime.now()
#		print("FFT step took {} ms".format((end - begin).total_seconds() * 1000))
		return bins


	def match(self, file, offset=0):
		matchthreshold = math.floor((self.freqmax - self.freqmin) / self.freqstep) * 0.1
		print("Matching threshold is {}".format(matchthreshold))
		channeldata = self._fft(file, file.samplerate, offset)
		matches = []
		for chan in range(file.channels):
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
	matcher = Matcher(file, slicewidth=2000)

#dumpChannelData(matcher.channels)

for i in range(0, 1):
	print("Offset {}".format(i/10))
	with SoundFile(fname) as file:
		times = {}
		for match in sorted(matcher.match(file, i/10), key=lambda entry: entry[1][3]):
#			print(match)
			seconds = str(match[1][1] - match[1][2])
			entry = None
			if not seconds in times:
				entry = [0, 0]
			else:
				entry = times[seconds]
			entry[0] += 1
			entry[1] += match[1][3]
			times[seconds] = entry

		timesort = []
		for key in times:
			times[key][1] /= times[key][0]
			timesort.append((key, times[key]))

		timesort = sorted(timesort, key=lambda x: x[1][1], reverse=True)

#		for time in timesort:
#			print(time)
		print(timesort[len(timesort) - 1])
