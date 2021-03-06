#!/bin/env python

import sys
import numpy as np
import soundfile as sf
from soundfile import SoundFile
import math
from multiprocessing.pool import ThreadPool
import itertools
import datetime
from datetime import timedelta as dt
import matplotlib.pyplot as plot
from argparse import ArgumentParser
import os

def dumpChannelData(channels):
	for chan in channels:
		for slice in chan:
			for bin in slice:
				print(bin)
class Match():
	def __init__(self, startpos, likeness):
		self.likeness = likeness
		self.startpos = startpos

class FftMatch(Match):
	def __init__(self, trackpos, refpos, likeness):
		super().__init__(trackpos - refpos, likeness)
		self.refpos = refpos
		self.trackpos = trackpos

class GroupMatch(Match):
	def __init__(self, startpos):
		super().__init__(startpos, 0)
		self.count = 0
		self.aggregate = 0

	def __str__(self):
		return '{}, {}, {}, {}'.format(self.startpos, self.count, self.likeness, self.aggregate)

	def __repr__(self):
		return str(self)

	def toCSV(self):
		return '{};{};{};{}'.format(self.startpos.total_seconds(), self.count, self.likeness, self.aggregate)

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
		foffset = math.floor(offset / Matcher.WIDTH_BASE * file.samplerate)
		file.seek(foffset)
		for slice in file.blocks(lenpart, always_2d=True):
			channels = [[sample[chan] for sample in slice] for chan in range(file.channels)]
			for chan in range(file.channels):
				results[chan].append(self.pool.apply_async(self._partfft, args=(channels[chan], samplerate)))
		return [[res.get() for res in results[chan]] for chan in range(file.channels)]

	def _partfft(self, data, samplerate):
#		begin = datetime.datetime.now()
#		print("Starting fft, {} data points".format(len(data)))
		fftdata = np.fft.fft(data)
		freqs = np.fft.fftfreq(len(fftdata))
		bins = [0 for _ in range(math.floor((self.freqmax - self.freqmin) / self.freqstep))]
#		fftend = datetime.datetime.now()
		maxval = 0
		for i in range(len(fftdata)):
			freq = abs(freqs[i] * samplerate)
			if freq < self.freqmin or freq >= self.freqmax:
				continue
			sample = abs(fftdata[i])
			bin = math.floor(freq / self.freqstep)
			bins[bin] += sample
			maxval = max(maxval, bins[bin])

		bins = np.divide(bins, maxval)
#		end = datetime.datetime.now()
#		print("FFT step took {} ms, postprocessing took {} ms, total : {} ms".format((fftend - begin).total_seconds() * 1000, (end - fftend).total_seconds() * 1000, (end - begin).total_seconds() * 1000))
		return bins


	def match(self, file, offset=0, threshold=0.9):
		matchthreshold = threshold
		print("Matching threshold is {}".format(matchthreshold))
		channeldata = self._fft(file, file.samplerate, offset)
		matches = []
		for chan in range(file.channels):
			for match in self._fft_fuzzy_match(self.channels[chan], channeldata[chan], matchthreshold):
				matches.append(match)

		timemap = {}
		aggrmax = 0
		for match in matches:
			key = str(match.startpos)
			groupmatch = None
			if not key in timemap:
				groupmatch = GroupMatch(match.startpos)
			else:
				groupmatch = timemap[key]
			groupmatch.count += 1
			groupmatch.likeness += match.likeness
			timemap[key] = groupmatch
			aggrmax = max(aggrmax, groupmatch.likeness)

		for key in timemap:
			match = timemap[key]
			match.aggregate = match.likeness / aggrmax
			match.likeness /= match.count

		return [timemap[key] for key in timemap]

	def _fft_fuzzy_match(self, reference, sample, matchthreshold):
		matches = []
		for refid in range(len(reference)):
			for sampleid in range(len(sample)):
				likeness = self._fft_match(reference[refid], sample[sampleid])
				if not likeness >= matchthreshold:
					continue
				avg = likeness
				for i in range(1, len(reference) - refid):
					if sampleid + i >= len(sample):
						break
					likeness = self._fft_match(reference[refid + i], sample[sampleid + i])
					if not likeness >= matchthreshold:
						break
					avg += likeness
					if i == len(reference) - refid - 1:
						matches.append(FftMatch(dt(seconds=sampleid * self.slicewidth / Matcher.WIDTH_BASE),
							dt(seconds=refid * self.slicewidth / Matcher.WIDTH_BASE),
							avg / (len(reference) - refid)))
		return matches


	def _fft_match(self, sample_a, sample_b):
		assert len(sample_a) == len(sample_b)
		diff = 0
		for bin in range(len(sample_a)):
			diff += math.sqrt(abs(sample_a[bin] - sample_b[bin]))
		return (len(sample_a) - diff) / len(sample_a)

parser = ArgumentParser()

parser.add_argument('-r', '--reportdir', type=str, help='Dir to save reports in')
parser.add_argument('-o', '--offset', type=int, help='Offset in files being matched')
parser.add_argument('-s', '--steps', type=int, help='Steps to apply offset')
parser.add_argument('-t', '--slicewidth', type=int, help='Width of audio fft slices')
parser.add_argument('reffile', help='Reference file')
parser.add_argument('matchfiles', help='Files to match against reffile', nargs='+')

args = parser.parse_args()

if not args.reportdir: args.reportdir = 'report'
if not args.offset: args.offset = 0
if not args.steps: args.steps = 1
if not args.slicewidth: args.slicewidth = 2000

if not os.path.isdir(args.reportdir):
	os.mkdir(args.reportdir)

print('Using slicewidth of {} s'.format(args.slicewidth / Matcher.WIDTH_BASE))

matcher = None
print("Loading reference file '{}'".format(args.reffile))
with SoundFile(args.reffile) as file:
	matcher = Matcher(file, slicewidth=args.slicewidth)

#dumpChannelData(matcher.channels)

songcount = 0
for matchfile in args.matchfiles:
	songcount += 1
	reportfile = '{}.txt'.format(os.path.join(args.reportdir, os.path.basename(matchfile)))
	print("[{}/{}] Starting matching on '{}', final offset {}, steps {}"
		.format(songcount, len(args.matchfiles), matchfile, args.offset, args.steps))
	with open(reportfile, 'w') as report:
		print("Writing report to '{}'".format(reportfile))
		report.write('# slicewidth={}, offset={}, steps={}\n'.format(args.slicewidth, args.offset, args.steps))
		for i in range(0, args.steps):
			offset = (i + 1) * args.offset / args.steps
			report.write(':{}\n'.format(offset))
			print("{}: Offset {}".format(matchfile, offset))
			with SoundFile(matchfile) as file:
				times = {}
				agrmax = 0

				begin = datetime.datetime.now()
				matches = matcher.match(file, offset)
				end = datetime.datetime.now()

				print("Step took {} seconds".format((end - begin).total_seconds()))

				for match in matches:
					report.write('{}\n'.format(match.toCSV()))

				print('Last few by likeness:')
				for match in sorted(matches, key=lambda entry: entry.likeness)[-5:]:
					print(match)

				print('Last few by aggregate:')
				for match in sorted(matches, key=lambda entry: entry.aggregate)[-5:]:
					print(match)

				# Visualize results as bins in time domain
#				TIME_GRANULARITY = 10 # seconds
#				bins = [0 for _ in range(0, math.ceil(matches[-1:][0].startpos.total_seconds()), TIME_GRANULARITY)]
#				x = range(0, math.ceil(matches[-1:][0].startpos.total_seconds()), TIME_GRANULARITY)
#				for t in range(len(bins)):
#					for match in matches:
#						time = match.startpos.total_seconds()
#						if(time >= t * TIME_GRANULARITY and time < (t + 1) * TIME_GRANULARITY):
#							bins[t] += match.count
#				plot.bar(x, bins, color="green")
#				plot.savefig("plot.png")
