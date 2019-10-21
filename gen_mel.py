"""generate mel-spectrogram from wav"""
import librosa
from scipy import misc
import pickle
import numpy as np
import audio
import hparams_gen_melspec as hparams
import os
import glob
from tqdm import tqdm
wavs=glob.glob('./test/p228_F_05069.wav')
write_path='./test/'
for wav_path in tqdm(wavs):

	basename=os.path.basename(wav_path).split('.wav')[0]
	wav = audio.load_wav(wav_path)
	wav = wav / np.abs(wav).max() * hparams.hparams.rescaling_max

	out = wav
	constant_values = 0.0
	out_dtype = np.float32

	mel_spectrogram = audio.melspectrogram(wav).astype(np.float32).T
	# print(mel_spectrogram.shape)


	misc.imsave(os.path.join(write_path,basename+'.png'),mel_spectrogram)
	mels=[]
	mels.append((basename,mel_spectrogram))
	with open(os.path.join(write_path,basename+'.pkl'),'wb') as handle:
		pickle.dump(mels,handle)

