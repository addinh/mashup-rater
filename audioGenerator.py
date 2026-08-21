import pyrubberband as pyrb
import librosa
import numpy as np
import pandas as pd
from random import randint
import os
from typing import List

DATA = pd.read_csv('../AutoMixer/data/data.csv')

SR = 44100

from typing import NamedTuple

class BasicStemInfo(NamedTuple):
    id: int
    ver: str
    bpm: int
    key: int
    length: int

class StemInfo(NamedTuple):
    id: int
    ver: str
    baseBPM: int
    startBeat: int
    durationBeat: int
    targetBPM: int
    pitchShift: int

def getStem(info: StemInfo, stemName: str) -> np.ndarray:
    stem, _ = librosa.load(f'../AutoMixer/data/audio/{info.id}/{info.id}{info.ver}_{stemName}.wav', sr=SR)
    stem = stem[int(info.startBeat*(240/info.baseBPM)*SR):int((info.startBeat+info.durationBeat)*(240/info.baseBPM)*SR)]
    stemAdjusted = pyrb.pitch_shift(pyrb.time_stretch(stem, SR, info.targetBPM / info.baseBPM), SR, info.pitchShift)
    return stemAdjusted

def mixInst(info: StemInfo):
    if os.path.exists(f'../AutoMixer/data/audio/{info.id}/{info.id}{info.ver}_inst.wav'):
        return getStem(info, 'inst')
    else:
        bass = getStem(info, 'bass')
        drums = getStem(info, 'drums')
        other = getStem(info, 'other')
        max_length = max(bass.shape[0], drums.shape[0], other.shape[0])
        mix = np.pad(bass, pad_width=(0, max_length - bass.shape[0]), constant_values=0) + np.pad(drums, pad_width=(0, max_length - drums.shape[0]), constant_values=0) \
                    + np.pad(other, pad_width=(0, max_length - other.shape[0]), constant_values=0)
        return mix

def mix(iInfo: StemInfo, vInfo: StemInfo):
    inst = mixInst(iInfo)
    vocals = getStem(vInfo, 'vocals')
    max_length = max(inst.shape[0], vocals.shape[0])
    mix = np.pad(inst, pad_width=(0, max_length - inst.shape[0]), constant_values=0) + np.pad(vocals, pad_width=(0, max_length - vocals.shape[0]), constant_values=0)
    return mix

def originalMix(info: StemInfo):
    inst = mixInst(StemInfo(info.id, info.ver, info.baseBPM, info.startBeat, info.durationBeat, info.baseBPM, 0))
    vocals = getStem(StemInfo(info.id, info.ver, info.baseBPM, info.startBeat, info.durationBeat, info.baseBPM, 0), 'vocals')
    max_length = max(inst.shape[0], vocals.shape[0])
    mix = np.pad(inst, pad_width=(0, max_length - inst.shape[0]), constant_values=0) + np.pad(vocals, pad_width=(0, max_length - vocals.shape[0]), constant_values=0)
    return mix

def getRandomStem(skip_id = None):
    randomRow = DATA.sample().iloc[0]
    id = int(randomRow['ID'][0])
    while id == skip_id:
        randomRow = DATA.sample().iloc[0]
        id = int(randomRow['ID'][0])
    ver = randomRow['ID'][1:]
    bpm = int(randomRow['BPM'])
    key = int(randomRow['EqKey'])
    length = int(randomRow['Length'])
    return BasicStemInfo(id, ver, bpm, key, length)

def getMashupID(instInfo: BasicStemInfo, vocalsInfo: BasicStemInfo):
    return f'{instInfo.id}{instInfo.ver}-{vocalsInfo.id}{vocalsInfo.ver}'

def getPitchShiftAmt(base: int, target: int):
    if target - base > 6:
        return target - base - 12
    elif target - base < -6:
        return target - base + 12
    return target - base

def generateInstVocalsMashupMetadata():
    instInfo = getRandomStem()
    vocalsInfo = getRandomStem(instInfo.id)
    print("inst:", instInfo)
    print("vocals:", vocalsInfo)

    avgBPM = (instInfo.bpm + vocalsInfo.bpm) // 2
    print("bpm:", avgBPM)
    chosenKey = vocalsInfo.key # randint(0, 11)
    print("key:", chosenKey)

    minLength = min(instInfo.length, vocalsInfo.length)
    length = randint(1, minLength // 4)
    print("duration in bars:", length)

    instOffset = 4 * randint(0, instInfo.length // 4 - length)
    vocalsOffset = 4 * randint(0, vocalsInfo.length // 4 - length)

    instMashupInfo = StemInfo(instInfo.id, instInfo.ver, instInfo.bpm, instOffset, 4 * length, avgBPM, getPitchShiftAmt(instInfo.key, chosenKey))
    vocalsMashupInfo = StemInfo(vocalsInfo.id, vocalsInfo.ver, vocalsInfo.bpm, vocalsOffset, 4 * length, avgBPM, getPitchShiftAmt(vocalsInfo.key, chosenKey))

    instExcerpt = originalMix(instMashupInfo)
    vocalsExcerpt = originalMix(vocalsMashupInfo)
    mashup = mix(instMashupInfo, vocalsMashupInfo)

    return instExcerpt, vocalsExcerpt, mashup, {
        'inst': str(instInfo),
        'vocals': str(vocalsInfo),
        'bpm': avgBPM,
        'key': chosenKey,
        'duration': length,
    }