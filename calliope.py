#!/usr/bin/env python3
import argparse
import logging
import colouredLogger
import numpy as np
import numpy.fft as fft
from bitstring import BitArray
from bitstring import ConstBitStream
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav

# CONSTANTS
SAMPLE_RATE = 48000
BAUD_RATE = 2000
AMPLITUDE = 0.5
FFT_SIZE = 1024
THRESHOLD = 2

FILTER_LOWER = 75

PREAMBLE_CHIPS   = [2200,2400]
SYMBOL_BITS     = 1
CARRIER_TABLE   = [2000, 2250] #, 3000, 3500]
SYMBOLS = ['0b0', '0b1']

ALLOWED_FORMATS = ['pcm', 'wav']

# derived parameters
SYM_SAMPLES = int(SAMPLE_RATE * (1/BAUD_RATE))
NUM_SYMBOLS = len(CARRIER_TABLE)
SYMBOL_TABLE = [ (AMPLITUDE * np.sin(2 * np.pi * FREQ * np.linspace(0, 1/BAUD_RATE, SYM_SAMPLES, dtype=np.float32))) for FREQ in CARRIER_TABLE]

# create logger with 'spam_application'
logger = logging.getLogger("calliope")
logger.setLevel(logging.DEBUG)

# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

ch.setFormatter(colouredLogger.CustomFormatter())

logger.addHandler(ch)


def chooseFormat(fileName, format):
    if format is not None:
        format = format.lower()
        if format in ALLOWED_FORMATS:
            return format
        else:
            logger.error("Format " + str(format) + " is not supported")
            raise SystemExit
    else:
        # infer from extension
        extension =  fileName.split('.')[-1]
        if extension.lower() in ALLOWED_FORMATS:
            return extension.lower()
        else:
            # can't figure it out, default to pcm
            return "pcm"


def diffEncode(bitarray):
    pass

def diffDecode(bitarray):
    pass

def pad(array, n):
    """
    ISO/IEC 9797-1 pad array to blocks of size n
    """
    pads = len(array) % n
    return array + [0]*(n - (len(array)%n))

def chunk(array, n):
    for i in range(0, len(array), n):
        yield array[i:i+n]

def modulate(bitarray):
    """
    FSK modulates the bitarray
        bitarray: bitstring BitArray like
        returns: a float numpy waveform of the signal
    """
    chunks = chunk(bitarray, SYMBOL_BITS)
    logger.debug("Modulating {} bits".format(len(bitarray)))
    wave = np.zeros(len(bitarray)*SYM_SAMPLES, dtype=np.float32)
    for i in range(0, len(bitarray)):
        wave[i*SYM_SAMPLES:(i+1)*SYM_SAMPLES] = SYMBOL_TABLE[bitarray[i]]
    plt.plot(wave)
    plt.show()
    return wave

def demodulate(rxSignal):
    """
    FSK demodulate the a signal
        rxSignal: numpy float32 array
        returns: BitArray like
    """
    # band pass the carrier
    # 
    demod = np.zeros((NUM_SYMBOLS, len(rxSignal)), dtype=np.float32)
    space  = np.linspace(0, len(rxSignal) / SAMPLE_RATE, len(rxSignal), dtype=np.float32)
    for i in range(NUM_SYMBOLS):
        wave = AMPLITUDE * np.sin(2 * np.pi * CARRIER_TABLE[i] * space)
        demod[i] = wave * rxSignal
    data = BitArray()
    for i in range(0, demod.shape[1], SYM_SAMPLES):
        offset = list()
        for j in range(0, NUM_SYMBOLS):
            offset += [np.abs(np.trapz(demod[j][i:i+SYM_SAMPLES]))]
        # look
        data.append(SYMBOLS[offset.index(max(offset))])
    return data

def carrierFilter(rxSignal):
    # spec = fft.fft(rxSignal)
    # freqs = fft.fftfreq(FFT_SIZE, 1/SAMPLE_RATE)
    # plt.plot(freqs, spec)
    # plt.show()
    # spec[100:FFT_SIZE-100] = 0.0
    # plt.plot(spec)
    # plt.show()
    # cleaned = fft.ifft(spec)
    return rxSignal

def correlate(rxSignal):
    return

def main():
    parser = argparse.ArgumentParser(description="Transmit files using sound")
    parser.add_argument('files', metavar="FILE", help="Files to be transmitted using sound", nargs="+")
    parser.add_argument('-o', dest='outFile', help="Save the output to file rather than playing")
    parser.add_argument('-f', dest='format', help="Format to save to 'wav/pcm'")
    parser.add_argument('-d', dest='demodulate', default=False, action='store_true', help="Demodulate file")
    args = parser.parse_args()
    
    logger.debug("Sample Rate: {}".format(SAMPLE_RATE))
    logger.debug("Baud Rate: {}".format(BAUD_RATE))

    # diffEncode(BitArray(bin="101"))
    usingFormat = chooseFormat(args.outFile, args.format)
    if(args.demodulate):
        with open(args.files[0], 'rb') as fp:
            if usingFormat == 'pcm':
                wave = np.fromfile(fp, dtype=np.float32)
            elif usingFormat == 'wav':
                rate, wave = wav.read(fp)
        cleaned = carrierFilter(wave)
        data = demodulate(cleaned)
        if args.outFile is not None:
            output = args.outFile
        else:
            output = 'calliope.out'
        with open(output, 'wb') as fp:
            data.tofile(fp)
    else:
        data = ConstBitStream(filename=args.files[0])
        wave = modulate(data)
        #plt.plot(wave)
        #plt.show()
        if(args.outFile is not None):
            with open(args.outFile, 'wb') as fp:
                if usingFormat == 'pcm':
                    logging.debug("Saving as pcm")
                    wave.tofile(fp)
                elif usingFormat == 'wav':
                    logging.debug("Saving as Wav")
                    wav.write(fp, SAMPLE_RATE, wave)



if __name__ == '__main__':
    main()
