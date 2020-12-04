#!/usr/bin/env python3
import unittest
import calliope
import colouredLogger
import numpy as np
import logging
from bitstring import BitArray
from bitstring import ConstBitStream
import scipy.io.wavfile as wav

logger = logging.getLogger("calliope")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

ch.setFormatter(colouredLogger.CustomFormatter())

logger.addHandler(ch)

calliope.VISUAL = False


DEAFHEAVEN = """Held my breath and drove through a maze of wealthy homes
I watched how green the trees were
I watched the steep walkways and the white fences
I gripped the wheel
I sweated against the leather
I watched the dogs twist through the wealthy garden
I watched you lay on a towel in grass that exceeded the height of your legs
I gazed into reflective eyes
I cried against an ocean of light
Crippled by the cushion, I sank into sheets frozen by rose pedal toes
My back shivered for your pressed granite nails
Dishonest and ugly through the space in my teeth
Break bones down to yellow and crush gums into blood
The hardest part for the weak was stroking your fingers with rings full of teeth
It's 5 AM and my heart flourishes at each passing moment
Always and forever"""

class TestMethods(unittest.TestCase):
    def testSanity(self):
        with open("./test/testData/test.txt") as file:
            text = file.read()
            self.assertTrue(DEAFHEAVEN == text)

    def test2FSKModulate(self):
        data = ConstBitStream(filename="test/testData/test.txt")
        wave = calliope.modulate(data)
        with open("test/testData/test2FSK_2000_3000.pcm") as file:
            answer = np.fromfile(file, dtype=np.float32)
            self.assertTrue(wave.all() == answer.all())
    
    def testCRCOperations(self):
        a = BitArray(bin="10101010")
        c = BitArray(bin='1010101001111011101001010000000111100100')
        b = a + calliope.bitstringCRC(a)
        self.assertEqual(c,b)
        self.assertTrue(calliope.checkCRC(c))

    def test2FSKDemodulate(self):
        answer = ConstBitStream(filename="test/testData/test.txt")
        with open("test/testData/test2FSK_2000_3000.pcm") as file:
            wave = np.fromfile(file, dtype=np.float32)
            demod = calliope.demodulate(wave)
            self.assertTrue(calliope.checkCRC(demod))
            self.assertTrue(demod[:-32] == answer)

    # def test2FSKDemodulateAWGN(self):
    #     answer = ConstBitStream(filename="test/testData/test.txt")
    #     with open("test/testData/test2FSK_2000_3000Corrupted.wav") as file:
    #         rate, wave = wav.read(file)
    #         demod = calliope.demodulate(wave)
    #         self.assertTrue(demod == answer)

if __name__ == '__main__':
    unittest.main()