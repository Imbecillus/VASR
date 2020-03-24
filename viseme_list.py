# Neti Map (Potamianos, Neti, Gravier,Garg, Senior: Recent advances in the automatic recognition of audiovisual speech. Proceedings of the IEEE, 91(9): 1306-1326, 2003)
# after Eoin Gillen: TCD-TIMIT: A New Database for Audio-Visual Speech Recognition [PhD thesis, University of Dublin]
phonemes_to_neti = {
    'ao' : 'V1',        # Lip-rounding based vowels
    'ah' : 'V1',
    'aa' : 'V1',
    'er' : 'V1',
    'oy' : 'V1',
    'aw' : 'V1',
    'hh' : 'V1',
    'uw' : 'V2',
    'uh' : 'V2',
    'ow' : 'V2',
    'ae' : 'V3',
    'eh' : 'V3',
    'ey' : 'V3',
    'ay' : 'V3',
    'ih' : 'V4',
    'iy' : 'V4',
    'ax' : 'V4',
    'l' : 'A',          # Alveolar-semivowels
    'el' : 'A',
    'r' : 'A',
    'y' : 'A',
    's' : 'B',          # Alveolar fricatives
    'z' : 'B',
    't' : 'C',          # Alveolar
    'd' : 'C',
    'n' : 'C',
    'en' : 'C',
    'sh' : 'D',         # Palato-alveolar
    'zh' : 'D',
    'ch' : 'D',
    'jh' : 'D',
    'p' : 'E',          # Bilabial
    'b' : 'E',
    'm' : 'E',
    'th' : 'F',         # Dental
    'dh' : 'F',
    'f' : 'G',          # Labio-dental
    'v' : 'G',
    'ng' : 'H',         # Velar
    'g' : 'H',
    'k' : 'H',
    'w' : 'H',
    'sil' : 'S',        # Silence
    'sp' : 'S'
}

neti_to_phonemes = {'V1': ['ao', 'ah', 'aa', 'er', 'oy', 'aw', 'hh'], 'V2': ['uw', 'uh', 'ow'], 'V3': ['ae', 'eh', 'ey', 'ay'], 'V4': ['ih', 'iy', 'ax'], 'A': ['l', 'el', 'r', 'y'], 'B': ['s', 'z'], 'C': ['t', 'd', 'n', 'en'], 'D': ['sh', 'zh', 'ch', 'jh'], 'E': ['p', 'b', 'm'], 'F': ['th', 'dh'], 'G': ['f', 'v'], 'H': ['ng', 'g', 'k', 'w'], 'S': ['sil', 'sp']}

visemes_neti = [
    'V1', 'V2', 'V3', 'V4', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'S'
]

# Jeffers and Barley Map (J. Jeffers and M. Barley. Speechreading (lipreading). Thomas, 1971)
# after Eoin Gillen: TCD-TIMIT: A New Database for Audio-Visual Speech Recognition [PhD thesis, University of Dublin]
phonemes_to_jeffersbarley = {
    'f' : 'A',          # Lip to Teeth
    'v' : 'A',
    'er' : 'B',         # Lips Puckered
    'ow' : 'B',
    'r' : 'B',
    'q' : 'B',
    'w' : 'B',
    'uh' : 'B',
    'uw' : 'B',
    'axr' : 'B',
    'ux' : 'B',
    'b' : 'C',          # Lips Together
    'p' : 'C',
    'm' : 'C',
    'em' : 'C',
    'aw' : 'D',         # Lips Relaxed - Moderate Opening to Lips Puckered-Narrow
    'dh' : 'E',         # Tongue Between Teeth
    'th' : 'E',
    'ch' : 'F',         # Lips Forward
    'jh' : 'F',
    'sh' : 'F',
    'zh' : 'F',
    'oy' : 'G',         # Lips Rounded
    'ao' : 'G',
    's' : 'H',          # Teeth Approximated
    'z' : 'H',
    'aa' : 'I',         # Lips Relaxed Narrow Opening
    'ae' : 'I',
    'ah' : 'I',
    'ay' : 'I',
    'ey' : 'I',
    'ih' : 'I',
    'iy' : 'I',
    'y' : 'I',
    'eh' : 'I',
    'ax-h' : 'I',
    'ax' : 'I',
    'ix' : 'I',
    'd' : 'J',          # Tongue Up or Down
    'l' : 'J',
    'n' : 'J',
    't' : 'J',
    'el' : 'J',
    'nx' : 'J',
    'en' : 'J',
    'dx' : 'J',
    'g' : 'K',          # Tongue Back
    'k' : 'K',
    'ng' : 'K',
    'eng' : 'K',
    'sil' : 'S',        # Silence
    'hh' : 'missing',
    'hv': 'missing'
}

jeffersbarley_to_phonemes = {'A': ['f', 'v'], 'B': ['er', 'ow', 'r', 'q', 'w', 'uh', 'uw', 'axr', 'ux'], 'C': ['b', 'p', 'm', 'em'], 'D': ['aw'], 'E': ['dh', 'th'], 'F': ['ch', 'jh', 'sh', 'zh'], 'G': ['oy', 'ao'], 'H': ['s', 'z'], 'I': ['aa', 'ae', 'ah', 'ay', 'ey', 'ih', 'iy', 'y', 'eh', 'ax-h', 'ax', 'ix'], 'J': ['d', 'l', 'n', 't', 'el', 'nx', 'en', 'dx'], 'K': ['g', 'k', 'ng', 'eng'], 'S': ['sil'], 'missing': ['hh', 'hv']}

visemes_jeffersbarley = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'S'
]

# Lee Map (Lee and Yook (2002): Audio-to-Visual Conversion Using Hidden Markov Models. In Proceedings of the 7th Pacific Rim International Conference on Artificial Intelligence: Trends in Artifical Intelligence. Springer-Verlag)
# after Cappalletta and Harte: Phoneme-to-Viseme Mapping for Visual Speech Recognition (http://www.mee.tcd.ie/~sigmedia/pmwiki/uploads/Publications.Bibtex2012/cappellettaICPRAM2012.pdf)
phonemes_to_lee = {
    'b' : 'P',
    'p' : 'P',
    'm' : 'P',
    'd' : 'T',
    't' : 'T',
    's' : 'T',
    'z' : 'T',
    'th' : 'T',
    'dh' : 'T',
    'g' : 'K',
    'k' : 'K',
    'n' : 'K',
    'ng' : 'K',
    'l' : 'K',
    'y' : 'K',
    'hh' : 'K',
    'jh' : 'CH',
    'ch' : 'CH',
    'sh' : 'CH',
    'zh' : 'CH',
    'f' : 'F',
    'v' : 'F',
    'r' : 'W',
    'w' : 'W',
    'iy' : 'IY',
    'ih' : 'IY',
    'eh' : 'EH',
    'ey' : 'EH',
    'ae' : 'EH',
    'aa' : 'AA',
    'aw' : 'AA',
    'ay' : 'AA',
    'ah' : 'AH',
    'ao' : 'AO',
    'oy' : 'AO',
    'ow' : 'AO',
    'uh' : 'UH',
    'uw' : 'UH',
    'er' : 'ER',
    'sil' : 'S'
}

lee_to_phonemes = {'P': ['b', 'p', 'm'], 'T': ['d', 't', 's', 'z', 'th', 'dh'], 'K': ['g', 'k', 'n', 'ng', 'l', 'y', 'hh'], 'CH': ['jh', 'ch', 'sh', 'zh'], 'F': ['f', 'v'], 'W': ['r', 'w'], 'IY': ['iy', 'ih'], 'EH': ['eh', 'ey', 'ae'], 'AA': ['aa', 'aw', 'ay'], 'AH': ['ah'], 'AO': ['ao', 'oy', 'ow'], 'UH': ['uh', 'uw'], 'ER': ['er'], 'S': ['sil']}

visemes_lee = [
    'P', 'T', 'K', 'CH', 'F', 'W', 'IY', 'EH', 'AA', 'AH', 'AO', 'UH', 'ER', 'S'
]

phonemes_to_woodwarddisney = {
    'b' : 'C1',
    'p' : 'C1',
    'm' : 'C1',
    'f' : 'C2',
    'v' : 'C2',
    'w' : 'C3',
    'r' : 'C3',
    't' : 'C4',
    'd' : 'C4',
    'n' : 'C4',
    'l' : 'C4',
    'th' : 'C4',
    'dh' : 'C4',
    's' : 'C4',
    'z' : 'C4',
    'ch' : 'C4',
    'jh' : 'C4',
    'sh' : 'C4',
    'zh' : 'C4',
    'j' : 'C4',
    'k' : 'C4',
    'g' : 'C4',
    'h' : 'C4',
    'uh' : 'V1',
    '' : 'V2',
    'iy' : 'V2',
    'ay' : 'V2',
    'eh' : 'V2',
    'ah' : 'V2',
    'uw' : 'V3',
    '' : 'V4',
    'aw' : 'V4',
    '' : 'V4',
}

woodwarddisney_to_phonemes = {'C1': ['b', 'p', 'm'], 'C2': ['f', 'v'], 'C3': ['w', 'r'], 'C4': ['t', 'd', 'n', 'l', 'th', 'dh', 's', 'z', 'ch', 'jh', 'sh', 'zh', 'j', 'k', 'g', 'h'], 'V1': ['uh'], 'V4': ['', 'aw'], 'V2': ['iy', 'ay', 'eh', 'ah'], 'V3': ['uw']}

# Standard reduced 39 phoneme set
phonemes = [
    'sil', 'aa', 'ae', 'ah', 'aw', 'ay', 'b', 'ch', 'd', 'dh',        'eh', 'er', 'ey', 'f', 'g', 'hh', 
    'ih', 'iy', 'jh', 'k', 'l', 'm', 'n', 'ng', 'ow', 'oy', 'p', 'r', 's', 'sh', 't', 'th', 'uh', 'uw', 
    'v', 'w', 'y', 'z'
]

# The 39 phoneme set without 'dx', since it is not present in TCD-TIMIT labels.
phonemes_38 = [
    'sil', 'aa', 'ae', 'ah', 'aw', 'ay', 'b', 'ch', 'd', 'dh',        'eh', 'er', 'ey', 'f', 'g', 'hh', 
    'ih', 'iy', 'jh', 'k', 'l', 'm', 'n', 'ng', 'ow', 'oy', 'p', 'r', 's', 'sh', 't', 'th', 'uh', 'uw', 
    'v', 'w', 'y', 'z'
]

phoneme_distribution_tcdtimit_train = {'sil': 0.4502456758943792, 'sh': 0.009931975644622297, 'iy': 0.027289030052446615, 'hh': 0.00691437139852198, 'ae': 0.016254019779999393, 'd': 0.019248304055109754, 'y': 0.005673748941856316, 'uh': 0.002070813423908101, 'r': 0.02623030336950261, 'aa': 0.02652646700107505, 'k': 0.022067688547795446, 's': 0.03914956729793828, 'uw': 0.008448825489661091, 't': 0.03230748780948516, 'ih': 0.019842963315353637, 'n': 0.0279862971850463, 'g': 0.0059349326169438246, 'w': 0.010183831331313824, 'er': 0.02042596259010254, 'l': 0.021207181618266065, 'ow': 0.011792909329620793, 'm': 0.014404746080495875, 'oy': 0.002956972321526432, 'ay': 0.014565653880326573, 'dh': 0.0045753783082293845, 'p': 0.01238990058696367, 'ah': 0.04002873020425963, 'eh': 0.01558706860968665, 'f': 0.012326936665290788, 'z': 0.02268799977612828, 'v': 0.006130820373259456, 'b': 0.008266929715939434, 'ey': 0.013931350669399767, 'ng': 0.005356597336392914, 'jh': 0.00540090528127383, 'th': 0.0017513298213457022, 'ch': 0.005396241287075839, 'aw': 0.004510082389457507}

# Old phoneme list without alphabetical ordering, used for training the first VASR models
phonemes_old = [
    'iy', 'ih', 'eh', 'ae', 'ah', 'uw', 'uh', 'aa', 'ey', 'ay', 'oy', 'aw', 'ow', 'l', 'r', 'y',
    'w', 'er', 'm', 'n', 'ng', 'ch', 'jh', 'dh', 'b', 'd', 'dx', 'g', 'p', 't', 'k', 'z', 'v',
    'f', 'th', 's', 'sh', 'hh', 'sil'
]