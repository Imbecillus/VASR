def remap_to_phn(posts, original, target, convert_to_hmm_states = False, perform_logsoftmax = True, perform_priorweighting = False):
    """
    Remaps the posterior probabilities in "posts" from the viseme set given in "original" to "target".
    "target" can be "39phn" (standard reduced 39 phoneme set) or "38phn" (39phn without /dx/)
    Optional: convert_to_hmm_states can be set to true in order to get post. prob.s for HMM states instead.
    """

    from viseme_list import phonemes

    if original == 'jeffersbarley':
        from viseme_list import visemes_jeffersbarley as oldset
        from viseme_list import jeffersbarley_to_phonemes as mapping
    elif original == 'lee':
        from viseme_list import visemes_lee as oldset
        from viseme_list import lee_to_phonemes as mapping
    elif original == 'neti':
        from viseme_list import visemes_neti as oldset
        from viseme_list import neti_to_phonemes as mapping
    else:
        raise AssertionError(f'The given original map ({original}) is unknown. Currently supported are "jeffersbarley", "lee" and "neti".')

    if target == '39phn':
        from viseme_list import phonemes
    elif target == '38phn':
        from viseme_list import phonemes_38 as phonemes
    else:
        raise AssertionError(f'The given target map ({target}) is unknown. Currently supported are "39phn" and "38phn".')

    # Create a dictionary filled with the posterior probabilities for all phonemes
    # by iterating through viseme posterior probabilities
    phoneme_posteriors = {}
    for i in range(len(oldset)):
        for phoneme in mapping[oldset[i]]:
            phoneme_posteriors[phoneme] = posts[i]

    # Create vector from the dictionary; default to 0 if phoneme was not in the viseme map
    if not convert_to_hmm_states:
        new_posts = [None] * len(phonemes)
        for i in range(len(phonemes)):
            new_posts[i] = phoneme_posteriors.get(phonemes[i], 0)
    else:
        new_posts = [None] * len(phonemes) * 3
        for i in range(len(phonemes)):
            new_posts[i*3] = phoneme_posteriors.get(phonemes[i], 0)
            new_posts[i*3+1] = phoneme_posteriors.get(phonemes[i], 0)
            new_posts[i*3+2] = phoneme_posteriors.get(phonemes[i], 0)
    
    # Perform weighting with priors
    # NOTE: This currently uses priors derived from the TCD-TIMIT train set! No support for other databases is currently implemented!
    if perform_priorweighting:
        from viseme_list import phoneme_distribution_tcdtimit_train as priors

        for i in range(len(phonemes)):
            prior = priors[phonemes[i]]
            new_posts[i] = new_posts[i] * prior

    # Perform LogSoftmax; in order to use PyTorch's implementation, we need to convert the list to a tensor and then back afterwards
    if perform_logsoftmax:
        from torch import tensor
        from torch.nn.functional import log_softmax

        new_posts = tensor(new_posts)
        new_posts = log_softmax(new_posts)
        new_posts = new_posts.tolist()

    return new_posts

# Function for remapping old predictions to Turbo map (alphabetic)
def reorder_38phn(oldvector):
    LUT = {'iy': 0, 'ih': 1, 'eh': 2, 'ae': 3, 'ah': 4, 'uw': 5, 'uh': 6, 'aa': 7, 'ey': 8, 'ay': 9, 'oy': 10, 'aw': 11, 'ow': 12, 'l': 13, 'r': 14, 'y': 15, 'w': 16, 'er': 17, 'm': 18, 'n': 19, 'ng': 20, 'ch': 21, 'jh': 22, 'dh': 23, 'b': 24, 'd': 25, 'dx': 26, 'g': 27, 'p': 28, 't': 29, 'k': 30, 'z': 31, 'v': 32, 'f': 33, 'th': 34, 's': 35, 'sh': 36, 'hh': 37, 'sil': 38}
    from viseme_list import phonemes
    from viseme_list import phonemes_old

    newvector = [None] * 38
    for i in range(38):
        phoneme = phonemes[i]
        if phoneme is not 'dx':
            ix_old = LUT[phoneme]
            newvector[i] = oldvector[ix_old]

    return newvector