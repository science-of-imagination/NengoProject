from numpy import dot


def stimulus_through_encs(encs, stimulus):
    return dot(dot(encs, stimulus), encs)
