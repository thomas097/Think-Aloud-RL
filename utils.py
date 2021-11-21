from pprint import pprint
import random
import numpy as np


def capsule_for_query(capsule):
    """ Casefolds the triple in a capsule so that its entities all match regardless of case.

        params
        dict capsule: JSON containing the input utterance, extracted triples,
                      the perspective and contextual information

        returns:      a casefolded capsule
    """
    if capsule['subject']['label']:
        capsule['subject']['label'] = capsule['subject']['label'].lower()
    if capsule['predicate']['label']:
        capsule['subject']['label'] = capsule['subject']['label'].lower()
    if capsule['object']['label']:
        capsule['subject']['label'] = capsule['subject']['label'].lower()
    return capsule

def triple_for_capsule(triple):
    """ Adds the triple to a capsule.

        params
        dict triple:  JSON with subject-predicate-object triple with multiple types.

        returns:      triple modified to be appended to a capsule
    """
    subject_type = []
    object_type = []
    predicate_type = []

    if triple['subject']['type']:
        subject_type = triple['subject']['type'][0]
    if triple['predicate']['type']:
        predicate_type = triple['predicate']['type'][0]
    if triple['object']['type']:
        object_type = triple['object']['type'][0]

    capsule_triple = {
        "subject": {'label': triple['subject']['label'], 'type': subject_type},
        "predicate": {'label': triple['predicate']['label'], 'type': predicate_type},
        "object": {'label': triple['object']['label'], 'type': object_type},
    }
    return capsule_triple
