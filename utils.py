from pprint import pprint
import random
import numpy as np


def thought_types_from_brain(brain_response):
    """ Takes a Brain response capsule and extracts the types of generated
        thoughts from it.

        params
        dict capsule: JSON containing the input utterance, extracted triples,
                      the perspective and contextual information

        returns:      a list of Thought types
    """
    t = brain_response['thoughts']

    thoughts = []
    if len(t['_overlaps']):
        overlaps = t['_overlaps']
        thoughts += ['overlaps'] if len(overlaps['_complement']) else []
        thoughts += ['overlaps'] if len(overlaps['_subject']) else []

    if len(t['_entity_novelty']):
        entity_novelty = t['_entity_novelty']
        thoughts += ['entity_novelty'] if entity_novelty['_complement'] else []  
        thoughts += ['entity_novelty'] if entity_novelty['_subject'] else []

    if len(t['_subject_gaps']):
        subject_gaps = t['_subject_gaps']
        thoughts += ['subject_gaps'] if len(subject_gaps['_complement']) else []           
        thoughts += ['subject_gaps'] if len(subject_gaps['_subject']) else []

    if len(t['_complement_gaps']):
        object_gaps = t['_complement_gaps']
        thoughts += ['object_gaps'] if len(object_gaps['_complement']) else []           
        thoughts += ['object_gaps'] if len(object_gaps['_subject']) else []

    if len(t['_statement_novelty']):
        thoughts += ['statement_novelty', 'statement_novelty']
    
    thoughts += ['complement_conflict'] if len(t['_complement_conflict']) else []
    thoughts += ['negation_conflicts'] if len(t['_negation_conflicts']) else []
    thoughts += ['trust']
    return set(thoughts)

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
