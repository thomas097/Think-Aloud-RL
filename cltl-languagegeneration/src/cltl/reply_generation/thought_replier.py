""" Filename:     RL_replier.py
    Author(s):    Thomas Bellucci
    Description:  Modified LenkaReplier for use in the RLChatbot. The replier
                  takes out the Thought selection step and only selects among
                  Thoughts of the same type, using WordNet to select among the
                  individual instances of Thoughts.
    Date created: Nov. 11th, 2021
"""

from pattern.text.en import singularize
import random
from itertools import permutations
from pprint import pprint
import numpy as np

from cltl.combot.backend.utils.casefolding import casefold_text, casefold_capsule
from cltl.reply_generation.api import BasicReplier
from cltl.reply_generation.data.sentences import NEW_KNOWLEDGE, EXISTING_KNOWLEDGE, \
     CONFLICTING_KNOWLEDGE, CURIOSITY, HAPPY, TRUST, NO_TRUST, NO_ANSWER
from cltl.reply_generation.utils.helper_functions import lexicon_lookup, thought_types_from_brain

from cltl.reply_generation.RL import UCB

    
class RLReplier(BasicReplier):
    def __init__(self, brain):
        """ Creates an RL-based replier used to respond to questions and statements
            by the user. Statements are replied to by phrasing a thought. Selectors
            for the thought type and instance are learnt using UCB defined in RL.py.

            params
            object brain: the brain of Leolani

            returns: None
        """
        super(RLReplier, self).__init__()
        self.__brain = brain
        self.__thought_type_selector = UCB()
        self.__thought_inst_selector = UCB()
        self.__last_thought = None

    @property
    def thought_types(self):
        return self.__thought_type_selector

    @property
    def thought_instances(self):
        return self.__thought_inst_selector

    def reward_thought(self, reward):
        """ Updates the thought type and thought instance selectors by rewarding
            the last of its selected thoughts using a numeric reward.

            params:
            float reward: reward obtained after thought

            returns: None
        """
        if self.__last_thought is not None:
            # THOUGHT TYPE
            self.__thought_type_selector.update_utility(self.__last_thought[0], reward)
            print("\nREWARD %s WITH %s" % (self.__last_thought[0], reward))

            # THOUGHT INSTANCE
            self.__thought_inst_selector.update_utility(self.__last_thought[1], reward)
            print("\nREWARD %s WITH %s" % (self.__last_thought[1], reward))

    def reply_to_statement(self, brain_response):
        """ Selects a Thought from the brain response. First, a high-level thought-type
            is selected using the thought-type selector. Individual thoughts (at the
            instance-level) are then selected within the individual _phrase routines.
            
            params
            dict brain_response: brain response from brain.update() converted to JSON

            returns: a string representing a verbalized thought
        """
        # Select thought-type from available types
        thought_types = thought_types_from_brain(brain_response)
        thought_type = self.__thought_type_selector.select_action(thought_types)
        print("\nTHOUGHT %s" % thought_type)

        # Preprocess statement (triples) and thoughts
        utterance = casefold_capsule(brain_response['statement'], format='natural')
        thoughts = casefold_capsule(brain_response['thoughts'], format='natural')

        # Select reply given selected thought-type
        say = None
        if thought_type == 'entity_similarity': # NEW
            thought_inst, say = self._phrase_entity_similarity(thoughts['_entity_similarity'], utterance)
            
        if thought_type == 'complement_conflict':
            thought_inst, say = self._phrase_cardinality_conflicts(thoughts['_complement_conflict'], utterance)

        elif thought_type == 'negation_conflicts':
            thought_inst, say = self._phrase_negation_conflicts(thoughts['_negation_conflicts'], utterance)

        elif thought_type == 'statement_novelty':
            thought_inst, say = self._phrase_statement_novelty(thoughts['_statement_novelty'], utterance)

        elif thought_type == 'entity_novelty':
            thought_inst, say = self._phrase_entity_novelty(thoughts['_entity_novelty'], utterance)

        elif thought_type == 'object_gaps':
            thought_inst, say = self._phrase_complement_gaps(thoughts['_complement_gaps'], utterance)

        elif thought_type == 'subject_gaps':
            thought_inst, say = self._phrase_subject_gaps(thoughts['_subject_gaps'], utterance)

        elif thought_type == 'overlaps':
            thought_inst, say = self._phrase_overlaps(thoughts['_overlaps'], utterance)

        elif thought_type == 'trust':
            thought_inst, say = self._phrase_trust(thoughts['_trust'])

        # Fallback strategy
        if say is None:
            thought_inst, say = self._phrase_fallback()

        # Update the last thought of the replier
        self.__last_thought = (thought_type, thought_inst)

        say = say.replace('-', ' ').replace('  ', ' ')
        return say

    def _phrase_entity_similarity(self, novelty, utterance):
        """ Phrases an entity similarity thought asking whether some entity is
            similar than thr one mentioned.

            params:
            dict novelty:    thought
            dict utterance:  an utterance

            returns: (thought instance, phrase)
        """
        # Get all known entities in the brain and their types
        entities = self.__brain.get_labels_and_classes()

        # If new subject, compare subject; object otherwise
        if novelty["_subject"] == 'true':
            subject = utterance['triple']['_subject']['_label']
            types = utterance['triple']['_subject']['_types']    
        else:
            subject = utterance['triple']['_complement']['_label']
            types = utterance['triple']['_complement']['_types']

        # Limit entities to the same type and discard subject
        entities = [ent for ent, type_ in entities.items() if type_ in types and ent != subject]
        if entities == []:
            entities = ['object'] # fallback

        # Select new object
        object_ = thought = self.__thought_inst_selector.select_action(entities)

        # Phrase thought
        say = "Is a %s similar to a %s?" % (singularize(subject), singularize(object_))
        return thought, say

    def _phrase_cardinality_conflicts(self, conflicts, utterance):
        """ Phrases a cardinality conflict thought.

            params:
            dict conflicts: thoughts 
            dict utterance: an utterance

            returns: (thought instance, phrase)
        """
        say = random.choice(CONFLICTING_KNOWLEDGE)
        conflict = random.choice(conflicts)
        x = 'you' if conflict['_provenance']['_author'] == utterance['author'] \
            else conflict['_provenance']['_author']
        y = 'you' if utterance['triple']['_subject']['_label'] == conflict['_provenance']['_author'] \
            else utterance['triple']['_subject']['_label']

        # Checked
        say += ' %s told me in %s that %s %s %s, but now you tell me that %s %s %s' \
               % (x, conflict['_provenance']['_date'], y, utterance['triple']['_predicate']['_label'],
                  conflict['_complement']['_label'],
                  y, utterance['triple']['_predicate']['_label'], utterance['triple']['_complement']['_label'])

        # Symbolic selection (no need to select here)
        thought = self.__thought_inst_selector.select_action(['_'])
        return thought, say

    def _phrase_negation_conflicts(self, conflicts, utterance):
        """ Phrases a negation conflict thought.

            params:
            dict conflicts: thoughts 
            dict utterance: an utterance
            
            returns: (thought instance, phrase)
        """
        # Separate positive and negative polarities
        affirmative_conflict = [item for item in conflicts if item['_polarity_value'] == 'POSITIVE']
        negative_conflict = [item for item in conflicts if item['_polarity_value'] == 'NEGATIVE']

        # There is a conflict, so we phrase it
        if affirmative_conflict and negative_conflict:
            say = random.choice(CONFLICTING_KNOWLEDGE)

            affirmative_conflict = random.choice(affirmative_conflict)
            negative_conflict = random.choice(negative_conflict)

            say += ' %s told me in %s that %s %s %s, but in %s %s told me that %s did not %s %s' \
                   % (affirmative_conflict['_provenance']['_author'], affirmative_conflict['_provenance']['_date'],
                      utterance['triple']['_subject']['_label'], utterance['triple']['_predicate']['_label'],
                      utterance['triple']['_complement']['_label'],
                      negative_conflict['_provenance']['_date'], negative_conflict['_provenance']['_author'],
                      utterance['triple']['_subject']['_label'], utterance['triple']['_predicate']['_label'],
                      utterance['triple']['_complement']['_label'])

        # Symbolic selection (no need to select here)
        thought = self.__thought_inst_selector.select_action(['_'])        
        return thought, say

    def _phrase_statement_novelty(self, novelties, utterance):
        """ Phrases a statement novelty thought.

            params:
            dict novelties: thoughts 
            dict utterance: an utterance
            
            returns: (thought instance, phrase)
        """        
        if not novelties:
            entity_role = random.choice(['subject', 'object'])

            say = random.choice(NEW_KNOWLEDGE)

            if entity_role == 'subject':
                if 'person' in ' or '.join(utterance['triple']['_complement']['_types']):
                    any_type = 'anybody'
                elif 'location' in ' or '.join(utterance['triple']['_complement']['_types']):
                    any_type = 'anywhere'
                else:
                    any_type = 'anything'

                # Checked
                say += ' I did not know %s that %s %s' % (any_type, utterance['triple']['_subject']['_label'],
                                                          utterance['triple']['_predicate']['_label'])

            elif entity_role == 'object':
                # Checked
                say += ' I did not know anybody who %s %s' % (utterance['triple']['_predicate']['_label'],
                                                              utterance['triple']['_complement']['_label'])

        # I already knew this
        else:
            say = random.choice(EXISTING_KNOWLEDGE)
            novelty = random.choice(novelties)

            # Checked
            say += ' %s told me about it in %s' % (novelty['_provenance']['_author'],
                                                   novelty['_provenance']['_date'])

        # Symbolic selection (no need to select here)
        thought = self.__thought_inst_selector.select_action(['_'])
        return thought, say

    def _phrase_entity_novelty(self, novelties, utterance):
        """ Phrases an entity novelty thought.

            params:
            dict novelties: thoughts 
            dict utterance: an utterance
            
            returns: (thought instance, phrase)
        """   
        entity_role = random.choice(['subject', 'object'])
        entity_label = utterance['triple']['_subject']['_label'] \
            if entity_role == 'subject' else utterance['triple']['_complement']['_label']
        novelty = novelties['_subject'] if entity_role == 'subject' else novelties['_complement']

        if novelty:
            entity_label = self._replace_pronouns(utterance['author'], entity_label=entity_label,
                                                  role=entity_role)
            say = random.choice(NEW_KNOWLEDGE)
            if entity_label != 'you':  # TODO or type person?
                # Checked
                say += ' I had never heard about %s before!' % self._replace_pronouns(utterance['author'],
                                                                                      entity_label=entity_label,
                                                                                      role='object')
            else:
                say += ' I am excited to get to know about %s!' % entity_label

        else:
            say = random.choice(EXISTING_KNOWLEDGE)
            if entity_label != 'you':
                # Checked
                say += ' I have heard about %s before' % self._replace_pronouns(utterance['author'],
                                                                                entity_label=entity_label,
                                                                                role='object')
            else:
                say += ' I love learning more and more about %s!' % entity_label

        # Symbolic selection (no need to select here)
        thought = self.__thought_inst_selector.select_action(['_'])        
        return thought, say

    def _phrase_subject_gaps(self, all_gaps, utterance):
        """ Phrases a subject gap thought.

            params:
            dict all_gaps:  thoughts 
            dict utterance: an utterance
            
            returns: (thought instance, phrase)
        """
        # Define function to construct a description from overlaps
        def describe(gap):
            return gap['_predicate']['_label'] + '_' + gap['_entity']['_types'][0]

        # Select object or subject to pursue
        entity_role = random.choice(['subject', 'object'])
        gaps = all_gaps['_subject'] if entity_role == 'subject' else all_gaps['_complement']

        thought = None
        say = random.choice(CURIOSITY)
        
        if entity_role == 'subject':

            if not gaps:               
                say += ' What types can %s %s' % (utterance['triple']['_subject']['_label'],
                                                  utterance['triple']['_predicate']['_label'])
            else:
                # Select gap to phrase
                gap_names = {describe(gap):gap for gap in gaps}
                thought = self.__thought_inst_selector.select_action(gap_names.keys())
                gap = gap_names[thought]
                
                if 'is ' in gap['_predicate']['_label'] or ' is' in gap['_predicate']['_label']:
                    say += ' Is there a %s that %s %s?' % (' or'.join(gap['_entity']['_types']),
                                                           gap['_predicate']['_label'],
                                                           utterance['triple']['_subject']['_label'])
                elif ' of' in gap['_predicate']['_label']:
                    say += ' Is there a %s that %s is %s?' % (' or'.join(gap['_entity']['_types']),
                                                              utterance['triple']['_subject']['_label'],
                                                              gap['_predicate']['_label'])

                elif ' ' in gap['_predicate']['_label']:
                    say += ' Is there a %s that is %s %s?' % (' or'.join(gap['_entity']['_types']),
                                                              gap['_predicate']['_label'],
                                                              utterance['triple']['_subject']['_label'])
                else:
                    # Checked
                    say += ' Has %s %s %s?' % (utterance['triple']['_subject']['_label'],
                                               gap['_predicate']['_label'],
                                               ' or'.join(gap['_entity']['_types']))
        else:
            if not gaps:
                say += ' What kinds of things can %s a %s like %s' % (utterance['triple']['_predicate']['_label'],
                                                                      utterance['triple']['_complement']['_label'],
                                                                      utterance['triple']['_subject']['_label'])
            else:
                # Select gap to phrase
                gap_names = {describe(gap):gap for gap in gaps}
                thought = self.__thought_inst_selector.select_action(gap_names.keys())
                gap = gap_names[thought]
                
                if '#' in ' or'.join(gap['_entity']['_types']):
                    say += ' What is %s %s?' % (utterance['triple']['_subject']['_label'],
                                                gap['_predicate']['_label'])
                elif ' ' in gap['_predicate']['_label']:
                    # Checked
                    say += ' Has %s ever %s %s?' % (' or'.join(gap['_entity']['_types']),
                                                    gap['_predicate']['_label'],
                                                    utterance['triple']['_subject']['_label'])

                else:
                    # Checked
                    say += ' Has %s ever %s a %s?' % (utterance['triple']['_subject']['_label'],
                                                      gap['_predicate']['_label'],
                                                      ' or'.join(gap['_entity']['_types']))
        # Symbolic selection
        if thought is None:
            thought = self.__thought_inst_selector.select_action(['_'])
        return thought, say

    def _phrase_complement_gaps(self, all_gaps, utterance):
        """ Phrases a object gap (complement gap) thought.

            params:
            dict all_gaps:  thoughts 
            dict utterance: an utterance
            
            returns: (thought instance, phrase)
        """
        # Define function to construct a description from overlaps
        def describe(gap):
            return gap['_predicate']['_label'] + '_' + gap['_entity']['_types'][0]

        # Select object or subject to pursue
        entity_role = random.choice(['subject', 'object'])
        gaps = all_gaps['_subject'] if entity_role == 'subject' else all_gaps['_complement']

        thought = None
        say = random.choice(CURIOSITY)
        
        if entity_role == 'subject':

            if not gaps:
                say += ' What types can %s %s' % (utterance['triple']['_subject']['_label'],
                                                  utterance['triple']['_predicate']['_label'])
            else:
                # Select gap to phrase
                gap_names = {describe(gap):gap for gap in gaps}
                thought = self.__thought_inst_selector.select_action(gap_names.keys())
                gap = gap_names[thought]
                
                if ' in' in gap['_predicate']['_label']:  # ' by' in gap['_predicate']['_label']
                    say += ' Is there a %s %s %s?' % (' or'.join(gap['_entity']['_types']),
                                                      gap['_predicate']['_label'],
                                                      utterance['triple']['_complement']['_label'])
                else:
                    say += ' Has %s %s by a %s?' % (utterance['triple']['_complement']['_label'],
                                                    gap['_predicate']['_label'],
                                                    ' or'.join(gap['_entity']['_types']))
        else: # object
            if not gaps:
                otypes = ' or'.join(utterance['triple']['_complement']['_types']) \
                    if ' or'.join(utterance['triple']['_complement']['_types']) != '' \
                    else 'things'
                stypes = ' or'.join(utterance['triple']['_subject']['_types']) \
                    if ' or '.join(utterance['triple']['_subject']['_types']) != '' \
                    else 'actors'
                say += ' What types of %s like %s do %s usually %s' % (otypes,
                                                                       utterance['triple']['_complement']['_label'],
                                                                       stypes,
                                                                       utterance['triple']['_predicate']['_label'])
            else:
                # Select gap to phrase
                gap_names = {describe(gap):gap for gap in gaps}
                thought = self.__thought_inst_selector.select_action(gap_names.keys())
                gap = gap_names[thought]
                
                if '#' in ' or'.join(gap['_entity']['_types']):
                    say += ' What is %s %s?' % (utterance['triple']['_complement']['_label'],
                                                gap['_predicate']['_label'])
                elif ' by' in gap['_predicate']['_label']:
                    say += ' Has %s ever %s a %s?' % (utterance['triple']['_complement']['_label'],
                                                      gap['_predicate']['_label'],
                                                      ' or'.join(gap['_entity']['_types']))
                else:
                    say += ' Has a %s ever %s %s?' % (' or'.join(gap['_entity']['_types']),
                                                      gap['_predicate']['_label'],
                                                      utterance['triple']['_complement']['_label'])

        # Symbolic selection
        if thought is None:
            thought = self.__thought_inst_selector.select_action(['_'])
        return thought, say

    def _phrase_overlaps(self, all_overlaps, entity_role, utterance):
        """ Phrases the overlap thought.

            params:
            dict all_overlaps: thoughts
            str entity_role:   subject or object
            dict utterance:    an utterance
            
            returns: (thought instance, phrase)
        """
        # Define function to construct a description from overlaps
        def describe(tup):
            return "_".join([x['_entity']['_label'] for x in tup])
        
        # Select object or subject whichever is available
        if all_overlaps['_subject']:
            entity_role == 'subject'
            overlaps = all_overlaps['_subject']
        else:
            entity_role == 'object'
            overlaps = all_overlaps['_complement']

        # Phrase overlaps
        say = random.choice(HAPPY)
        if len(overlaps) < 2:
            # Phrase overlap
            overlaps = {describe([overlap]):overlap for overlap in overlaps}
            thought = self.__thought_inst_selector.select_action(overlaps.keys())
            overlap = overlaps[thought]

            if entity_role == 'subject':
                say += ' Did you know that %s also %s %s' % (utterance['triple']['_subject']['_label'],
                                                             utterance['triple']['_predicate']['_label'],
                                                             overlap['_entity']['_label'])
            else:
                say += ' Did you know that %s also %s %s' % (overlap['_entity']['_label'],
                                                             utterance['triple']['_predicate']['_label'],
                                                             utterance['triple']['_complement']['_label'])
        else:
            # Phrase a pair of overlaps
            overlaps = {describe([o1, o2]):(o1, o2) for o1, o2 in permutations(overlaps, r=2)}
            thought = self.__thought_inst_selector.select_action(overlaps.keys())
            sample = overlaps[thought]

            if entity_role == 'subject':
                say += ' Now I know %s items that %s %s, like %s and %s' % (len(overlaps),
                                                                            utterance['triple']['_subject']['_label'],
                                                                            utterance['triple']['_predicate']['_label'],
                                                                            sample[0]['_entity']['_label'],
                                                                            sample[1]['_entity']['_label'])
            else:
                types = ' or '.join(sample[0]['_entity']['_types']) if sample[0]['_entity']['_types'] else 'things'
                say += ' Now I know %s %s that %s %s, like %s and %s' % (len(overlaps), types,
                                                                         utterance['triple']['_predicate']['_label'],
                                                                         utterance['triple']['_complement']['_label'],
                                                                         sample[0]['_entity']['_label'],
                                                                         sample[1]['_entity']['_label'])
        return thought, say

    def _phrase_trust(self, trust):
        """ Phrases the trust of the robot w.r.t. the speaker.

            params:
            str trust: trust value for the speaker
            
            returns: (thought instance, phrase)
        """
        if float(trust) > 0.75: # bugfix
            say = random.choice(TRUST)
        else:
            say = random.choice(NO_TRUST)

        # Symbolic selection as _phrase_trust is deterministic
        thought = self.__thought_inst_selector.select_action(['_'])
        return thought, say

    def _phrase_fallback(self):
        """ Phrases a fallback utterance when an error has ocurred or no
            thoughts were generated.

            returns: (thought instance, phrase)
        """
        say = "I do not know what to say."
        thought = "no_choice"
        return thought, say



    ## Did not touch!
    def reply_to_question(self, brain_response):
        say = ''
        previous_author = ''
        previous_predicate = ''
        gram_person = ''
        gram_number = ''

        utterance = brain_response['question']
        response = brain_response['response']

        # TODO revise (we conjugate the predicate by doing this)
        utterance = casefold_capsule(utterance, format='natural')

        if not response:
            subject_types = ' or '.join(utterance['subject']['type']) \
                if utterance['subject']['type'] is not None else ''
            object_types = ' or '.join(utterance['object']['type']) \
                if utterance['object']['type'] is not None else ''

            if subject_types and object_types and utterance['predicate']['type']:
                say += "I know %s usually %s %s, but I do not know for sure" % (
                    utterance['subject']['label'],
                    utterance['predicate']['label'],
                    utterance['object']['label'])
                return say

            else:
                return random.choice(NO_ANSWER)

        # Each triple is hashed, so we can figure out when we are about the say things double
        handled_items = set()
        response.sort(key=lambda x: x['authorlabel']['value'])

        for item in response:

            # INITIALIZATION
            subject, predicate, object = self._assign_spo(utterance, item)
            predicate = utterance['predicate']['label'] # bugfix

            author = self._replace_pronouns(utterance['author'], author=item['authorlabel']['value'])
            subject = self._replace_pronouns(utterance['author'], entity_label=subject, role='subject')
            object = self._replace_pronouns(utterance['author'], entity_label=object, role='object')

            subject = self._fix_entity(subject, utterance['author'])
            object = self._fix_entity(object, utterance['author'])

            # Hash item such that duplicate entries have the same hash
            item_hash = '{}_{}_{}_{}'.format(subject, predicate, object, author)

            # If this hash is already in handled items -> skip this item and move to the next one
            if item_hash in handled_items:
                continue
            # Otherwise, add this item to the handled items (and handle item the usual way (with the code below))
            else:
                handled_items.add(item_hash)

            # Get grammatical properties
            subject_entry = lexicon_lookup(subject.lower())
            if subject_entry and 'person' in subject_entry:
                gram_person = subject_entry['person']
            if subject_entry and 'number' in subject_entry:
                gram_number = subject_entry['number']

            # Deal with author
            say, previous_author = self._deal_with_authors(author, previous_author, predicate, previous_predicate, say)

            if predicate.endswith('is'):

                say += object + ' is'
                if utterance['object']['label'].lower() == utterance['author'].lower() or \
                        utterance['subject']['label'].lower() == utterance['author'].lower():
                    say += ' your '
                elif utterance['object']['label'].lower() == 'leolani' or \
                        utterance['subject']['label'].lower() == 'leolani':
                    say += ' my '
                say += predicate[:-3]

                return say

            else:  # TODO fix_predicate_morphology
                be = {'first': 'am', 'second': 'are', 'third': 'is'}
                if predicate == 'be':  # or third person singular
                    if gram_number:
                        if gram_number == 'singular':
                            predicate = be[gram_person]
                        else:
                            predicate = 'are'
                    else:
                        # TODO: Is this a good default when 'number' is unknown?
                        predicate = 'is'
                elif gram_person == 'third' and '-' not in predicate:
                    predicate += 's'

                if item['certaintyValue']['value'] != 'CERTAIN':  # TODO extract correct certainty marker
                    predicate = 'maybe ' + predicate

                if item['polarityValue']['value'] != 'POSITIVE':
                    if ' ' in predicate:
                        predicate = predicate.split()[0] + ' not ' + predicate.split()[1]
                    else:
                        predicate = 'do not ' + predicate

                say += subject + ' ' + predicate + ' ' + object

            say += ' and '
        say = say[:-5]

        return say.replace('-', ' ').replace('  ', ' ')
    

    @staticmethod
    def _assign_spo(utterance, item):
        empty = ['', 'unknown', 'none']

        # INITIALIZATION
        predicate = utterance['predicate']['type']

        if utterance['subject']['label'] is None or utterance['subject']['label'].lower() in empty:
            subject = item['slabel']['value']
        else:
            subject = utterance['subject']['label']

        if utterance['object']['label'] is None or utterance['object']['label'].lower() in empty:
            object = item['olabel']['value']
        else:
            object = utterance['object']['label']

        return subject, predicate, object

    @staticmethod
    def _deal_with_authors(author, previous_author, predicate, previous_predicate, say):
        # Deal with author
        if author != previous_author:
            say += author + ' told me '
            previous_author = author
        else:
            if predicate != previous_predicate:
                say += ' that '

        return say, previous_author

    def _fix_entity(self, entity, speaker):
        new_ent = ''
        if '-' in entity:
            entity_tokens = entity.split('-')

            for word in entity_tokens:
                new_ent += self._replace_pronouns(speaker, entity_label=word, role='pos') + ' '

        else:
            new_ent += self._replace_pronouns(speaker, entity_label=entity)

        entity = new_ent
        return entity

    @staticmethod
    def _replace_pronouns(speaker, author=None, entity_label=None, role=None):
        if entity_label is None and author is None:
            return speaker

        if role == 'pos':
            # print('pos', speaker, entity_label)
            if speaker.lower() == entity_label.lower():
                pronoun = 'your'
            elif entity_label.lower() == 'leolani':
                pronoun = 'my'
            else:
                pronoun = entity_label  # third person pos.
            return pronoun

        # Fix author
        elif author is not None and author.lower() not in ['', 'unknown', 'none']:
            if speaker.lower() == author.lower():
                pronoun = 'you'
            elif author.lower() == 'leolani':
                pronoun = 'I'
            else:
                pronoun = author.title()

            return pronoun

        # Entity
        if entity_label is not None and entity_label.lower() not in ['', 'unknown', 'none']:
            if speaker.lower() in [entity_label.lower(), 'speaker'] or entity_label == 'Speaker':
                pronoun = 'you'
            elif entity_label.lower() == 'leolani':
                pronoun = 'I'
            else:
                pronoun = entity_label

            return pronoun
