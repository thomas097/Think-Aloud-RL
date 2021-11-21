""" Filename:     Chatbot.py
    Author(s):    Thomas Bellucci
    Description:  Implementation of the RLChatbot based on Leolani.
    Date created: Nov. 11th, 2021
"""

from pathlib import Path
from datetime import date
from pprint import pprint
from random import choice

import sys
sys.path.insert(0, 'cltl-knowledgeextraction/src')
sys.path.insert(0, 'cltl-knowledgerepresentation/src')
sys.path.insert(0, 'cltl-languagegeneration/src')

import os
os.environ['JAVAHOME'] = "C:/Program Files/Java/jre1.8.0_311/bin/java.exe"

from cltl.triple_extraction.api import Chat, UtteranceHypothesis
from cltl.brain.long_term_memory import LongTermMemory
from cltl.reply_generation.thought_replier import RLReplier
from cltl.reply_generation.data.sentences import GREETING, TALK_TO_ME, SORRY, GOODBYE
from cltl.brain.utils.helper_functions import brain_response_to_json 
from cltl.combot.backend.api.discrete import UtteranceType

from utils import capsule_for_query, triple_for_capsule


class RLChatbot:
    def __init__(self, speaker):
        """ Sets up a Chatbot with a Leolani backend.

            params
            str speaker: name of speaker
            str address: address of the brain repository
        """
        # Set up Leolani modules
        self.__address = "http://localhost:7200/repositories/sandbox"
        self.__brain = LongTermMemory(address = self.__address,
                                      log_dir = Path("logs"),
                                      clear_all = False)
        self.__chat = Chat(speaker)
        self.__replier = RLReplier(self.__brain)
        
        # General book-keeping
        self.__brain_states = []
        self.__turns = 0

    @property
    def greet(self):
        """ Generates a random greeting
        """
        return choice(GREETING) + " " + choice(TALK_TO_ME)

    @property
    def farewell(self):
        """ Generates a random farewell message
        """
        return choice(GOODBYE)

    def show_thoughts(self):
        """ Shows the UCB plots of the thought-type and thought-
            instance selectors using pyplot.

            returns: None
        """
        self.__replier.thought_types.plot()
        self.__replier.thought_instances.plot()
    
    def __reward_replier__(self):
        """ Rewards the last thought phrased by the replier by updating
            its utility estimate with the relative improvement of the brain
            as a result of the user response (i.e. a reward).

            returns: None
        """
        # Define the state of the brain as a function of claims and entities
        state = float(self.__brain.count_statements())
        self.__brain_states.append(state)
        print("\nBRAIN STATE %s" % state)

        # If _input was a thought reply, update with a reward.
        if len(self.__brain_states) > 1:
            state_new = self.__brain_states[-1]
            state_old = self.__brain_states[-2]
            self.__replier.reward_thought(state_new - state_old)

    def __parse__(self, _input):
        """ Takes an input utterance from the user and returns a capsule
            with which to store and/or query the Brain.

            params
            str _input:  input of the user

            returns:  capsule dict with utterance and context
        """
        self.__turns += 1

        # Input classification / triple extraction
        self.__chat.add_utterance([UtteranceHypothesis(_input, 1.0)])
        self.__chat.last_utterance.analyze()
        utt = self.__chat.last_utterance
        
        capsule = {"chat": 1,
                   "turn": self.__turns,
                   "author": self.__chat.speaker,
                   "utterance": utt.transcript,
                   "utterance_type": utt.type,
                   "position": '{}-{}'.format(0, len(utt.transcript) - 1),
                   "context_id": 247,
                   "date": date.today(),
                   "place": "Piek's office",
                   "place_id": 106,
                   "country": "Netherlands",
                   "region": "North Holland",
                   "city": "Amsterdam",
                   "objects": [],
                   "people": []}

        if utt.triple is None: # Parsing failure
            return None
        
        capsule.update(triple_for_capsule(utt.triple))
            
        if utt.perspective is not None:
            capsule['perspective'] = utt.perspective

        print("\nCAPSULE")
        pprint(capsule)
        print()
        return capsule
            
    def respond(self, _input):
        """ Takes an input from the user, extracts triples (using __parse__) from it,
            updates/queries the brain and replies with a phrased thought (in case
            of the input being a statement) or an answer (in case of a question).

            params:
            str _input: input utterance of the user, e.g. a response to a Thought
                        or a question posed by the user.

            returns: a reply to the input
        """
        capsule = self.__parse__(_input)

        # ERROR
        if capsule is None:
            reply = choice(SORRY) + " I could not parse that. Can you rephrase?"

        # QUESTION
        elif capsule['utterance_type'] == UtteranceType.QUESTION:
            # Query Brain -> try to answer
            brain_response = self.__brain.query_brain(capsule_for_query(capsule))
            brain_response = brain_response_to_json(brain_response)
            
            self.__reward_replier__()
            reply = self.__replier.reply_to_question(brain_response)

        # STATEMENT
        elif capsule['utterance_type'] == UtteranceType.STATEMENT:
            # Update Brain -> communicate a thought
            brain_response = self.__brain.update(capsule, reason_types=True, create_label=True)
            brain_response = brain_response_to_json(brain_response)
            
            self.__reward_replier__()
            reply = self.__replier.reply_to_statement(brain_response)
            
        return reply

    
