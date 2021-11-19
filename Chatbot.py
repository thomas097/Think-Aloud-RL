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
from cltl.reply_generation.thought_replier import ThoughtReplier
from cltl.reply_generation.data.sentences import GREETING, TALK_TO_ME, SORRY, GOODBYE
from cltl.brain.utils.helper_functions import brain_response_to_json 
from cltl.combot.backend.api.discrete import UtteranceType

from utils import thought_types_from_brain, capsule_for_query, triple_for_capsule
from RL import UCB1



class RLChatbot:
    def __init__(self, speaker):
        """ Sets up the Chatbot with a Leolani backend.

            params
            str speaker: name of speaker
        """
        # Set up Leolani modules
        self.__turns = 0
        self.__chat = Chat(speaker)
        self.__replier = ThoughtReplier()
        self.__selector = UCB1()

        self.__address = "http://localhost:7200/repositories/sandbox"
        self.__brain = LongTermMemory(address = self.__address,
                                      log_dir = Path("logs"),
                                      clear_all = False)
        # Book-keeping for RL agent
        self.__thought = None
        self.__brain_states = []

    @property
    def thought_selector(self):
        return self.__selector

    @property
    def greeting(self):
        return choice(GREETING) + " " + choice(TALK_TO_ME)

    @property
    def goodbye(self):
        return choice(GOODBYE)
    
    
    def __update_thought_utility__(self):
        """ Updates the utility of the last Thought of the robot given the
            observed improvement of the brain as a result of the user's response.
            Improvement is defined here as the relative difference between the
            last two brain states; i.e. R_t = S(brain_{t}) - S(brain_{t-1})

            returns: None
        """
        # Define the state of the brain as a function of claims and entities
        state = float(self.__brain.count_statements())
        self.__brain_states.append(state)
        print("\nBRAIN STATE %s" % state)

        # If _input was a reply to the last Thought, update the utility
        # of the Thought by the improvement in Brain size.
        if self.__thought is not None:    
            util_new = self.__brain_states[-1]
            util_old = self.__brain_states[-2]
            reward = util_new - util_old
            
            self.__selector.update_utility(self.__thought, reward)
            print("\nREWARD %s WITH %s" % (self.__thought, reward))

    def __parse_input__(self, _input):
        """ Takes an input utterance from the user and returns a capsule
            with which to store and/or query the Brain.

            params
            str _input: input of the user

            returns:    capsule dict with utterance and context
        """
        self.__turns += 1

        # -> input classification / analysis
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

        if utt.triple is not None:
             capsule.update(triple_for_capsule(utt.triple))
        else:
            return None # Signal failure
            
        if utt.perspective is not None:
            capsule['perspective'] = utt.perspective

        print("\nCAPSULE")
        pprint(capsule)
        print()
        
        return capsule
            
    def reply(self, _input):
        """ Takes an input from the user, extract triples (a capsule) from it,
            updates/queries the brain and replies with verbalized Thoughts/answers.

            params:
            str _input: input utterance of the user, e.g. a response to a Thought
                        or a question posed by the user.

            returns:    a reply, i.e. a Thought or an answer to the question.
        """
        capsule = self.__parse_input__(_input)

        # ERROR
        if capsule is None:
            self.__update_thought_utility__()
            reply = choice(SORRY) + " I could not parse that. Can you rephrase?"

        # QUESTION
        elif capsule['utterance_type'] == UtteranceType.QUESTION:
            # Query Brain -> communicate answer
            response = self.__brain.query_brain(capsule_for_query(capsule))
            response = brain_response_to_json(response)

            self.__update_thought_utility__()

            reply = self.__replier.reply_to_question(response)

        # STATEMENT
        elif capsule['utterance_type'] == UtteranceType.STATEMENT:
            # Update Brain -> communicate thoughts
            response = self.__brain.update(capsule, reason_types=True, create_label=True)
            response = brain_response_to_json(response)

            self.__update_thought_utility__()

            all_thoughts = thought_types_from_brain(response)
            self.__thought = self.__selector.select_action(all_thoughts)
            print("\nTHOUGHT %s" % self.__thought)

            reply = self.__replier.reply_to_statement(response, self.__thought)
            
        return reply

    
