import json
import logging
import os
import sys
from collections import Counter
from datetime import datetime
from random import getrandbits
from typing import List, Optional

from nltk import CFG, RecursiveDescentParser, edit_distance
from nltk import pos_tag

from cltl.combot.backend.api.discrete import UtteranceType
from cltl.combot.backend.utils.casefolding import casefold_text
from cltl.triple_extraction.analyzer import Analyzer
from cltl.triple_extraction.data.base_cases import friends
from cltl.triple_extraction.ner import NER
from cltl.triple_extraction.pos import POS

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('\r%(asctime)s - %(levelname)8s - %(name)40s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


class UtteranceHypothesis(object):
    """
    Automatic Speech Recognition (ASR) Hypothesis

    Parameters
    ----------
    transcript: str
        Utterance Hypothesis Transcript
    confidence: float
        Utterance Hypothesis Confidence
    """

    def __init__(self, transcript, confidence):
        # type: (str, float) -> None

        self._transcript = transcript
        self._confidence = confidence

    @property
    def transcript(self):
        # type: () -> str
        """
        Automatic Speech Recognition Hypothesis Transcript

        Returns
        -------
        transcript: str
        """
        return self._transcript

    @transcript.setter
    def transcript(self, value):
        # type: (str) -> None
        self._transcript = value

    @property
    def confidence(self):
        # type: () -> float
        """
        Automatic Speech Recognition Hypothesis Confidence

        Returns
        -------
        confidence: float
        """
        return self._confidence

    @confidence.setter
    def confidence(self, value):
        # type: (float) -> None
        self._confidence = value

    def __repr__(self):
        return "<'{}' [{:3.2%}]>".format(self.transcript, self.confidence)


class Chat(object):
    def __init__(self, speaker):
        """
        Create Chat

        Parameters
        ----------
        speaker: str
            Name of speaker (a.k.a. the person Pepper has a chat with)
        """

        self._id = getrandbits(128)
        self._speaker = speaker
        self._utterances = []

        self._log = self._update_logger()
        self._log.info("<< Start of Chat with {} >>".format(speaker))

    @property
    def speaker(self):
        # type: () -> str
        """
        Returns
        -------
        speaker: str
            Name of speaker (a.k.a. the person Pepper has a chat with)
        """
        return self._speaker

    @speaker.setter
    def speaker(self, value):
        self._speaker = value

    @property
    def id(self):
        # type: () -> int
        """
        Returns
        -------
        id: int
            Unique (random) identifier of this chat
        """
        return self._id

    def set_id(self, value):
        self._id = value

    @property
    def utterances(self):
        # type: () -> List[Utterance]
        """
        Returns
        -------
        utterances: list of Utterance
            List of utterances that occurred in this chat
        """
        return self._utterances

    @property
    def last_utterance(self):
        # type: () -> Utterance
        """
        Returns
        -------
        last_utterance: Utterance
            Most recent Utterance
        """
        return self._utterances[-1]

    def add_utterance(self, hypotheses):
        # type: (List[UtteranceHypothesis]) -> Utterance
        """
        Add Utterance to Conversation

        Parameters
        ----------
        hypotheses: list of UtteranceHypothesis

        Returns
        -------
        utterance: Utterance
        """
        utterance = Utterance(self, hypotheses, len(self._utterances))
        self._utterances.append(utterance)

        self._log = self._update_logger()
        self._log.info(utterance)

        return utterance

    def _update_logger(self):
        return logger.getChild("Chat {:19s} {:03d}".format("({})".format(self.speaker), len(self._utterances)))

    def __repr__(self):
        return "\n".join([str(utterance) for utterance in self._utterances])


class Utterance(object):
    def __init__(self, chat, hypotheses, turn):
        # type: (Chat, List[UtteranceHypothesis], int) -> Utterance
        """
        Construct Utterance Object

        Parameters
        ----------
        chat: Chat
            Reference to Chat Utterance is part of
        hypotheses: List[UtteranceHypothesis]
            Hypotheses on uttered text (transcript, confidence)
        turn: int
            Utterance Turn
        """

        self._log = logger.getChild(self.__class__.__name__)

        self._datetime = datetime.now()
        self._chat = chat
        self._chat_speaker = self._chat.speaker
        self._turn = turn

        self._hypothesis = self._choose_hypothesis(hypotheses)
        self._tokens = self._clean(self._tokenize(self.transcript))

        # TODO: Optimize: takes 2.6 seconds now! Should be < 1 second!?
        self._parser = Parser(self)
        # TODO analyze sets triple, perspective and type, but currently is not called on constructor
        self._type = None
        self._triple = None
        self._perspective = None

    @property
    def chat(self):
        # type: () -> Chat
        """
        Returns
        -------
        chat: Chat
            Utterance Chat
        """
        return self._chat

    @property
    def chat_speaker(self):
        # type: () -> str
        """
        Returns
        -------
        speaker: str
            Name of speaker (a.k.a. the person Pepper has a chat with)
        """
        return self._chat_speaker

    @property
    def type(self):
        # type: () -> UtteranceType
        """
        Returns
        -------
        type: UtteranceType
            Whether the utterance was a statement, a question or an experience
        """
        return self._type

    @property
    def transcript(self):
        # type: () -> str
        """
        Returns
        -------
        transcript: str
            Utterance Transcript
        """
        return self._hypothesis.transcript

    @property
    def confidence(self):
        # type: () -> float
        """
        Returns
        -------
        confidence: float
            Utterance Confidence
        """
        return self._hypothesis.confidence

    @property
    def turn(self):
        # type: () -> int
        """
        Returns
        -------
        turn: int
            Utterance Turn
        """
        return self._turn

    def set_turn(self, value):
        self._turn = value

    @property
    def triple(self):
        # type: () -> dict
        """
        Returns
        -------
        triple: Triple
            Structured representation of the utterance
        """
        return self._triple

    @property
    def perspective(self):
        # type: () -> dict
        """
        Returns
        -------
        perspective: Perspective
            NLP features related to the utterance
        """
        return self._perspective

    @property
    def datetime(self):
        return self._datetime

    @property
    def language(self):
        """
        Returns
        -------
        language: str
            Original language of the Transcript
        """
        raise NotImplementedError()

    @property
    def tokens(self):
        """
        Returns
        -------
        tokens: list of str
            Tokenized transcript
        """
        return self._tokens

    @property
    def parser(self):
        # type: () -> Optional[Parser]
        """
        Returns
        -------
        parsed_tree: ntlk Tree generated by the CFG parser
        """
        return self._parser

    def analyze(self):
        """
        Determines the type of utterance, extracts the RDF triple and perspective attaching them to the last utterance
        Returns
        -------

        """
        # Analyze utterance
        analyzer = Analyzer.analyze(self._chat)

        if not analyzer:
            return "I cannot parse your input"

        for el in ["subject", "predicate", "object"]:
            Analyzer.LOG.info(
                "RDF {:>10}: {}".format(el, json.dumps(analyzer.triple[el], sort_keys=True, separators=(', ', ': '))))

        # Set type, triple and perspective
        self._type = analyzer.utterance_type
        self.set_triple(analyzer.triple)
        if analyzer.utterance_type == UtteranceType.STATEMENT:
            self.set_perspective(analyzer.perspective)

    def set_triple(self, triple):
        # type: (dict) -> ()
        self._triple = triple

    def set_perspective(self, perspective):
        # type: (dict) -> ()
        self._perspective = perspective

    def casefold(self, format='triple'):
        # type (str) -> ()
        """
        Format the labels to match triples or natural language
        Parameters
        ----------
        format

        Returns
        -------

        """
        self._triple.casefold(format)
        self._chat_speaker = casefold_text(self.chat_speaker, format)

    def _choose_hypothesis(self, hypotheses):
        return sorted(self._patch_names(hypotheses), key=lambda hypothesis: hypothesis.confidence, reverse=True)[0]

    @staticmethod
    def _patch_names(hypotheses):
        names = []

        # Patch Transcripts with Names
        for hypothesis in hypotheses:

            transcript = []

            for word in hypothesis.transcript.split():
                name = Utterance._get_closest_name(word, friends)

                if name:
                    names.append(name)
                    transcript.append(name)
                else:
                    transcript.append(word)

            hypothesis.transcript = " ".join(transcript)

        if names:
            # Count Name Frequency and Adjust Hypothesis Confidence
            names = Counter(names)
            max_freq = max(names.values())

            for hypothesis in hypotheses:
                for name in list(names.keys()):
                    if name in hypothesis.transcript:
                        hypothesis.confidence *= float(names[name]) / float(max_freq)

        return hypotheses

    @staticmethod
    def _get_closest_name(word, names, max_name_distance=2):
        # type: (str, List[str], int) -> str
        if word[0].isupper() and names:
            name, distance = sorted([(name, edit_distance(name, word)) for name in names], key=lambda key: key[1])[0]

            if distance <= max_name_distance:
                return name

    def _tokenize(self, transcript):
        """
        Parameters
        ----------
        transcript: str
            Uttered text (Natural Language)

        Returns
        -------
        tokens: list of str
            Tokenized transcript: list of cleaned tokens for POS tagging and syntactic parsing
                - removes contractions and openers/introductions
        """

        # possible openers/greetings/introductions are removed from the beginning of the transcript
        # it is done like this to avoid lowercasing the transcript as caps are useful and google puts them
        openers = ['Leolani', 'Sorry', 'Excuse me', 'Hey', 'Hello', 'Hi']
        introductions = ['Can you tell me', 'Do you know', 'Please tell me', 'Do you maybe know']

        for o in openers:
            if transcript.startswith(o):
                transcript = transcript.replace(o, '')
            if transcript.startswith(o.lower()):
                transcript = transcript.replace(o.lower(), '')

        for i in introductions:
            if transcript.startswith(i):
                tmp = transcript.replace(i, '')
                first_word = tmp.split()[0]
                if first_word in ['what', 'that', 'who', 'when', 'where', 'which']:
                    transcript = transcript.replace(i, '')
            if transcript.startswith(i.lower()):
                tmp = transcript.replace(i.lower(), '')
                first_word = tmp.split()[0]
                if first_word.lower() in ['what', 'that', 'who', 'when', 'where', 'which']:
                    transcript = transcript.replace(i.lower(), '')

        # separating typical contractions
        tokens_raw = transcript.replace("'", " ").split()
        dict = {'m': 'am', 're': 'are', 'll': 'will'}
        dict_not = {'won': 'will', 'don': 'do', 'doesn': 'does', 'didn': 'did', 'haven': 'have', 'wouldn': 'would',
                    'aren': 'are'}

        for key in dict:
            tokens_raw = self.replace_token(tokens_raw, key, dict[key])

        if 't' in tokens_raw:
            tokens_raw = self.replace_token(tokens_raw, 't', 'not')
            for key in dict_not:
                tokens_raw = self.replace_token(tokens_raw, key, dict_not[key])

        # in case of possessive genitive the 's' is just removed, while for the aux verb 'is' is inserted
        if 's' in tokens_raw:
            index = tokens_raw.index('s')
            try:
                tag = pos_tag([tokens_raw[index + 1]])
                if tag[0][1] in ['DT', 'JJ', 'IN'] or tag[0][1].startswith('V'):  # determiner, adjective, verb
                    tokens_raw.remove('s')
                    tokens_raw.insert(index, 'is')
                else:
                    tokens_raw.remove('s')
            except:
                tokens_raw.remove('s')

        return tokens_raw

    @staticmethod
    def replace_token(tokens_raw, old, new):
        """
        :param tokens_raw: list of tokens
        :param old: token to replace
        :param new: new token
        :return: new list with the replaced token
        """
        if old in tokens_raw:
            index = tokens_raw.index(old)
            tokens_raw.remove(old)
            tokens_raw.insert(index, new)
        return tokens_raw

    @staticmethod
    def _clean(tokens):
        """
        Parameters
        ----------
        tokens: list of str
            Tokenized transcript

        Returns
        -------
        cleaned_tokens: list of str
            Tokenized & Cleaned transcript
        """
        return tokens

    def __repr__(self):
        author = self.chat.speaker
        return '{:>10s}: "{}"'.format(author, self.transcript)


class Parser(object):
    POS_TAGGER = None  # Type: POS
    NER_TAGGER = None
    GRAMMAR = None
    CFG_GRAMMAR_FILE = os.path.join(os.path.dirname(__file__), 'data', 'cfg.txt')

    def __init__(self, utterance):
        self._log = logger.getChild(self.__class__.__name__)

        if not Parser.POS_TAGGER:
            Parser.POS_TAGGER = POS()
            logger.info("Started POS tagger")

        if not Parser.NER_TAGGER:
            Parser.NER_TAGGER = NER()
            logger.info("Started NER tagger")

        with open(Parser.CFG_GRAMMAR_FILE) as cfg_file:
            if not Parser.GRAMMAR:
                Parser.GRAMMAR = cfg_file.read()
                logger.info("Loaded grammar")
            self._cfg = Parser.GRAMMAR

        self._forest, self._constituents = self._parse(utterance)

    @property
    def forest(self):
        return self._forest

    @property
    def constituents(self):
        return self._constituents

    def _parse(self, utterance):
        """
        :param utterance: an Utterance object, typically last one in the Chat
        :return: parsed syntax tree and a dictionary of syntactic realizations
        """
        self._log.debug("Start parsing")
        tokenized_sentence = utterance.tokens
        pos = self.POS_TAGGER.tag(tokenized_sentence)  # standford
        alternative_pos = pos_tag(tokenized_sentence)  # nltk

        self._log.debug(pos)
        self._log.debug(alternative_pos)

        if pos != alternative_pos:
            self._log.debug('DIFFERENT POS tag: %s != %s' % (pos, alternative_pos))

        # fixing issues with POS tagger (Does and like)
        ind = 0
        for w in tokenized_sentence:
            if w == 'like':
                pos[ind] = (w, 'VB')
            ind += 1

        if pos and pos[0][0] == 'Does':
            pos[0] = ('Does', 'VBD')

        # the POS tagger returns one tag with a $ sign (POS$) and this needs to be fixed for the CFG parsing
        ind = 0
        for word, tag in pos:
            if '?' in word:
                word = word[:-1]
            if tag.endswith('$'):
                new_rule = tag[:-1] + 'POS -> \'' + word + '\'\n'
                pos[ind] = (pos[ind][0], 'PRPPOS')
            else:
                # CFG grammar is created dynamically, with the terminals added each time from the specific utterance
                new_rule = tag + ' -> \'' + word + '\'\n'
            if new_rule not in self._cfg:
                self._cfg += new_rule
            ind += 1

        try:
            cfg_parser = CFG.fromstring(self._cfg)
            rd = RecursiveDescentParser(cfg_parser)

            last_token = tokenized_sentence[len(tokenized_sentence) - 1]

            if '?' in last_token:
                tokenized_sentence[len(tokenized_sentence) - 1] = last_token[:-1]

            parsed = rd.parse(tokenized_sentence)

            s_r = {}  # syntactic_realizations are the topmost branches, usually VP/NP
            index = 0

            forest = [tree for tree in parsed]

            if len(forest):
                if (len(forest)) > 1:
                    self._log.debug('* Ambiguity in grammar *')
                for tree in forest[0]:  # alternative trees? f
                    for branch in tree:
                        s_r[index] = {'label': branch.label(), 'structure': branch}
                        raw = ''
                        for node in branch:
                            for leaf in node.leaves():
                                raw += leaf + '-'

                        s_r[index]['raw'] = raw[:-1]
                        index += 1
            else:
                self._log.debug("no forest")

            for el in s_r:
                if type(s_r[el]['raw']) == list:
                    string = ''
                    for e in s_r[el]['raw']:
                        string += e + ' '
                    s_r[el]['raw'] = string
                s_r[el]['raw'] = s_r[el]['raw'].strip()

            return forest, s_r
        except:
            return [], {}
