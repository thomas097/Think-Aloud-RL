# Make sure JAVAHOME path is set to your Java installation
import sys, os
sys.path.insert(0, 'src')
os.environ['JAVAHOME'] = "C:/Program Files/Java/jre1.8.0_311/bin/java.exe"

from cltl.triple_extraction.api import Chat, UtteranceHypothesis


if __name__ == "__main__":
    # Example utterance
    utt = "Thomas likes dogs"

    # Request triple(s) from Leolani
    chat = Chat("Lenka")
    chat.add_utterance([UtteranceHypothesis(utt, 1.0)])
    chat.last_utterance.analyze()

    # If no triple was extracted, be sad :(
    if chat.last_utterance.triple is None:
        print("No triple extracted...")
    else:
        triple = chat.last_utterance.triple
        print(triple)
