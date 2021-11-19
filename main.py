""" Filename:     main.py
    Author(s):    Thomas Bellucci
    Description:  This file provides an example of how to implement a simple
                  interaction loop for the RLChatbot defined in Chatbot.py.
    Date created: Nov. 11th, 2021
"""

from Chatbot import RLChatbot


if __name__ == "__main__":

    chatbot = RLChatbot(speaker="Thomas")
    print("\nLeo:", chatbot.greeting)
    
    while True:
        # Say something to Leolani
        _input = input("\nYou: ")

        if _input == "quit": break
        if _input == "plot":
            chatbot.thought_selector.plot()
            continue

        # Ask her to respond
        print("\nLeo:", chatbot.reply(_input))

    print("\nLeo:", chatbot.goodbye)
    

    
