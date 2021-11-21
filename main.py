""" Filename:     main.py
    Author(s):    Thomas Bellucci
    Description:  This file provides an example of an interaction loop for
                  the RLChatbot defined in Chatbot.py. By typing 'plot' the
                  replier will plot the learnt thought utilities and 'quit'
                  ends the interaction.
    Date created: Nov. 11th, 2021
"""

from Chatbot import RLChatbot


if __name__ == "__main__":

    chatbot = RLChatbot(speaker="thomas")
    print("\nLeo:", chatbot.greet)

    while True:
        # Say something
        input_ = input("\nYou: ")

        if input_ == "quit":
            break
        if input_ == "plot":
            chatbot.show_thoughts()
            continue

        # Reply to user input
        print("\nLeo:", chatbot.respond(input_))

    print("\nLeo:", chatbot.farewell)
    

    
