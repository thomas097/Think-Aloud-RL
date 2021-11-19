# Project Think Aloud

This repository contains a modified version of the [_Leolani platform_](https://github.com/leolani) as used in the _Think Aloud_ project for the _Communicative Robots_ Course at the Vrije Universiteit (VU).

In this repo, we provide an alternative _reinforcement learning_ approach to the random _thought_ selection mechanism implemented by Leolani v1, allowing the robot to learn 
which of her thoughts to verbalize in order to optimally attain new knowledge about her environment and do so in an online manner. To choose between thoughts, the Upper 
Confidence Bounds (UCB) algorithm was implemented and used to select between high-level _thought types_ (e.g. negation conflicts, object gaps, entity novelties, etc.); specific 
thoughts of the selected type are then verbalized according to the likelihood of their occurrance in natural open-domain dialogue.
