# ISEF 2022 Candidate
#### Taking Control: A Novel Galvanic Stimulation Device for the Visually Impaired 
  ![githubfrontpage](https://user-images.githubusercontent.com/100437179/155861465-5c7b1c7a-2796-4e0b-b0ac-685588bb3837.jpg)


## Table of Contents
* [Introduction](#introduction)
* [Technologies Used](#technologies-used)
* [Features](#features)
* [Screenshots](#screenshots)
* [Setup](#setup)
* [Usage](#usage)
* [Project Status](#project-status)
* [Room for Improvement](#room-for-improvement)
* [Acknowledgements](#acknowledgements)
* [Contact](#contact)
<!-- * [License](#license) -->


## Introduction
   It's a futuristic concept - being able to fully control and steer a person out of the trajectory of incoming objects and hazards. Although this seems impossible, utilizing a non-evasive procedure named the Galvanic Vestibular Stimulation (GVS), was done safely and with few side effects. Using a special stereo sensor and an object detection algorithm, incoming objects within a 3-meter range are recognized. A python algorithm then determines the best trajectory to avoid the object through UDP packets to another device. The GVS device receives the packets and steers the subject autonomously according to the packets received.


## Technologies Used
  #### Hardware
  -	1 Intel RealSense Depth Camera D435
  -	1 Chest mount harness
  -	1 Tripod mount
  -	1 Tripod mount to ¼” screw conversion adapter
  -	1 5ft+ USB-A (v3.0+) to USB-C
  -	1 Laptop (CUDA-Enabled)
  -	1 Backpack
  #### GVS System
  -	1 Custom Designed PCB
  -	1 ESP8266 WIFI MCU
  -	1 MT3608 Boost Regulator
  -	2 1/4W 1.8k Ohm 5% Resistors
  -	2 1/4W 12k Ohm 5% Resistors
  -	1 1/4W 220 Ohm 5% Resistors
  -	7 1/4W 5.6k Ohm 5% Resistors
  -	4 2N3904 Transistors
  -	4 2N3906 Transistors
  -	1 Conn_01x03_Male Pin Header
  -	1 Conn_01x04_Male Pin Header
  -	1 Conn_02x04_Female Pin Header
  -	4 Conn_01x01_Male Pin Header
  -	1 3.5mm stereo jack (female) to 3P terminal
  -	1 Battery holder 2xAA with on/off switch
  -	4 M2x14mm screws
  -	4 M2x6mm female to female standoffs
  -	4 M2 Hex Nuts
  -	6 Female to female jumper wires
  -	1 Unshielded jumper wire
  -	1 USB ESP8266 Breakout
  -	1 3.5mm Electrode pad wire
  -	2 5x5cm Electrode pad
  #### Software
  -	Arduino IDE development platform
  -	Python 3.6.0
  #### 3D printed
  -	Housing Case
  -	Housing Lid



## Features
List the ready features here:
- Awesome feature 1
- Awesome feature 2
- Awesome feature 3


## Screenshots
![Example screenshot](./img/screenshot.png)
<!-- If you have screenshots you'd like to share, include them here. -->


## Setup
What are the project requirements/dependencies? Where are they listed? A requirements.txt or a Pipfile.lock file perhaps? Where is it located?

Proceed to describe how to install / setup one's local environment / get started with the project.


## Usage
How does one go about using it?
Provide various use cases and code examples here.

`write-your-code-here`


## Project Status
Project is: _in progress_ / _complete_ / _no longer being worked on_. If you are no longer working on it, provide reasons why.


## Room for Improvement
Include areas you believe need improvement / could be improved. Also add TODOs for future development.

Room for improvement:
- Improvement to be done 1
- Improvement to be done 2

To do:
- Feature to be added 1
- Feature to be added 2


## Acknowledgements
Give credit here.
- This project was inspired by...
- This project was based on [this tutorial](https://www.example.com).
- Many thanks to...


## Contact
Created by [@flynerdpl](https://www.flynerd.pl/) - feel free to contact me!


<!-- Optional -->
<!-- ## License -->
<!-- This project is open source and available under the [... License](). -->

<!-- You don't have to include all sections - just the one's relevant to your project -->
