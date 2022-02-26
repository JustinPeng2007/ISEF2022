# ISEF2022

Taking Control: A Novel Galvanic Stimulation Device for the Visually Impaired



It is a futuristic concept - being capable of controlling the safety with which a visually impaired person can move. Although this may seem impossible, using a non-evasive procedure named the Galvanic Vestibular Stimulation (GVS), it can be done safely and with few side effects. By applying a current of electrons (1- 1.5mA) delivered just behind the ear and in a certain direction, the vestibular system is stimulated to send additional signals to the brain, causing the sensation of being “steered”. 
  
  The white cane has been the most prominent and widely used mobility aid for the visually impaired for many decades. However, this mobility aid has many limitations. The white cane is only capable of detecting ground-level obstacles that are within proximity of the user. To target these limitations, the GVS device was created. Using a combination of Intel’s D435 depth camera and an object detection model YOLOv5, incoming objects within a 3-meter range are recognized and processed. In less than 0.2 seconds from detecting the object, an algorithm assesses the situation and sends out instructions via UDP (User Datagram Protocol) packets wirelessly to the GVS device. The GVS device receives the packets and steers the subject autonomously according to the packets received. 
  
  In total, four trials were conducted, each with different a scenario. Through experimentation, it was found that the device could successfully detect, process, and transmit geospatial information to the GVS system, then steer the user into the correct trajectory to avoid any hazards obstructing the user’s path. 
