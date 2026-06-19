# Year 3 Intelligent Systems and Robotics

This repository contains the Year 3 project **Autonomous Robotic Navigation and Image Processing for Spacecraft Simulation**, completed as part of Intelligent Systems and Robotics at the University of Leeds.

The project focused on building an autonomous robotic system capable of navigating through a simulated spacecraft environment while using computer vision techniques to support safe navigation, object recognition, and image-based analysis.

## Project Overview

The system was developed using **Python** and **ROS 2** for autonomous robotic navigation in a spacecraft simulation environment. The robot was required to navigate safely, avoid false positives, identify relevant visual targets, and complete the mission within a strict time limit.

A key part of the project involved combining navigation logic with image processing methods. This included using feature detection and image matching techniques to create panoramic views of celestial bodies, supporting navigation error correction and visual interpretation in a simulated space mission scenario.

## Key Achievements

- Developed an autonomous navigation system using Python and ROS 2
- Enabled a robot to move through a simulated spacecraft environment
- Implemented logic to distinguish safe navigation targets from false positives
- Engineered an image stitching pipeline using SIFT and FLANN
- Created panoramic views of celestial bodies for image-based analysis
- Optimised navigation strategies to meet a critical 5-minute mission deadline
- Conducted simulation testing and real robot validation
- Improved robustness and reliability under varied environmental conditions

## Main Features

| Feature | Description |
|---|---|
| Autonomous Navigation | Robot navigation through a simulated spacecraft environment using ROS 2 |
| Safe Target Recognition | Logic for distinguishing safe targets from false positives |
| Image Processing | Computer vision methods for analysing visual information during the mission |
| Panorama Creation | SIFT and FLANN-based image stitching for celestial body imagery |
| Mission-Time Optimisation | Real-time navigation strategy designed around a 5-minute mission deadline |
| Testing and Validation | Simulation and real robot testing to evaluate robustness and reliability |

## Technologies Used

- Python
- ROS 2
- Linux
- OpenCV
- SIFT
- FLANN
- Computer vision
- Image stitching
- Autonomous robotics
- Robot simulation
- Real-time navigation

## Computer Vision and Image Processing

The project included an image stitching component designed to generate panoramic views from multiple images. This was achieved using feature detection and matching techniques.

Key techniques included:

- SIFT feature detection
- FLANN-based feature matching
- Image alignment
- Panorama generation
- Visual analysis for spacecraft navigation support

This component helped create clearer views of celestial bodies and supported navigation error correction in the simulated mission environment.

## Robotics and Navigation

The robotics component focused on autonomous movement and decision-making. The robot needed to navigate through the environment efficiently while avoiding incorrect detections and completing mission objectives within the time constraint.

Key robotics concepts included:

- Autonomous navigation
- ROS 2 nodes and communication
- Real-time decision-making
- Environment perception
- False positive handling
- Mission planning under time constraints
- Simulation-based testing
- Real robot validation

## Testing and Validation

Testing was an important part of the project. The system was evaluated through simulation and real robot testing to check whether the navigation and image processing components worked reliably in different environmental conditions.

The testing process focused on:

- Navigation accuracy
- Robustness of image processing results
- Handling of false positives
- Mission completion within 5 minutes
- Reliability in simulation
- Readiness for real robot deployment

## Skills Demonstrated

- Robotics software development
- Python programming
- ROS 2 development
- Computer vision with OpenCV
- Feature detection and matching
- Image stitching and panorama generation
- Autonomous navigation logic
- Real-time optimisation
- Simulation testing
- Practical validation on robotic systems

## Summary

This project demonstrates the integration of autonomous robotics and computer vision for a simulated spacecraft mission. It combines Python, ROS 2, OpenCV, navigation logic, SIFT and FLANN-based image stitching, and rigorous testing to support reliable robotic behaviour under time-constrained mission conditions.
