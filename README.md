# gesture-craft
Bridging Play and Learning ‍Immerse yourself in a groundbreaking world of education. This visionary project takes the beloved Minecraft and infuses it with the magic of gestures, offering children an unparalleled hands-on experience. "Gesture - Craft" empowers kids to navigate and build within the virtual realm using their own hands.

## Introduction

The purpose of this research/project is to explore and demonstrate the potential benefits of embodied learning and gesturing in educational settings, particularly for children with attention deficiencies. This README provides an overview of the project's objectives, methodologies, findings, and implications.

### Research Objective

This project aims to investigate how interactive gestures and control gestures, as identified through gesture coding, impact children's cognitive processes and learning outcomes in educational settings. 

## Importance of Gestures in Learning

- Gesture research and its connection to learning.
- Relationship between gesturing and language learning in young children.
- Benefits of observing teachers' gestures in mathematics education.
- Positive effects of gesturing on long-term memory and learning outcomes.
- Gesturing as an important aspect of embodiment-based learning in later years.

## Bodily Activity and Embodied Learning

- Enactment as a theoretical component of embodied cognition (EC) theory.
- Bodily enactment of learning targets and its relation to embodied learning.
- Examples of bodily enactment studies in reading comprehension.
- The use of digital learning media to investigate embodiment in education.

### What is Embodied Cognition?

Embodied cognition is a cognitive science research paradigm that explores the connection between the body, environment, and cognitive processes. It highlights the role of the body in language comprehension, problem-solving, and forming rich multimodal representations.

## Survey Design

Conducted a survey to explore kids' preferred gestures in gaming. Identified commonalities and categorized gestures into two main groups: Interactive Gestures and Control Gestures. Analyzed survey data to understand the popularity and engagement levels of each gesture category.

### Dynamic vs Static Gestures

Distinguishes between gestures that involve dynamic physical movements and those that are more static or stationary in nature.

### Bodily vs Input-Driven Gestures

Focuses on the source or origin of the gestures, differentiating between gestures that rely on the physical movements of the body and those that rely on external inputs or devices.

### Spatial Manipulation vs Control Gestures

Distinguishes between gestures that involve manipulating the spatial aspects of the virtual environment and gestures that focus on controlling or interacting with virtual elements.

## Data Collection Program

- [Mp.py](mp.py) (media pipe action recorder and collector.)
- This code collects hand landmark data for different actions using a webcam and saves the data as numpy files for further analysis or training machine learning models.

## Gesture Recognition Program

- [cv.py](cv.py) (recognizes the recorded actions so that they can be implemented)
- Data Preparation, Classifier Training, and Real-time Action Recognition are explained in detail.

## Gameplay Program 
- [pre.py](pre.py)
- The program loads pre-recorded landmark data for various actions like walking, breaking/attacking, placing/using, jumping, and looking.
- It combines the loaded landmark data into training and testing datasets and labels them accordingly.
- The program establishes a connection to a Minecraft server (if available).
- It defines functions for various actions like moving forward, breaking blocks, placing blocks, looking left and right, and jumping in the Minecraft world.

## Gameplay Design

- Gesture-Based Interaction
- Dynamic Game Controls
- Real-Time Gameplay
- Continuous Improvement

## Results and Findings

- Preferred Gameplay
- Gesture Effectiveness
- Natural Gesture Usage
- Enjoyment and Immersion
- Feedback on Gesture System

## Future Work, Implications, and Takeaways

### Future Work

- Interface Testing
- Personalized Gestures
- Collaborative Gameplay
- Computational Components
- Long-Term Effects
- Educational Integration
- Accessibility Focus

### Implications and Takeaways

- Embodied Learning
- Seamless Interaction
- User-Centered Design
- Educational Tools
- Continued Research

## Conclusion: Shaping the Future of Learning

In the world of "Gesture - Craft," a realm where innovation seamlessly intertwines with education, a new paradigm emerges. The fusion of gesturing and Minecraft beckons children, particularly those with attention deficiencies, into an immersive learning experience that transcends traditional boundaries. This project's significance extends beyond the virtual realm, demonstrating the transformative capabilities of technology in education. By highlighting the cognitive advantages inherent in embodied learning, it underscores the potential of a generation engaged not as passive recipients but as active participants in their own educational narrative.

