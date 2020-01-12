# StateOfHealthTracker-NeuralNetwork

The application builds a neural network in the Python language.

Datasets are input, where:
* azimut - type: number, description: azimuth of phone sensor
* pitch - type: number, description: pitch of phone sensor
* roll - type: number, description: roll of phone sensor
* missClicks - type: number, description: user missed the button
* textDistance - type: number, description: average Levenshtein distance between words
* textTime - type: number, description: how long user retype original text in ms
* textErased - type: number, description: average number of symbols user erased
* leftCount - type: number, description: average number of taps for left hand
* rightCount - type: number, description: average number of taps for right hand
* state - type: number, description: state of health in time
* pill - type: number, description: number of tablets that user took in time
* dyskinesia - type: number, description: number of dyskinesias per time
* speed - type: number, description: average user speed per time
* volume - type: number, description: average user voice volume
* pauseCount - type: number, description: avarage number of pauses between words
* pauseTime -  type: number, description: average pause time

Details: 
State has 3 meanings: good(1), bad(0) and neutral(0.5). 
The output should determine the patient's condition has improved, worsened or remained unchanged.
