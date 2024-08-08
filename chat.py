import random
import json

import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
def addBags():
    bookingRef = input("You: ")
    print(f"{bot_name}: How many bags do you want to add?")
    addedBags = int(input("You: "))
    print(f"{bot_name}: I have successfully added {addedBags} bags to your booking. ")
    return
def changeBooking():
    bookingRef = input("You: ")
    print(f"{bot_name}: When do you want to fly ? Please give me your date in DD/MM/YEAR format")
    date = input("You: ")
def bookFlight():
    arr = input("You: ")
    print(f"{bot_name}: Where are you leaving from?")
    dep = input("You: ")
    print(f"{bot_name}: When do you want to fly ? Please give me your date in DD/MM/YEAR format")
    date = input("You: ")
    print(f"{bot_name}: Great! I have booked your flight from {dep} to {arr} on {date}. We can't wait to welcome you on board!")
    
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Cedar"
while True:
     sentence = input("You: ")
     if sentence == "quit":
         break

     sentence = tokenize(sentence)
     X = bag_of_words(sentence, all_words)
     X = X.reshape(1, X.shape[0])
     X = torch.from_numpy(X).to(device)

     output = model(X)
     _, predicted = torch.max(output, dim=1)

     tag = tags[predicted.item()]

     probs = torch.softmax(output, dim=1)
     prob = probs[0][predicted.item()]
     if tag == "AddBaggage":
        for intent in intents['intents']:
             if tag == intent["tag"]:
                 print(f"{bot_name}: {random.choice(intent['responses'])}")
        addBags()
        break
     if tag == "ChangeBooking":
        for intent in intents['intents']:
             if tag == intent["tag"]:
                 print(f"{bot_name}: {random.choice(intent['responses'])}")
        changeBooking()
        break
     if tag == "BookFlight":        
        for intent in intents['intents']:
             if tag == intent["tag"]:
                 print(f"{bot_name}: {random.choice(intent['responses'])}")
        bookFlight()
        break
     if prob.item() > 0.75:
         for intent in intents['intents']:
             if tag == intent["tag"]:
                 print(f"{bot_name}: {random.choice(intent['responses'])}")
     else:
         print(f"{bot_name}: I am not sure I can help with that. Please hold tight as I connect you to one of our agents")
def get_response(sentence):
    if sentence == "quit":
        return 
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return (f"{bot_name}: {random.choice(intent['responses'])}"), tag
    else:
        return (f"{bot_name}: I am not sure I can help with that. Please hold tight as I connect you to one of our agents"),tag
