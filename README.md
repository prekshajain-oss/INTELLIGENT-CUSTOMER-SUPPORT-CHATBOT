                                             INTELLIGENT CUSTOMER SUPPORT CHATBOT

import random

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB

# Training Data
questions = [
    "hello",
    "hi",
    "hey",
    "how are you",
    "i need help",
    "can you help me",
    "where is my order",
    "order status",
    "refund policy",
    "i want refund",
    "thanks",
    "thank you"
]

labels = [
    "greeting",
    "greeting",
    "greeting",
    "greeting",
    "help",
    "help",
    "order",
    "order",
    "refund",
    "refund",
    "thanks",
    "thanks"
]

# Responses
responses = {
    "greeting": ["Hello!", "Hi there!", "Hey! How can I help you?"],
    "order": ["Your order will arrive in 3-5 days."],
    "refund": ["Refunds are processed within 7 days."],
    "thanks": ["You're welcome!", "Happy to help!"]
}

# Convert text to numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(questions)

# Train model
model = MultinomialNB()
model.fit(X, labels)

print("Chatbot is ready! Type 'Quit' to stop.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "Quit":
        break

    user_vec = vectorizer.transform([user_input])
    prediction = model.predict(user_vec)[0]

    reply = random.choice(responses[prediction])
    print("Bot:", reply)



