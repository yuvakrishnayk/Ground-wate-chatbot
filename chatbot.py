import json
import random
import nltk
from nltk.tokenize import word_tokenize

class GroundwaterChatbot:
    def __init__(self):
        # Load intents from the intents.json file
        with open('intents.json', 'r') as file:
            self.intents = json.load(file)

    def get_response(self, query):
        # Normalize the query to lower case and tokenize it
        query = query.lower()
        tokens = word_tokenize(query)

        for intent in self.intents['intents']:
            for pattern in intent['patterns']:
                # Tokenize the pattern for comparison
                pattern_tokens = word_tokenize(pattern.lower())
                # Check for matching tokens
                if all(token in tokens for token in pattern_tokens):
                    return random.choice(intent['responses'])
        
        return "I'm sorry, I don't understand your query."

    def generate_comprehensive_report(self):
        # Implement the logic to generate a comprehensive report
        report = {
            "Groundwater Resource Assessment": "Assessment details go here...",
            "Categorization of Area": "Categorization details go here...",
            "Management Practices": "Recommended management practices go here...",
            "NOC Conditions": "Conditions for NOC go here...",
            "NOC Guidance": "Guidance details go here...",
            "Definitions": "Definitions of groundwater terms go here...",
            "Training Opportunities": "Details about training opportunities go here..."
        }
        # Return the report as a formatted string or in a more structured format
        return report

# Example usage
if __name__ == "__main__":
    chatbot = GroundwaterChatbot()
    while True:
        user_input = input("You: ")
        if user_input.lower() == "generate report":
            report = chatbot.generate_comprehensive_report()
            print("Chatbot: Generating comprehensive report...")
            for section, details in report.items():
                print(f"{section}: {details}")
        else:
            response = chatbot.get_response(user_input)
            print(f"Chatbot: {response}")
