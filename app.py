from flask import Flask, render_template, request, jsonify
from chatbot import GroundwaterChatbot

app = Flask(__name__)
chatbot = GroundwaterChatbot()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_query = request.form['query']
    response = chatbot.get_response(user_query)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)
