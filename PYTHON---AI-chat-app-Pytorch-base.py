import torch
import torch.nn as nn
import tkinter as tk
from tkinter import scrolledtext

# Define the neural network model
class Chatbot(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Chatbot, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)

# Create the chatbot instance
input_size = 100  # Input size
hidden_size = 100  # Hidden layer size
output_size = 100  # Output size
chatbot = Chatbot(input_size, hidden_size, output_size)

# Load pre-trained weights (if available)
try:
    chatbot.load_state_dict(torch.load("chatbot_weights.pth"))
    print("Pre-trained weights loaded successfully!")
except:
    print("No pre-trained weights found. Starting with random weights.")

# Create the GUI
def send_message():
    input_text = input_entry.get()
    input_entry.delete(0, tk.END)
    output_text = "Chatbot: " + input_text + "\n"
    chat_history.insert(tk.END, output_text)

def quit_app():
    # Save the model weights before quitting
    torch.save(chatbot.state_dict(), "chatbot_weights.pth")
    root.quit()

root = tk.Tk()
root.title("PyTorch Chatbot")
root.geometry("400x400")

input_frame = tk.Frame(root)
input_frame.pack(pady=10)

input_entry = tk.Entry(input_frame, width=40)
input_entry.pack(side=tk.LEFT, padx=5)

send_button = tk.Button(input_frame, text="Send", width=10, command=send_message)
send_button.pack(side=tk.LEFT, padx=5)

chat_history = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=40, height=20)
chat_history.pack(padx=10, pady=10)

root.protocol("WM_DELETE_WINDOW", quit_app)
root.mainloop()
