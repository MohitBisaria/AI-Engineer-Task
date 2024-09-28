import torch
#from datasets import load_from_csv
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from datasets import load_from_disk
from flask import Flask, request, jsonify
import pandas as pd
import os

# Load pre-trained model and tokenizer
model_name = "facebook/bart-large-cnn"  # Example model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Load datasets from CSV files (replace with your actual file paths)
#train_dataset = load_from_disk(r"C:\Users\DELL\Downloads\archive (2)\cnn_dailymail\train")
#validation_dataset = load_from_disk(r"C:\Users\DELL\Downloads\archive (2)\cnn_dailymail\validation")
#test_dataset = load_from_disk(r"C:\Users\DELL\Downloads\archive (2)\cnn_dailymail\test")

#train_dataset = pd.read_csv(r"C:\Users\DELL\Downloads\archive (2)\cnn_dailymail\train\train.csv")
#validation_dataset = pd.read_csv(r"C:\Users\DELL\Downloads\archive (2)\cnn_dailymail\validation\validation.csv")
#test_dataset = pd.read_csv(r"C:\Users\DELL\Downloads\archive (2)\cnn_dailymail\test\test.csv")


#train_dataset = list(train_dataset.to_dict(orient='records'))
#validation_dataset = list(validation_dataset.to_dict(orient='records'))
#test_dataset = list(test_dataset.to_dict(orient='records'))

# Load datasets from CSV files (replace with your actual file paths)
#train_dataset = load_from_disk(os.path.abspath(r"C:\Users\DELL\Downloads\archive (2)\cnn_dailymail\dataset\train"))
#validation_dataset = load_from_disk(os.path.abspath(r"C:\Users\DELL\Downloads\archive (2)\cnn_dailymail\dataset\validation"))
#test_dataset = load_from_disk(os.path.abspath(r"C:\Users\DELL\Downloads\archive (2)\cnn_dailymail\dataset\test"))


#train_dataset = load_from_csv(os.path.abspath(r"C:\Users\DELL\Downloads\archive (2)\cnn_dailymail\train\train.csv"))  
#validation_dataset = load_from_csv(os.path.abspath(r"C:\Users\DELL\Downloads\archive (2)\cnn_dailymail\validation\validation.csv"))
#test_dataset = load_from_csv(os.path.abspath(r"C:\Users\DELL\Downloads\archive (2)\cnn_dailymail\test\test.csv"))


#train_dataset = pd.read_csv(r"C:\Users\DELL\Desktop\archive (2)\cnn_dailymail\dataset\train\train.csv")
#validation_dataset = pd.read_csv(r"C:\Users\DELL\Desktop\archive (2)\cnn_dailymail\dataset\validation\validation.csv")
#test_dataset = pd.read_csv(r"C:\Users\DELL\Desktop\archive (2)\cnn_dailymail\dataset\test\test.csv")


train_dataset = pd.read_csv(r"C:\Users\DELL\Desktop\samsum\samsum-train.csv")
validation_dataset = pd.read_csv(r"C:\Users\DELL\Desktop\samsum\samsum-validation.csv")
test_dataset = pd.read_csv(r"C:\Users\DELL\Desktop\samsum\samsum-test.csv")

print(type(train_dataset))

def preprocess_function(examples):
    #inputs = examples["article"]
    #targets = examples["highlights"]
    #inputs = [example["article"] for example in examples]
    #targets = [example["highlights"] for example in examples]

    #inputs = examples  # Convert to a list
    #targets = examples["summary"]
    targets = train_dataset["summary"].tolist()

    model_inputs = tokenizer(examples, max_length=1024, truncation=True)
    labels = tokenizer(targets, max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Preprocess the datasets
tokenized_train_dataset = train_dataset.map(preprocess_function)
tokenized_validation_dataset = validation_dataset.map(preprocess_function)
tokenized_test_dataset = test_dataset.map(preprocess_function)

#tokenized_train_dataset=preprocess_function(train_dataset)
#tokenized_validation_dataset=preprocess_function(validation_dataset)
#tokenized_test_dataset=preprocess_function(test_dataset)

# Create PyTorch dataloaders
train_dataloader = torch.utils.data.DataLoader(tokenized_train_dataset, batch_size=4, shuffle=True)
validation_dataloader = torch.utils.data.DataLoader(tokenized_validation_dataset, batch_size=4)
test_dataloader = torch.utils.data.DataLoader(tokenized_test_dataset, batch_size=4)

# Define training parameters
num_epochs = 3
learning_rate = 1e-5
warmup_steps = 500 

# Optimization and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=learning_rate)
num_training_steps = len(train_dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps
)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop
for epoch in range(num_epochs):
    model.train()
    for batch_idx, batch in enumerate(train_dataloader):
        for k, v in batch.items():
            batch[k] = v.to(device)

        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        print(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}")

    # Evaluation on validation set
    model.eval()
    validation_loss = 0
    with torch.no_grad():
        for batch in validation_dataloader:
            for k, v in batch.items():
                batch[k] = v.to(device)

            outputs = model(**batch)
            validation_loss += outputs.loss.item()

    validation_loss /= len(validation_dataloader)
    print(f"Epoch: {epoch}, Validation Loss: {validation_loss}")

# Save the fine-tuned model
model.save_pretrained("fine_tuned_summarization_model")
tokenizer.save_pretrained("fine_tuned_summarization_model")

# Load the fine-tuned model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained("fine_tuned_summarization_model")
tokenizer = AutoTokenizer.from_pretrained("fine_tuned_summarization_model")

# Create Flask app
app = Flask(__name__)

@app.route('/summarise', methods=['POST'])
def summarise():
    data = request.get_json()
    prompt = data['prompt']

    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(**inputs, max_length=100, num_beams=4)
    summary = tokenizer.decode(output[0], skip_special_tokens=True)

    return jsonify({'response': summary})

if __name__ == "__main__":
    app.run(debug=True)