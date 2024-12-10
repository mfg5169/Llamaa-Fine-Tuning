from transformers import LlamaForCausalLM, LlamaTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

dataset_path = "dataset"  # Replace with your dataset path or Hugging Face dataset name
dataset = load_dataset("text", data_files={"train": f"{dataset_path}/train.txt", "test": f"{dataset_path}/test.txt"})

model_name = "huggingface/llama3b"  # Replace with the appropriate model checkpoint
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Remove unnecessary columns for training
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets.set_format("torch")

training_args = TrainingArguments(
    output_dir="./results",          # Output directory
    overwrite_output_dir=True,      # Overwrite existing outputs
    evaluation_strategy="epoch",    # Evaluate at the end of each epoch
    learning_rate=5e-5,             # Learning rate
    num_train_epochs=3,             # Number of training epochs
    per_device_train_batch_size=8,  # Batch size for training
    per_device_eval_batch_size=8,   # Batch size for evaluation
    save_strategy="epoch",          # Save checkpoint every epoch
    logging_dir="./logs",           # Logging directory
    save_total_limit=2,             # Save only the last two checkpoints
    fp16=True,                      # Enable mixed precision (if supported by GPU)
    load_best_model_at_end=True,    # Load the best model at the end of training
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
)

trainer.train()

trainer.save_model("./fine_tuned_llama")
tokenizer.save_pretrained("./fine_tuned_llama")
