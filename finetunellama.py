
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
import os
import torch
from datasets import load_dataset
from peft import PeftModel

from unsloth import FastLanguageModel
import torch
import json
import torch
from datasets import Dataset

from unsloth.chat_templates import train_on_responses_only
from unsloth import FastLanguageModel


import pandas as pd
df = pd.read_pickle('personas_and_tweets.df.pkl')

max_seq_length = 2048
dtype = None  # Auto detection of dtype
load_in_4bit = True  # Reduce memory usage with 4-bit quantization

quantized_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,)


    #change name to lora model here
lora_layers_and_quantized_model = FastLanguageModel.get_peft_model(
    quantized_model,
    r=16,  # LoRA rank
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,  # Optimized at 0
    bias="none",  # No additional bias terms
    use_gradient_checkpointing="unsloth",  # Gradient checkpointing to save memory
    random_state=3407,
    #use_rslora=False,  # Rank stabilized LoRA, can be enabled for stability
    use_rslora=True,  # Enable RS-LoRA for stable training

)


# Add special tokens for each user to improve differentiation
special_tokens = [f"<|user_{user}|>" for user in df['username'].unique()]
tokenizer.add_tokens(special_tokens)

# Prepare formatted conversations
conversations = []
dff = df.loc[df['training'] == 1]  # Filter training data

for _, row in dff.iterrows():
    conversation = [
        {"role": "system", "content": f'You are @{row["username"]} <|user_{row["username"]}|>'},
        {"role": "user", "content": row['reply_to']},
        {"role": "assistant", "content": row['message']}
    ]
    conversations.append(conversation)

# Formatting function for batch processing
def formatting_prompts_func(examples):
    return {
        "text": [
            tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
            for convo in examples["conversations"]
        ]
    }

# Convert conversations into a dataset
dataset = Dataset.from_dict({"conversations": conversations})

# Apply formatting in a batched way to avoid memory issues
dataset = dataset.map(formatting_prompts_func, batched=True, remove_columns=["conversations"])

# Tokenize the dataset
dataset = dataset.map(lambda x: tokenizer(x["text"], truncation=True, padding="max_length"), batched=True)

# Verify data structure
print(dataset[0])




from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported
trainer = SFTTrainer(
 model = lora_layers_and_quantized_model, # only lora layers will be trained, and quantized base model will remain frozen
 tokenizer = tokenizer,
 train_dataset = dataset,
 dataset_text_field = "text",
 max_seq_length = max_seq_length,
 data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
 dataset_num_proc = 2,
 packing = False,
 args = TrainingArguments(
    per_device_train_batch_size=4, #8,  # Increase batch size if VRAM allows
    gradient_accumulation_steps=4,
    warmup_steps=100,  # Increase warmup to stabilize learning rate
    max_steps= 400, #1000,  # Train longer for better adaptation
    learning_rate=5e-4,  # Higher LR speeds adaptation (fine-tune based on model loss)
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    logging_steps=20,
    optim="adamw_torch_fused",  # More stable optimizer
    weight_decay=0.01,  # Regularization to avoid overfitting
    lr_scheduler_type="cosine",  # Smooth LR decay
    seed=3407,
    output_dir="outputs",
),
)


trainer = train_on_responses_only(
 trainer,
 instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
 response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
)
trainer_stats = trainer.train()


# Prepare the model for Unsloth inference
finetuned_model = FastLanguageModel.for_inference(lora_layers_and_quantized_model)

# Define the input messages with explicit instruction to avoid system messages
# messages = [
#     {"role": "system", "content":"ಪರಿಸರದ ಬಗ್ಗೆ ಬರೆಯಿರಿ ಮತ್ತು ಪ್ರಬಂಧವನ್ನು ಬರೆಯಿರಿ."},
#     {"role": "user", "content":"ಪರಿಸರದ ಬಗ್ಗೆ ಬರೆಯಿರಿ ಮತ್ತು ಪ್ರಬಂಧವನ್ನು ಬರೆಯಿರಿ."}
# ]

# Save model 
import os

# Define the path for saving the model
save_dir = "./model2"
os.makedirs(save_dir, exist_ok=True)

print(f"Directory created: {save_dir}")
from transformers import AutoModelForCausalLM, AutoTokenizer

# Save the fine-tuned model
lora_layers_and_quantized_model.save_pretrained(save_dir)

# Save the tokenizer
tokenizer.save_pretrained(save_dir)

print(f"Model and tokenizer saved to {save_dir}")

def predict(row):
    messages = [
        {"role": "system", "content": f'You are @{row["username"]}'},
        {"role": "user", "content": row["reply_to"]}
    ]

    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to("cuda")

    outputs = finetuned_model.generate(
        input_ids=inputs,
        max_new_tokens=200,
        temperature=1.0,  # Reduce randomness for more accurate imitation
        top_p=0.9,  # Nucleus sampling for controlled creativity
        repetition_penalty=1.1  # Avoid repetitive responses
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response
