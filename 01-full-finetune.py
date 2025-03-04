import torch
from transformers import AutoTokenizer
from trl import SFTTrainer, ModelConfig, SFTConfig
from datasets import load_dataset
from liger_kernel.transformers import AutoLigerKernelForCausalLM

model_name = 'Qwen/Qwen2.5-0.5B-Instruct'
dataset_name = 'lg-abstract-dataset.jsonl.xz'
output_dir = "Qwen2.5-0.5B-Instruct-lg-abstracts"

train_dataset = load_dataset("json", data_files=dataset_name, split="train")

tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None: 
    tokenizer.pad_token = tokenizer.eos_token

model = AutoLigerKernelForCausalLM.from_pretrained(model_name, 
            attn_implementation='flash_attention_2',
            torch_dtype='bfloat16', use_cache=False,
            low_cpu_mem_usage=True, device_map="cuda")

training_args = SFTConfig(output_dir=output_dir, num_train_epochs=4,
     bf16=True, packing=True, max_seq_length=1024,
     per_device_train_batch_size=8,
     gradient_accumulation_steps=2,
     gradient_checkpointing=True,
     gradient_checkpointing_kwargs = { "use_reentrant": False },
     learning_rate=2.0e-4, lr_scheduler_type="constant",
     use_liger=True, warmup_ratio=0.1,)

trainer = SFTTrainer(model=model, args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer)

train_result = trainer.train()
print(train_result.metrics)

trainer.model.config.use_cache = True
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"saved to {output_dir}")

