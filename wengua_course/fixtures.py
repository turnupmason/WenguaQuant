"""Tiny random Qwen3 CPU fixtures; not real 8B training or Qimen validation."""
def fixture_factory():
 import torch
 from tokenizers import Tokenizer,models,pre_tokenizers
 from transformers import PreTrainedTokenizerFast,Qwen3Config,Qwen3ForCausalLM,set_seed
 torch.set_num_threads(2);set_seed(3407)
 vocab={'[PAD]':0,'[UNK]':1,'[EOS]':2,'system':3,'user':4,'assistant':5,':':6,'one':7,'two':8,'answer':9,'question':10}
 tokenizer=Tokenizer(models.WordLevel(vocab=vocab,unk_token='[UNK]'));tokenizer.pre_tokenizer=pre_tokenizers.Whitespace()
 tok=PreTrainedTokenizerFast(tokenizer_object=tokenizer,pad_token='[PAD]',unk_token='[UNK]',eos_token='[EOS]')
 tok.chat_template="{% for message in messages %}{{ message['role'] + ': ' + message['content'] + '[EOS]' }}{% endfor %}{% if add_generation_prompt %}{{ 'assistant: ' }}{% endif %}"
 model=Qwen3ForCausalLM(Qwen3Config(vocab_size=len(tok),hidden_size=32,intermediate_size=64,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,head_dim=16,max_position_embeddings=2048,pad_token_id=0,eos_token_id=2))
 return tok,model,torch.float32
