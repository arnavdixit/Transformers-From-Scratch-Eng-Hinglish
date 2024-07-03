import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

from torch.utils.tensorboard import SummaryWriter

import torchmetrics

from dataset import BilingualDataset, causal_mask

from model import build_transformer

from config import get_config, get_latest_weights_file_path, get_weights_file_path

from pathlib import Path

from tqdm.auto import tqdm

def greedy_decode(model, src, src_mask, tokenizer_src, tokeinzer_tgt, max_len, device):
    sos_index = tokeinzer_tgt.token_to_id('[SOS]')
    eos_index = tokeinzer_tgt.token_to_id('[EOS]')
    
    encoder_output = model.encode(src, src_mask)
    decoder_input = torch.empty(1,1).fill_(sos_index).type_as(src).to(device)
    
    while True:
        if decoder_input.size(1) == max_len:
            break
        
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(src_mask).to(device)
        
        out = model.decode(encoder_output, src_mask, decoder_input, decoder_mask)
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1,1).type_as(src).fill_(next_word.item()).to(device)],
            dim = 1
        )
        
        if next_word == eos_index:
            break
    return decoder_input.squeeze(0)

def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples = 2):
    model.eval()
    count = 0
    
    src_texts = []
    predicted_text = []
    truth_text = []
    
    console_width = 80
    
    with torch.inference_mode():
        for batch in validation_ds:
            count += 1
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"
            
            model_output = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)
            
            src_text = batch['src_text'][0]
            tgt_text = batch['tgt_text'][0]
            model_out_text = tokenizer_tgt.decode(model_output.detach().cpu().numpy())
            
            src_texts.append(src_text)
            truth_text.append(tgt_text)
            predicted_text.append(model_out_text)
            
            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{src_text}")
            print_msg(f"{f'TARGET: ':>12}{tgt_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")
            
            if count == num_examples:
                print_msg('-'*console_width)
                break
            
        if writer:
            
            metric = torchmetrics.CharErrorRate()
            cer = metric(predicted_text, truth_text)
            writer.add_scalar('validation cer', cer, global_step)
            writer.flush()

            # Compute the word error rate
            metric = torchmetrics.WordErrorRate()
            wer = metric(predicted_text, truth_text)
            writer.add_scalar('validation wer', wer, global_step)
            writer.flush()

            # Compute the BLEU metric
            metric = torchmetrics.BLEUScore()
            bleu = metric(predicted_text, truth_text)
            writer.add_scalar('validation BLEU', bleu, global_step)
            writer.flush()

def get_all_utterances(ds, lang: str):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang: str) -> Tokenizer:
    tokenizer_path = Path(config["tokenizer_file"].format(lang))
    
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(BPE(unk_token = '[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(special_tokens = ['[UNK]', '[SOS]', '[EOS]', '[PAD]'], min_frequency = 2)
        tokenizer.train_from_iterator(get_all_utterances(ds, lang), trainer = trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    ds_raw = load_dataset(config['datasource'], split="train")
    
    # Build or fetch pre-trained tokenizer
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['src_lang'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['tgt_lang'])
    
    # Split dataset into training and validation set
    
    train_ds_size = int(len(ds_raw) * 0.9)
    valid_ds_size = len(ds_raw) - train_ds_size
    
    train_ds_raw, valid_ds_raw = random_split(ds_raw, [train_ds_size, valid_ds_size])
    
    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['src_lang'], config['tgt_lang'], config['seq_len'])
    valid_ds = BilingualDataset(valid_ds_raw, tokenizer_src, tokenizer_tgt, config['src_lang'], config['tgt_lang'], config['seq_len'])
    
    max_src_len = 0
    max_tgt_len = 0
    
    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['src_lang']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['tgt_lang']])
        max_src_len = max(max_src_len, len(src_ids))
        max_tgt_len = max(max_tgt_len, len(tgt_ids))

    print(f"Max length of {config['src_lang']} text is: {max_src_len}")
    print(f"Max length of {config['tgt_lang']} text is: {max_tgt_len}")
    
    train_dataloader = DataLoader(train_ds, batch_size = config['batch_size'], shuffle = True)
    valid_dataloader = DataLoader(valid_ds, batch_size = 1, shuffle = True)
    
    return train_dataloader, valid_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len: int, vocab_tgt_len: int):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'])
    return model

def train_model(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)
    
    train_dataloader, valid_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    
    writer = SummaryWriter(config['experiment_name'])
    
    optimizer = torch.optim.Adam(model.parameters(), lr = config['lr'], eps = 1e-8)
    
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = get_latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f"Preloading model: {model_filename}")
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        initial_epoch = state['epoch'] + 1
        global_step = state['global_step']
    else:
        print("No model to preload.")
        
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing = 0.1).to(device)
    for epoch in range(initial_epoch, config['epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc = f'Processing epoch: {epoch:02d}')
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device) # (batch, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (batch, seq_len)
            label = batch['label'].to(device)                 # (batch, seq_len)
            encoder_mask = batch['encoder_mask'].to(device)   # (batch, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device)   # (batch, 1, seq_len, seq_len)
            
            encoder_output = model.encode(encoder_input, encoder_mask) # (batch, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (batch, seq_len, d_model)
            projection_layer = model.proj_layer(decoder_output) # (batch, seq_len, tgt_vocab_size)
            
            loss = loss_fn(projection_layer.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
            
            writer.add_scalar("train_loss", loss)
            writer.flush()
            
            optimizer.zero_grad()
            
            loss.backward()
            
            optimizer.step()
            
            global_step += 1
        
        run_validation(model, valid_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg),
                    global_step, writer)
            
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
            
        torch.save({
            'epoch' : epoch,
            "model_state_dict" : model.state_dict(),
            "optimizer_state_dict" : optimizer.state_dict(),
            'global_step' : global_step
        }, model_filename)
            
            
            
if __name__ == "__main__":
    config = get_config()
    train_model(config)