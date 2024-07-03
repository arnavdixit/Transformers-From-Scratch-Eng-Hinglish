from pathlib import Path

def get_config():
    return {
        "datasource" : "findnitai/english-to-hinglish",
        "main_exp" : "attention",
        "src_lang" : "en",
        "tgt_lang" : "hi_ng",
        "batch_size" : 8,
        "d_model" : 512,
        "seq_len" : 350,
        "epochs" : 5,
        "lr" : 10**-4,
        "model_folder" : "weights",
        "model_basename" : "transformer_",
        "preload" : None,
        "tokenizer_file" : "tokenizer_{}.json",
        "experiment_name" : "runs/transformer"
    }
    
def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['main_exp']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

def get_latest_weights_file_path(config):
    model_folder = f"{config['main_exp']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort(reverse = True)
    return weights_files[0]