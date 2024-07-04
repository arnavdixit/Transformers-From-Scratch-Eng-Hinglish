from pathlib import Path

def get_config():
    return {
        "datasource" : "findnitai/english-to-hinglish",
        "src_lang" : "en",
        "tgt_lang" : "hi_ng",
        "batch_size" : 32,
        "d_model" : 512,
        "seq_len" : 350,
        "epochs" : 15,
        "lr" : 10**-3,
        "lr_step_size" : 5,
        "step_gamma" : 0.5,
        "max_norm" : 1,
        "model_folder" : "weights",
        "model_basename" : "transformer_",
        "preload" : None,
        "tokenizer_file" : "tokenizer_{}.json",
        "experiment_name" : "runs/transformer"
    }
    
def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

def get_latest_weights_file_path(config):
    model_folder = f"{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort(reverse = True)
    return weights_files[0]