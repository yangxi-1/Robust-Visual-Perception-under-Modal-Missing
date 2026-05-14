def get_config():
    return {
        "dataset": "mm_imdb",  
        "task": "multilabel",   # 自动改
        
        # "dataset": "hateful_memes",
        # "task": "classification",

        "freeze_encoder":False,
        "use_prompt":False,

        "batch_size": 32,
        "lr": 1e-4,
        "epochs": 10,

        
        
        "num_classes": None,    # 自动推断

        "drop_prob": 0.3,
        "device": "cuda"
    }
