from models.base_model import BaseForecaster
from config.base_config import ModelConfig
from models.naive import SeasonalNaive
from models.lstm_seq2seq import LSTMSeq2Seq
from models.tcn import TCN
from models.transformer import TransformerForecaster

def create_model(config: 'ModelConfig') -> BaseForecaster:
    """
    Factory function to create models
    
    Args:
        config: Model configuration
        
    Returns:
        Model instance
    """
    model_type = config.model_type.lower()
    
    if model_type == "naive":
        return SeasonalNaive(config)
    elif model_type in ["lstm", "gru"]:
        return LSTMSeq2Seq(config, cell_type=model_type.upper())
    elif model_type == "tcn":
        return TCN(config)
    elif model_type == "transformer":
        return TransformerForecaster(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")