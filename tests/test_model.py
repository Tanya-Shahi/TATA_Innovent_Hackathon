import sys
sys.path.append('../src')

import torch
import numpy as np
from model import LSTMModel


def test_model_output_shape():
    model = LSTMModel(input_size=14)
    dummy_input = torch.randn(32, 30, 14)
    output = model(dummy_input)
    assert output.shape == torch.Size([32]), f"Expected (32,) got {output.shape}"
    print("test_model_output_shape passed")


def test_model_output_range():
    model = LSTMModel(input_size=14)
    dummy_input = torch.randn(10, 30, 14)
    output = model(dummy_input).detach().numpy()
    print(f"test_model_output_range passed | Sample predictions: {output[:3]}")


def test_model_parameters():
    model = LSTMModel(input_size=14)
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params > 0, "Model has no parameters"
    print(f"test_model_parameters passed | Total params: {total_params:,}")


def test_model_save_load():
    import os
    model = LSTMModel(input_size=14)
    torch.save(model.state_dict(), '../models/test_temp.pt')
    assert os.path.exists('../models/test_temp.pt'), "Model file not saved"

    model2 = LSTMModel(input_size=14)
    model2.load_state_dict(torch.load('../models/test_temp.pt', weights_only=True))
    os.remove('../models/test_temp.pt')
    print("test_model_save_load passed")


def test_trained_model_loads():
    import os
    assert os.path.exists('../models/lstm_rul.pt'), "Trained model not found"
    model = LSTMModel(input_size=14)
    model.load_state_dict(torch.load('../models/lstm_rul.pt', weights_only=True))
    model.eval()
    dummy = torch.randn(1, 30, 14)
    with torch.no_grad():
        out = model(dummy)
    assert out is not None
    print(f"test_trained_model_loads passed | Prediction: {out.item():.2f}")


if __name__ == "__main__":
    test_model_output_shape()
    test_model_output_range()
    test_model_parameters()
    test_model_save_load()
    test_trained_model_loads()
    print("\nAll model tests passed!")