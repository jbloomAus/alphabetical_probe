import pytest 
import os 

from transformer_lens import HookedTransformer

from src.probe_training import all_probe_training_runner


MODEL_NAME = "gpt2"

@pytest.fixture(scope="module")
def model():
    return HookedTransformer.from_pretrained(MODEL_NAME)


def test_all_probe_training_runner(model):
    # set wandb to offline mode
    os.environ["WANDB_MODE"] = "offline"

    vocab = model.tokenizer.get_vocab()
    probe_weights_tensor = all_probe_training_runner(
        embeddings=model.W_E.detach(),
        vocab=vocab,
        alphabet="ABC",
        criteria_mode="starts",
        probe_type="linear",
        num_epochs=4,
        batch_size=32,
        learning_rate=0.005,
        train_test_split=0.95,
        rebalance=True,
        use_wandb=True,
    )
    
    assert probe_weights_tensor is not None