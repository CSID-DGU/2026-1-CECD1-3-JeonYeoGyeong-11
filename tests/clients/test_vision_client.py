import unittest

import torch
from torch.utils.data import TensorDataset

from spectral_fl.clients.vision import VisionFlowerClient


class VisionClientOptimizerStateTest(unittest.TestCase):
    def test_fit_rebuilds_optimizer_after_server_parameters_are_loaded(self):
        train = TensorDataset(
            torch.zeros((2, 1, 28, 28), dtype=torch.float32),
            torch.tensor([0, 1], dtype=torch.long),
        )
        test = TensorDataset(
            torch.zeros((1, 1, 28, 28), dtype=torch.float32),
            torch.tensor([0], dtype=torch.long),
        )
        client = VisionFlowerClient(
            cid=0,
            train_dataset=train,
            train_indices=torch.tensor([0, 1]).numpy(),
            test_dataset=test,
            model_name="mlp",
            num_classes=10,
            in_channels=1,
            lr=0.01,
            momentum=0.9,
            weight_decay=0.0,
            batch_size=2,
            local_epochs=1,
            device=torch.device("cpu"),
            seed=1,
        )
        params = client.get_parameters({})
        client.optim.state["sentinel"] = {"momentum_buffer": torch.ones(1)}

        client.fit(params, {"seed": 1, "server_round": 1})

        self.assertNotIn("sentinel", client.optim.state)


if __name__ == "__main__":
    unittest.main()
