import torch
import torch.nn.functional as F
import numpy as np
import tqdm


class MLPClassifier(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int, network_config: dict):
        super(MLPClassifier, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.config = network_config
        self.hidden_layers = [torch.nn.Linear(input_size, self.config['network_width'])] + [
            torch.nn.Linear(self.config['network_width'], self.config['network_width']) for i in range(self.config['network_layers'] - 1)
        ]
        self.predict = torch.nn.Linear(self.config['network_width'], output_size)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = F.relu(layer(x))

        x = self.predict(x)  # linear output
        return F.softmax(x, dim=-1)

    def draw_output(self, image_size, colors_to_use, beta, noise):
        out_image = np.zeros((image_size, image_size, 3), dtype='uint8')
        with torch.no_grad():
            for i in tqdm.tqdm(range(image_size), leave=False):
                for j in range(image_size):
                    x = (i + 0.5) / image_size
                    y = (j + 0.5) / image_size
                    point = torch.tensor([x, y, beta, noise])
                    output = self.forward(point)
                    output_class = output.argmax().item()
                    if (output < 0.5).all().item():
                        pass

                    out_image[i, j, :] = colors_to_use[output_class]

        return out_image
