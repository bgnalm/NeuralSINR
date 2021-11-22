import cv2
import torch
import model
import os
import data
import tqdm
import argparse
import json
import matplotlib.pyplot as plt
import csv


def train(network, optimizer, loss_function, data_generator, config, output_dir):
    iterations = config['iterations']
    batch_size = config['network_config']['batch_size']
    f = open(os.path.join(output_dir, 'losses.csv'), 'w')
    writer = csv.DictWriter(f, fieldnames=['iteration', 'loss'])
    writer.writeheader()

    image_size = config['images']['image_size']
    _, colors = data_generator.draw_problem(image_size=image_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network.train()
    batch_losses = []
    for iteration in tqdm.tqdm(range(iterations)):
        data_generator.update_vars()
        beta = data_generator.get_beta
        noise = data_generator.get_noise
        betas = torch.ones((batch_size, 1), dtype=torch.float32) * beta
        noises = torch.ones((batch_size, 1), dtype=torch.float32) * noise

        if config['images']['output_images']:
            if iteration % (iterations // config['images']['images_to_output']) == 0:
                # if we should compare outputs
                data_generator_image, colors = data_generator.draw_problem(image_size=image_size, colors=colors)
                model_image = network.draw_output(image_size=image_size, colors_to_use=colors, beta=beta, noise=noise)
                # save outputs
                cv2.imwrite(os.path.join(output_dir, f'iter_{iteration}_gt.png'), data_generator_image)
                cv2.imwrite(os.path.join(output_dir, f'iter_{iteration}_model.png'), model_image)

        x, y = data_generator.get_batch(batch_size)
        # currently don't support positional encoding
        x, y = torch.tensor(x.astype('float32')), torch.tensor(y.astype('float32'))
        x = torch.hstack([x, betas, noises])
        x, y = x.to(device), y.to(device)
        prediction = network(x)
        loss = loss_function(prediction, y)
        batch_losses.append(loss.item())
        writer.writerow({'loss': loss.item(), 'iteration': iteration})
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return batch_losses


def run_test(config, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    data_generator = data.DataGenerator(config['problem_configuration'])

    input_size = data_generator.get_dimension # x,y as input
    if config['network_config']['positional_encoding']:
        input_size *= config['network_config']['encoding_size'] # if we encode each value
    input_size += 2 # beta and noise are also inputs

    output_size = config['problem_configuration']['transmitters']['static_value']
    if not config['problem_configuration']['transmitters']['is_static']:
        output_size = config['problem_configuration']['transmitters']['dynamic']['range_max']

    network = model.MLPClassifier(input_size, output_size, config['network_config'])
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(network.parameters(), lr=0.2, momentum=0.95)

    return train(network, optimizer, loss_function, data_generator, config, output_dir)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help='path to config file', type=str)
    parser.add_argument("output", help='where to log the output')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    with open(args.config, 'r') as f:
        json_data = json.load(f)
        losses = run_test(json_data, args.output)
        plt.plot(losses)
        plt.show()
