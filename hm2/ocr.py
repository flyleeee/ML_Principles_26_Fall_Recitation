import argparse
import math
import numpy as np
import random
from PIL import Image
import torch

import torchvision
import yaml

import example_models
import updated_hmm


def idxToAlpha(idx):
    if idx == 26:
        return ' '
    else:
        return chr(ord('a') + idx)

def alphaToIdx(alpha):
    if alpha == ' ':
        return 26
    else:
        return ord(alpha) - ord('a')


def makeBFromConfusion(confusion):
    """Convert the confusion matrix to only lowercase alphabetic entries."""
    # 26 letters plus a space
    B = np.zeros((27, 27))
    # Each column in the confusion matrix maps from a prediction to an actual
    # Convert columns in the confusion matrix to rows in B
    B[:26,:26] = np.array(confusion).T
    # Assume that a space is never confused with anything else
    B[26,26] = 1.0
    return B


def makeAFrom1Gram(transitions):
    """Convert the 1 grams to a transition matrix and the initial state estimates, pi."""
    # 26 letters plus a space
    A = np.zeros((27, 27))
    pi = np.zeros(27)
    # Each column in the confusion matrix maps from a prediction to the true label
    # Convert columns in the confusion matrix to rows in B
    for entry, prob in transitions.items():
        # Use first letters of words to initialize pi
        if ' ' == entry[0]:
            pi[alphaToIdx(entry[1].lower())] = prob
        A[alphaToIdx(entry[0].lower()),alphaToIdx(entry[1].lower())] += prob

    # Normalize pi and transition rows
    pi = pi / pi.sum()
    for i in range(26):
        A[:,i] = A[:,i] / np.sum(A[:,i])

    return A, pi


def computeImgClasses(img, model, patch_size=28, stride=28, batch_size=100):
    # Ensure image has batch dimension: (1, C, H, W)
    if img.dim() == 3:
        img = img.unsqueeze(0)

    # Unfold along height (dim 2) and width (dim 3)
    # Output shape: (1, C, patch_size, patch_size, L_h, L_w)
    unfold = torch.nn.Unfold(kernel_size=(patch_size, patch_size), stride=stride)
    patches = unfold(img)

    width = math.ceil((img.size(3) - (patch_size-1)) / stride)
    height = math.ceil((img.size(2) - (patch_size-1)) / stride)

    # Reshape patches into a batch format: (N, C, patch_size, patch_size)
    # Currently the batch dimension is last
    patches = patches.view(1, patch_size, patch_size, -1)
    patches = patches.permute(3, 0, 1, 2).contiguous()

    # Get model predictions and extract the maximum value for each patch
    # Batch things to conserve memory
    model.eval()
    batches = int(np.ceil(patches.size(0)/float(batch_size)))
    class_outputs = []
    with torch.no_grad():
        for batch in range(batches):
            begin = batch*batch_size
            end = (batch+1)*batch_size

            patch_batch = patches[begin:end]
            outputs = model(patch_batch)

            # Take max across class/output dimension
            if outputs.dim() > 1:
                batch_max_outputs, _ = torch.max(outputs, dim=1)
            else:
                batch_max_outputs = outputs
            class_outputs.append(outputs)

    class_outputs = torch.concatenate(class_outputs).view((height, width, -1))

    return class_outputs


def prepareImage(image, device):
    # Convert a black on white 0-255 image to white on black, 0 or 1
    # Add a channel and batch dimension
    img = (torchvision.transforms.ToTensor()(image)).float().unsqueeze(0)
    return img


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        type=str,
        default=None,
        help="Yaml config file.")
    parser.add_argument(
        "--image",
        required=True,
        type=str,
        default=None,
        help="Test image.")
    parser.add_argument(
        "--device",
        required=False,
        type=str,
        default=None,
        help="Override the automatically determined device (cuda or cpu).")

    args = parser.parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Read the model name and confusion matrix
    with open(f"{args.config}", 'r') as infile:
        config = yaml.safe_load(infile)

    version = config['model_version']

    # Create the model
    if version == 'convnet':
        model = example_models.ConvNet(classes=26)
    elif version == 'wider':
        model = example_models.WiderConvNet(classes=26)
    elif version == 'deeper':
        model = example_models.DeeperConvNet(classes=26)
    elif version == 'next':
        model = example_models.ConvNeXtEMNIST(classes=26)

    # Load the model
    model.load_state_dict(torch.load(config['model'], weights_only=True, map_location=device))
    model.to(device)

    np.set_printoptions(precision=3, suppress=True)

    # Get the emission matrix from the confusion matrix
    # Reduce to just the lower-case characters, and add in the space
    confusion = config['confusion']
    B = makeBFromConfusion(confusion)

    #print(f"B is {B}")


    # Get the transition matrix from grammar rules
    # Note: We can actually use multiple ngrams of different length.
    # Start with the short ones, and then get to a longer one with enough
    # history. This would give more robust results.
    transitions = config['1gram']
    A, pi = makeAFrom1Gram(transitions)

    #print(f"A is {A}")
    #print(f"pi is {pi}")

    # The image is already color, just convert to a float from 0 to 1
    img = Image.open(args.image)
    img = prepareImage(img, device).to(device)

    # Get all DNN predictions over the image. Use stride 28 by default
    raw_predictions = computeImgClasses(img, model)
    # Remove the height dimension from predictions
    raw_predictions = raw_predictions[0]
    classes = torch.argmax(raw_predictions, axis=1).cpu().numpy()

    # Some of the observations are spaces
    # We can kind of cheat here, since the spaces are blank, and just check image values
    # We also need to add a 0 to the end of all predictions for the space character
    observations = []
    for idx, prediction in enumerate(classes):
        patch = img[0, 0, :, idx*28:(idx+1)*28]
        observation = np.zeros(27)
        if torch.max(patch) < 0.5:
            # This is a space character
            observation[-1] = 1.0
        else:
            observation[prediction] = 1
        observations.append(observation.reshape(1, 27))
    observations = np.concat(observations)
    
    original_prediction = ''.join([idxToAlpha(ob) for ob in np.argmax(observations, axis=1)])
    print(f"DNN raw: {original_prediction}")

    log_A = np.log(A + 10e-6)
    log_B = np.log(B + 10e-6)
    log_pi = np.log(pi)

    log_alpha, log_likelihood = updated_hmm.forward_pass_log(log_pi, log_A, log_B, observations)
    viterbi_path = np.argmax(log_alpha, axis=1)
    viterbi_prediction = ''.join([idxToAlpha(ob) for ob in viterbi_path])
    print(f"Viterbi: {viterbi_prediction}")
    log_beta = updated_hmm.backward_pass_log(log_A, log_B, observations)
    log_gamma, log_xi = updated_hmm.compute_expectations_log(log_alpha, log_beta, log_A, log_B, observations, log_likelihood)

    smooth_prediction = ''.join([idxToAlpha(ob) for ob in np.argmax(log_gamma, axis=1)])
    print(f"MLE (smooth): {smooth_prediction}")

    # The final output is the one that will be scored
    print(smooth_prediction)

