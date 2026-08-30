# Udacity Image Classifier

This project is part of Udacity's **Future AWS Programmer** Nanodegree. It uses pretrained CNN model architectures (ResNet, AlexNet, and VGG) to classify images of pets, determine whether each image is of a dog or not, and — for dog images — identify the breed. The goal is to compare the three architectures and determine which one performs best across both objectives.

## Project Objectives

1. Identify which pet images are of dogs and which are not.
2. For images that are of dogs, correctly classify the breed.

## How It Works

- `get_pet_labels.py` extracts the "true" pet label from each image filename (e.g. `Boston_terrier_02259.jpg` → `boston terrier`).
- `classify_images.py` runs each image through the selected pretrained CNN model (via `classifier.py`) and compares the model's predicted label to the true pet label.
- `adjust_results4_isadog.py` cross-references both the pet label and the classifier label against `dognames.txt` to determine whether each is a dog.
- `calculates_results_stats.py` computes summary statistics (match rate, dog/not-dog accuracy, breed accuracy) from the results.
- `print_results.py` prints a summary of the results, along with details of any incorrect dog/not-dog assignments or incorrect breed assignments.
- `check_images.py` ties everything together and is the main entry point.

## Usage

Run a single model:

```
python check_images.py --dir pet_images/ --arch vgg --dogfile dognames.txt
```

`--arch` can be `resnet`, `alexnet`, or `vgg`.

Run all three models at once and save each output to a file:

```
sh run_models_batch.sh
```

## Results

| Model   | % Match | % Correct Dogs | % Correct Breed | % Correct Not-a-Dog |
|---------|---------|-----------------|-------------------|------------------------|
| VGG     | 87.5%   | 100.0%          | 93.3%             | 100.0%                 |
| ResNet  | 82.5%   | 100.0%          | 90.0%             | 90.0%                  |
| AlexNet | 75.0%   | 100.0%          | 80.0%             | 100.0%                 |

**VGG** was the best-performing architecture overall. It was one of two models (along with AlexNet) that achieved perfect 100% accuracy on distinguishing dog images from non-dog images, and it had the highest breed classification accuracy at 93.3%. ResNet came closest on breed accuracy (90.0%) but incorrectly classified one non-dog image (a cat) as a dog breed, bringing its not-a-dog accuracy down to 90.0%. AlexNet, while perfect on dog/not-dog classification, had the weakest breed accuracy of the three at 80.0%.

## Project Structure

```
data/
├── pet_images/                     # Test images
├── dognames.txt                    # Valid dog breed names
├── imagenet1000_clsid_to_human.txt # ImageNet class labels
├── classifier.py                   # Pretrained CNN wrapper (provided)
├── get_input_args.py               # Command-line argument parsing
├── get_pet_labels.py               # Extracts pet labels from filenames
├── classify_images.py              # Runs classifier, compares labels
├── adjust_results4_isadog.py       # Flags dog vs. not-dog for both labels
├── calculates_results_stats.py     # Computes summary statistics
├── print_results.py                # Prints final results summary
├── check_images.py                 # Main program
└── run_models_batch.sh             # Runs all three models in sequence
```

## Requirements

- Python 3
- PyTorch / torchvision

## Acknowledgements

Project template and starter files provided by [Udacity](https://www.udacity.com/) as part of the AI Programming with Python Nanodegree.
