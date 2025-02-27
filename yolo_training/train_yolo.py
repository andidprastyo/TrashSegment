import argparse
from ultralytics import YOLO
import torch
import shutil
import os

def main(args):    
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device {device_type}')

    # Identify split based on train size
    split_id = f'split_{args.train_size}_{100-args.train_size}'
    if args.train_size < 10:
        split_id = f'split_0{args.train_size}_{100-args.train_size}'
    
    # Create split folder based on selected locations used. Remove duplicates.
    locations_used = list(set([int(x) for x in args.locations.split(' ')]))
    if len(locations_used) > 6 or max(locations_used) > 6 or min(locations_used) < 1:
        raise ValueError('Invalid selection of locations. Should only use numbers between 1 and 6.')
    print(f'Using locations {locations_used}')

    # If we use all locations, we can use an existing split. Otherwise, we create a custom split
    # containing only the images and annotations of the selected locations
    if len(locations_used) == 6:
        split_folder_path = f'../data/annotations/{split_id}' 
    else:
        split_folder_path = create_location_split(split_id, locations_used)

    # Create dataset yaml file
    data_path = './dataset.yaml'
    detection_classes = {
        "1": "floating_trash",
        "2": "water",
        "3": "barrier"
    }
    with open(data_path, 'w') as f:
        f.write(f'path: {split_folder_path}\n')
        f.write('train: train/images\n')
        f.write('val: test/images\n')
        f.write('\n')
        f.write(f'nc: {str(len(detection_classes.keys()))}\n')
        f.write('\n')
        f.write(f'names: {[x for x in detection_classes.values()]}')

    # Set experiment name -- 'split-experiment_name-locations'
    run_name = get_run_name(split_id, args.locations, args.experiment_name)

    # Load a model
    yolo_model = YOLO(args.model_version)

    # # Train
    print("Starting training")
    _ = yolo_model.train(data=data_path, epochs=int(args.epochs), imgsz=int(args.img_size), batch=int(args.batch_size),
                project='training_files', name=run_name, device=device_type)

def get_run_name(split_id, locations, exp_name):
    run_name = f'{split_id}-'
    run_name += exp_name + '-'
    run_name += locations.replace(' ', '')
    return run_name

def create_location_split(split_id, target_locations):
    """
    Select images from the right split and locations for training and put them into a seperate split folder.
    """
    # Set the current split to be created and the target split directory
    current_split_path = f'../data/annotations/{split_id}_{"".join([str(x) for x in target_locations])}'
    target_split_path = f'../data/annotations/{split_id}'

    # Create 'current_split' directory and create train/test folders
    os.makedirs(current_split_path, exist_ok=True)
    [os.makedirs(x, exist_ok=True) for x in [current_split_path, 
        f'{current_split_path}/train/images', f'{current_split_path}/train/labels',
        f'{current_split_path}/test/images', f'{current_split_path}/test/labels']]
    
    for location in os.listdir('../data/images'):
        # Only check folders of locations we're interested in 
        location_id = int(location[-1])
        if location_id not in target_locations:
            continue
        
        # Move the image and corresponding label file to correct folder
        for location_img in os.listdir(f'../data/images/{location}'):
            traintest = 'train' if location_img in os.listdir(f'{target_split_path}/train/images') else 'test'
            label_img_id = location_img.replace('.jpg', '.txt')
            shutil.copy(f'{target_split_path}/{traintest}/images/{location_img}', f'{current_split_path}/{traintest}/images/{location_img}')
            shutil.copy(f'{target_split_path}/{traintest}/labels/{label_img_id}', f'{current_split_path}/{traintest}/labels/{label_img_id}')

    # Debugging
    for traintest in ['train', 'test']:
        n_images, n_labels = len(os.listdir(f"{current_split_path}/{traintest}/images")), len(os.listdir(f"{current_split_path}/{traintest}/labels"))
        print(f'{traintest} images: {n_images}, labels: {n_labels}')
    
    return current_split_path.replace('../', './')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # General settings
    parser.add_argument('--split_folder', help='Path to folder containing the splits', default='./data/annotations')
    parser.add_argument('--train_size', type=int, default=80, help='Determine size of split to train on.')
    parser.add_argument('--experiment_name', default='', help='Helps identify experiment')
    parser.add_argument('--locations', help='Ids of locations to train on, seperate by a space', type=str, default='1 2 3 4 5 6')

    # Training Hyperparameters
    parser.add_argument('--model_version', default='yolov8n-seg.pt', help='training model version to use')
    parser.add_argument('--batch_size', default=4, help='batch size for training. depends on size of gpu/cpu and image size')
    parser.add_argument('--img_size', default=1280, help='image size for training')
    parser.add_argument('--epochs', default=1, help='number of epochs used for training')

    args = parser.parse_args()
    main(args)
