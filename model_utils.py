import os
import torch
from torchviz import make_dot

def save_model(model, epoch, directory="../models"):
    """
    Save the model to the specified directory, overwriting any existing file with the same name.
    Args:
    - model (torch.nn.Module): The model to be saved.
    - epoch (int): The current epoch number.
    - directory (str): The directory to save the model.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    model_type = type(model).__name__
    save_path = os.path.join(directory, f"{model_type}.pt")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict()
    }, save_path)
    print(f"Model saved to {save_path}")

def load_model(model, directory="../models"):
    """
    Load the most recent saved model from the specified directory.
    Args:
    - model (torch.nn.Module): The model to load the state dictionary into.
    - directory (str): The directory to load the model from.
    """
    model_type = type(model).__name__
    load_path = os.path.join(directory, f"{model_type}.pt")
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"No saved model found at {load_path}")
    else:
        print(f"Loading model {model_type} from {load_path}")
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Model loaded from {load_path} (epoch {epoch})")
    return model, epoch

def save_model_visualization(model, model_name, input_tensor, directory="../models"):
    """
    Save the model architecture visualization.
    
    Args:
    - model (torch.nn.Module): The model to visualize.
    - model_name (str): The name of the model.
    - input_tensor (torch.Tensor): A dummy input tensor to pass through the model for visualization.
    - directory (str): Directory to save the visualization.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Generate the visualization
    y = model(input_tensor)
    dot = make_dot(y, params=dict(model.named_parameters()))

    # Save the visualization
    save_path = os.path.join(directory, f"{model_name}_architecture")
    dot.format = 'png'
    dot.render(save_path)
    print(f"Model architecture visualization saved to {save_path}.png")