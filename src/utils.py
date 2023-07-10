import torch
from .models import CustomModel

def save_checkpoint(epoch, model, optimizer, filename):
    checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
    torch.save(checkpoint, filename)

def get_model(args):
    model = CustomModel(args.model_name, num_classes=args.num_classes)
    if args.resume:
        checkpoint = torch.load(args.resume,
                                    map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['model_state_dict'])
    return model