import torch
import torchvision.transforms as transforms
import torchvision.datasets as dset

def create_data_loader(args):
    # Asegúrate de que el tamaño de la imagen sea 16x16 o 32x32
    if args.image_size not in [16, 32]:
        raise ValueError("El tamaño de la imagen debe ser 16x16 o 32x32")

    dataset = dset.ImageFolder(root=args.dataroot,
        transform=transforms.Compose([
            transforms.Resize(args.image_size),  # Usa el tamaño de imagen especificado en args.image_size
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]))

    # Configuración del DataLoader
    dataloader = torch.utils.data.DataLoader(dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers
    )

    return dataloader