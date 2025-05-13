config = {
    'batch_size': 32,
    'model_dim': 128,
    'num_heads': 8,
    'num_layers': 4,
    'lr': 0.0001,
    'epochs': 50,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}