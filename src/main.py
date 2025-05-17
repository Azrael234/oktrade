import argparse
from train import train_transformer, train_tcn, train_lstm
from test import test_model
from predict import predict

def main():
    parser = argparse.ArgumentParser(description='OKTrade 模型训练、测试和预测')
    
    # 添加子命令
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # 训练命令
    train_parser = subparsers.add_parser('train', help='训练模型')
    train_parser.add_argument('model_type', choices=['transformer', 'tcn', 'lstm'], 
                            help='选择要训练的模型类型')
    train_parser.add_argument('--experiment', default='default',
                            help='使用的实验配置名称')
    
    # 测试命令
    test_parser = subparsers.add_parser('test', help='测试模型')
    test_parser.add_argument('model_path', help='要测试的模型路径')
    
    # 预测命令
    predict_parser = subparsers.add_parser('predict', help='进行预测')
    predict_parser.add_argument('model_path', help='用于预测的模型路径')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        if args.model_type == 'transformer':
            train_transformer(experiment_name=args.experiment)
        elif args.model_type == 'tcn':
            train_tcn(experiment_name=args.experiment)
        elif args.model_type == 'lstm':
            train_lstm(experiment_name=args.experiment)
    elif args.command == 'test':
        test_model(args.model_path)
    elif args.command == 'predict':
        predict(args.model_path)

if __name__ == '__main__':
    main()