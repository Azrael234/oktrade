import os
import json
import logging
import argparse
import datetime
from train import train_model  # 改为导入统一训练函数
from evaluate import evaluate_model, compare_models  # 更新评估函数
from database.db_schema import DatabaseManager
from typing import Literal

# def setup_logging():
#     """日志设置保持不变"""
#     log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
#     os.makedirs(log_dir, exist_ok=True)
#     log_file = os.path.join(log_dir, f'oktrade_models_{datetime.datetime.now().strftime("%Y%m%d")}.log')
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#         handlers=[
#             logging.FileHandler(log_file),
#             logging.StreamHandler()
#         ]
#     )
def setup_logging(log_file=None, experiment=None):
    """设置日志系统，支持指定日志文件"""
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)

    # 如果有实验名称，创建实验子目录
    if experiment:
        log_dir = os.path.join(log_dir, experiment)
    
    os.makedirs(log_dir, exist_ok=True)

    # 如果没有指定日志文件，则使用默认的按天日志
    if log_file is None:
        log_file = os.path.join(log_dir, f'oktrade_models_{datetime.datetime.now().strftime("%Y%m%d")}.log')
    else:
        # 如果指定了日志文件，确保路径完整
        log_file = os.path.join(log_dir, log_file)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

# def generate_experiment_log_path(args, filename):
#     log_dir = os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs'), args.experiment)
#     os.makedirs(log_dir, exist_ok=True)
#     return os.path.join(log_dir, filename)

def load_config():
    """加载配置保持不变"""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../config_moni.json')
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"加载配置文件失败: {str(e)}")
        return {}

def main():
    # setup_logging()
    # logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser(description='OKTrade 模型训练、测试和预测')
    
    # 添加子命令
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # 训练命令
    train_parser = subparsers.add_parser('train', help='训练模型')
    train_parser.add_argument('model_type', 
                            choices=['transformer', 'tcn', 'lstm', 'informer', 'autoformer'],
                            help='选择要训练的模型类型')
    train_parser.add_argument('--inst', default='BTC-USDT',
                            help='交易对ID，默认BTC-USDT')
    train_parser.add_argument('--mode', choices=['regression', 'classification'], default='regression',
                            help='训练模式: regression(默认)或classification')
    train_parser.add_argument('--experiment', default='default',
                            help='使用的实验配置名称')
    
    # 测试命令
    test_parser = subparsers.add_parser('test', help='测试模型')
    test_parser.add_argument('--model_path', default="", help='要测试的模型路径')
    test_parser.add_argument('--experiment', required=True,
                           help='实验名称(必需)')
    test_parser.add_argument('--mode', choices=['regression', 'classification'], default='regression',
                           help='测试模式: regression(默认)或classification')
    
    # 比较命令
    compare_parser = subparsers.add_parser('compare', help='比较多个模型')
    compare_parser.add_argument('--model_paths', nargs='+', required=True,
                              help='要比较的模型路径列表(必需)，例如 --model_paths path1 path2 path3')
    compare_parser.add_argument('--experiment', required=True,
                              help='实验名称(必需)')
    compare_parser.add_argument('--inst', default='BTC-USDT',
                              help='交易对ID，默认BTC-USDT')
    compare_parser.add_argument('--mode', choices=['regression', 'classification'], default='regression',
                              help='比较模式: regression(默认)或classification')
    
    args = parser.parse_args()
    config = load_config()
    
    # 创建数据库管理器
    db_config = config.get('database', {})
    db_manager = DatabaseManager(
        host=db_config.get('host', 'localhost'),
        database=db_config.get('database', 'oktrade'),
        user=db_config.get('user', 'postgres'),
        password=db_config.get('password', 'postgres'),
        port=db_config.get('port', 5432)
    )
    
    # 打印交易对信息
    instruments = db_manager.get_all_instruments()
    print(f"交易对信息（共{len(instruments)}个）：")
    for inst in instruments[:3]:  # 只打印前3个
        print(f" - {inst}")
    
    logging.shutdown()
    if args.command == 'train':
        # 为本次实验生成专属日志文件
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"{timestamp}_train_{args.inst}_{args.model_type}_{args.mode}.log"
        # log_file = generate_experiment_log_path(args, log_filename)
        setup_logging(log_file=log_filename, experiment=args.experiment)  # 使用指定的日志文件和实验名称

        logger = logging.getLogger(__name__)
        logger.info("🚀 开始训练任务")
        logger.info(f"模型类型: {args.model_type}")
        logger.info(f"交易对: {args.inst}")
        logger.info(f"任务模式: {args.mode}")
        train_model(
            db_manager=db_manager,
            inst_id=args.inst,
            model_type=args.model_type,
            mode=args.mode,
            experiment_name=args.experiment,
            log_filename = log_filename
        )
    elif args.command == 'test':
        # 测试日志设置
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"{timestamp}_test_{args.mode}.log"
        log_file = setup_logging(log_filename, args.experiment)
        logger = logging.getLogger(__name__)
        logger.info("🚀 开始测试任务")
        logger.info(f"模型路径: {args.model_path}")
        logger.info(f"任务模式: {args.mode}")
        evaluate_model(
            db_manager=db_manager,
            model_path=args.model_path,
            mode=args.mode
        )
    elif args.command == 'compare':
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"{timestamp}_compare_{args.inst}_{args.mode}.log"
        log_file = setup_logging(log_filename, args.experiment)
        logger = logging.getLogger(__name__)
        logger.info("🚀 开始比较任务")
        logger.info(f"实验名称: {args.experiment}")
        logger.info(f"交易对: {args.inst}")
        logger.info(f"任务模式: {args.mode}")
        logger.info(f"比较模型列表: {args.model_paths}")
        compare_models(
            db_manager=db_manager,
            model_paths=args.model_paths,
            inst_id=args.inst,
            mode=args.mode,
            config=config
        )

if __name__ == '__main__':
    main()