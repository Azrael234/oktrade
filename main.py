import os
import json
import logging
import argparse
import datetime
from api.okx_client import OKXClient
from database.db_schema import DatabaseManager
from data.collector import DataCollector
from data.aggregator import KlineAggregator
from models.ma_cross_model import MACrossModel
from visualization.chart import ChartGenerator

def setup_logging():
    """
    设置日志
    """
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f'oktrade_{datetime.datetime.now().strftime("%Y%m%d")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def load_config():
    """
    加载配置文件
    """
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
    
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"加载配置文件失败: {str(e)}")
        return {}

def main():
    """
    主函数
    """
    # 设置日志
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='OKX交易数据采集和分析工具')
    
    # 数据采集参数
    parser.add_argument('--collect-instruments', action='store_true', help='采集交易对信息')
    parser.add_argument('--collect-kline', action='store_true', help='采集K线数据')
    parser.add_argument('--inst-type', type=str, default='SPOT', help='产品类型')
    parser.add_argument('--inst-id', type=str, help='产品ID')
    parser.add_argument('--days', type=int, default=1, help='采集多少天的数据')
    parser.add_argument('--limit', type=int, help='限制采集的交易对数量')
    parser.add_argument('--use-proxy', action='store_true', help='使用代理')
    parser.add_argument('--start-date', type=str, help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('--batch-size', type=int, default=2000, help='批量处理的数据量')
    parser.add_argument('--skip-duplicates', action='store_true', help='跳过重复数据')
    
    # 数据可视化参数
    parser.add_argument('--plot-kline', action='store_true', help='绘制K线图')
    parser.add_argument('--timeframe', type=str, default='1h', help='时间周期')
    parser.add_argument('--save-chart', type=str, help='保存图表路径')
    
    # 模型回测参数
    parser.add_argument('--backtest', action='store_true', help='回测交易策略')
    parser.add_argument('--model', type=str, default='ma_cross', help='交易模型')
    parser.add_argument('--short-window', type=int, default=20, help='短期移动平均线窗口')
    parser.add_argument('--long-window', type=int, default=50, help='长期移动平均线窗口')
    parser.add_argument('--initial-capital', type=float, default=10000.0, help='初始资金')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config()
    
    # 创建OKX客户端
    okx_client = OKXClient(
        api_key=config.get('api_key'),
        secret_key=config.get('secret_key'),
        passphrase=config.get('passphrase'),
        use_proxy=args.use_proxy,
        proxy_host=config.get('proxy_host', '127.0.0.1'),
        proxy_port=config.get('proxy_port', 10808)
    )
    
    # 创建数据库管理器
    db_config = config.get('database', {})
    db_manager = DatabaseManager(
        host=db_config.get('host', 'localhost'),
        database=db_config.get('database', 'oktrade'),
        user=db_config.get('user', 'postgres'),
        password=db_config.get('password', 'postgres'),
        port=db_config.get('port', 5432)
    )
    
    # 初始化数据库
    try:
        db_manager.initialize_database()
    except Exception as e:
        logger.error(f"初始化数据库失败: {str(e)}")
        return
    
    # 创建数据采集器
    collector = DataCollector(db_manager, okx_client)
    
    # 采集交易对信息
    if args.collect_instruments:
        logger.info("开始采集交易对信息")
        count = collector.collect_instruments()
        logger.info(f"成功采集 {count} 个交易对信息")
    
    # 采集K线数据
    if args.collect_kline:
        if args.inst_id:
            logger.info(f"开始采集 {args.inst_id} 的K线数据")
            count = collector.collect_kline_data(args.inst_id, args.days)
            logger.info(f"成功采集 {count} 条K线数据")
        else:
            logger.info(f"开始采集 {args.inst_type} 类型的所有交易对的K线数据")
            count = collector.collect_all_kline_data(args.inst_type, args.days, args.limit)
            logger.info(f"成功采集 {count} 个交易对的K线数据")
    
    # 解析日期参数
    start_date = None
    end_date = None
    
    if args.start_date:
        try:
            start_date = datetime.datetime.strptime(args.start_date, '%Y-%m-%d')
        except ValueError:
            logger.error(f"无效的开始日期格式: {args.start_date}，应为YYYY-MM-DD")
            return
    
    if args.end_date:
        try:
            end_date = datetime.datetime.strptime(args.end_date, '%Y-%m-%d')
        except ValueError:
            logger.error(f"无效的结束日期格式: {args.end_date}，应为YYYY-MM-DD")
            return
    
    # 采集K线数据
    if args.collect_kline:
        # 如果指定了日期范围，使用日期范围而不是days参数
        if start_date and end_date:
            days = None
            logger.info(f"使用指定的日期范围: {start_date} 到 {end_date}")
        else:
            days = args.days
            # 如果未指定日期，使用默认值
            if not start_date:
                start_date = datetime.datetime.now() - datetime.timedelta(days=days)
            if not end_date:
                end_date = datetime.datetime.now()
        
        if args.inst_id:
            logger.info(f"开始采集 {args.inst_id} 的K线数据")
            if days:
                count = collector.collect_kline_data(args.inst_id, days, args.batch_size)
            else:
                count = collector.collect_kline_data_by_date(args.inst_id, start_date, end_date, args.batch_size)
            logger.info(f"成功采集 {count} 条K线数据")
        else:
            logger.info(f"开始采集 {args.inst_type} 类型的所有交易对的K线数据")
            if days:
                count = collector.collect_all_kline_data(args.inst_type, days, args.limit, args.batch_size)
            else:
                count = collector.collect_all_kline_data_by_date(args.inst_type, start_date, end_date, args.limit, args.batch_size)
            logger.info(f"成功采集 {count} 个交易对的K线数据")
    
    # 如果未指定日期，使用默认值（用于可视化和回测）
    if not start_date:
        start_date = datetime.datetime.now() - datetime.timedelta(days=30)
    if not end_date:
        end_date = datetime.datetime.now()
    
    # 绘制K线图
    if args.plot_kline:
        if not args.inst_id:
            logger.error("绘制K线图需要指定交易对ID")
            return
            
        logger.info(f"绘制 {args.inst_id} 的K线图")
        
        chart_generator = ChartGenerator(db_manager)
        fig = chart_generator.plot_kline(
            args.inst_id, start_date, end_date, args.timeframe, args.save_chart
        )
        
        if fig:
            logger.info(f"成功绘制 {args.inst_id} 的K线图")
            if args.save_chart:
                logger.info(f"图表已保存到: {args.save_chart}")
            else:
                plt.show()
        else:
            logger.error(f"绘制 {args.inst_id} 的K线图失败")
    
    # 回测交易策略
    if args.backtest:
        if not args.inst_id:
            logger.error("回测交易策略需要指定交易对ID")
            return
            
        logger.info(f"回测 {args.inst_id} 的交易策略")
        
        # 创建交易模型
        if args.model == 'ma_cross':
            model = MACrossModel(
                db_manager, args.inst_id, args.timeframe,
                args.short_window, args.long_window
            )
        else:
            logger.error(f"不支持的交易模型: {args.model}")
            return
        
        # 运行回测
        results = model.run(start_date, end_date, args.initial_capital)
        
        if results:
            # 打印回测结果
            logger.info("回测结果:")
            logger.info(f"总收益率: {results['total_return'] * 100:.2f}%")
            logger.info(f"年化收益率: {results['annual_return'] * 100:.2f}%")
            logger.info(f"夏普比率: {results['sharpe_ratio']:.2f}")
            logger.info(f"最大回撤: {results['max_drawdown'] * 100:.2f}%")
            logger.info(f"交易次数: {results['trades']}")
            logger.info(f"胜率: {results['win_rate'] * 100:.2f}%")
            
            # 绘制回测结果图表
            chart_generator = ChartGenerator(db_manager)
            
            # 绘制策略图表
            strategy_chart_path = None
            if args.save_chart:
                strategy_chart_path = args.save_chart.replace('.png', '_strategy.png')
                
            fig1 = chart_generator.plot_ma_strategy(model, start_date, end_date, strategy_chart_path)
            
            # 绘制回测结果图表
            results_chart_path = None
            if args.save_chart:
                results_chart_path = args.save_chart.replace('.png', '_results.png')
                
            fig2 = chart_generator.plot_backtest_results(results, results_chart_path)
            
            if fig1 and fig2:
                logger.info("成功绘制回测结果图表")
                if args.save_chart:
                    logger.info(f"策略图表已保存到: {strategy_chart_path}")
                    logger.info(f"回测结果图表已保存到: {results_chart_path}")
                else:
                    plt.show()
            else:
                logger.error("绘制回测结果图表失败")
        else:
            logger.error("回测失败")
    
    # 如果没有指定任何操作，显示帮助信息
    if not any([
        args.collect_instruments, args.collect_kline,
        args.plot_kline, args.backtest
    ]):
        parser.print_help()

if __name__ == "__main__":
    main()