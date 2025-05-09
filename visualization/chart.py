import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import os
from data.aggregator import KlineAggregator

class ChartGenerator:
    """
    图表生成器
    """
    def __init__(self, db_manager):
        """
        初始化图表生成器
        
        参数:
            db_manager: 数据库管理器
        """
        self.db_manager = db_manager
        self.kline_aggregator = KlineAggregator(db_manager)
        
    def plot_kline(self, inst_id, start_time, end_time, timeframe='1h', save_path=None):
        """
        绘制K线图
        
        参数:
            inst_id: 交易对ID
            start_time: 开始时间
            end_time: 结束时间
            timeframe: 时间周期
            save_path: 保存路径
            
        返回:
            matplotlib.figure.Figure: 图表对象
        """
        # 获取K线数据
        df = self.kline_aggregator.get_aggregated_kline(
            inst_id, start_time, end_time, timeframe
        )
        
        if df.empty:
            print(f"未获取到 {inst_id} 在指定时间范围内的数据")
            return None
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 绘制K线
        width = 0.6
        width2 = 0.05
        
        # 上涨
        up = df[df.close >= df.open]
        # 下跌
        down = df[df.close < df.open]
        
        # 绘制K线实体
        ax.bar(up.index, up.close - up.open, width, bottom=up.open, color='red')
        ax.bar(down.index, down.close - down.open, width, bottom=down.open, color='green')
        
        # 绘制上下影线
        ax.bar(up.index, up.high - up.close, width2, bottom=up.close, color='red')
        ax.bar(up.index, up.low - up.open, width2, bottom=up.open, color='red')
        ax.bar(down.index, down.high - down.open, width2, bottom=down.open, color='green')
        ax.bar(down.index, down.low - down.close, width2, bottom=down.close, color='green')
        
        # 设置图表标题和标签
        ax.set_title(f'{inst_id} {timeframe} K线图')
        ax.set_xlabel('时间')
        ax.set_ylabel('价格')
        
        # 设置x轴日期格式
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        
        # 自动调整日期标签
        fig.autofmt_xdate()
        
        # 显示网格
        ax.grid(True)
        
        # 保存图表
        if save_path:
            plt.savefig(save_path)
        
        return fig
    
    def plot_ma_strategy(self, model, start_time, end_time, save_path=None):
        """
        绘制移动平均线策略图表
        
        参数:
            model: 移动平均线模型
            start_time: 开始时间
            end_time: 结束时间
            save_path: 保存路径
            
        返回:
            matplotlib.figure.Figure: 图表对象
        """
        # 获取数据
        df = model.get_data(start_time, end_time)
        if df is None:
            return None
            
        # 预处理数据
        df = model.preprocess_data(df)
        
        # 生成交易信号
        df = model.generate_signals(df)
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # 绘制价格和移动平均线
        ax1.plot(df.index, df['close'], label='价格')
        ax1.plot(df.index, df[f'MA{model.short_window}'], label=f'{model.short_window}日均线')
        ax1.plot(df.index, df[f'MA{model.long_window}'], label=f'{model.long_window}日均线')
        
        # 标记买入和卖出信号
        buy_signals = df[df['position'] == 1]
        sell_signals = df[df['position'] == -1]
        
        ax1.scatter(buy_signals.index, buy_signals['close'], marker='^', color='g', label='买入信号')
        ax1.scatter(sell_signals.index, sell_signals['close'], marker='v', color='r', label='卖出信号')
        
        # 设置图表标题和标签
        ax1.set_title(f'{model.inst_id} 移动平均线交叉策略 ({model.short_window}/{model.long_window})')
        ax1.set_ylabel('价格')
        ax1.legend()
        ax1.grid(True)
        
        # 绘制持仓
        ax2.fill_between(df.index, df['signal'], 0, alpha=0.5, color='g', where=df['signal'] > 0)
        ax2.fill_between(df.index, df['signal'], 0, alpha=0.5, color='r', where=df['signal'] < 0)
        
        # 设置y轴范围
        ax2.set_ylim([-1.1, 1.1])
        ax2.set_yticks([-1, 0, 1])
        ax2.set_yticklabels(['做空', '空仓', '做多'])
        
        # 设置x轴日期格式
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
        
        # 设置标签
        ax2.set_xlabel('时间')
        ax2.set_ylabel('持仓')
        ax2.grid(True)
        
        # 自动调整日期标签
        fig.autofmt_xdate()
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        if save_path:
            plt.savefig(save_path)
        
        return fig
    
    def plot_backtest_results(self, results, save_path=None):
        """
        绘制回测结果
        
        参数:
            results: 回测结果
            save_path: 保存路径
            
        返回:
            matplotlib.figure.Figure: 图表对象
        """
        if results is None:
            return None
            
        portfolio = results['portfolio']
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 绘制资产曲线
        ax1.plot(portfolio.index, portfolio['total'], label='总资产')
        ax1.plot(portfolio.index, portfolio['holdings'], label='持仓价值')
        ax1.plot(portfolio.index, portfolio['cash'], label='现金')
        
        # 设置图表标题和标签
        ax1.set_title('回测结果 - 资产曲线')
        ax1.set_ylabel('资产价值')
        ax1.legend()
        ax1.grid(True)
        
        # 绘制回撤曲线
        drawdown = (portfolio['total'] / portfolio['total'].cummax() - 1) * 100
        ax2.fill_between(drawdown.index, drawdown, 0, alpha=0.5, color='r')
        
        # 设置图表标题和标签
        ax2.set_title('回测结果 - 回撤曲线')
        ax2.set_ylabel('回撤 (%)')
        ax2.set_xlabel('时间')
        ax2.grid(True)
        
        # 设置x轴日期格式
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
        
        # 自动调整日期标签
        fig.autofmt_xdate()
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        if save_path:
            plt.savefig(save_path)
        
        return fig