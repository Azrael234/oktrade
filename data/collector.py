import logging
import time
import datetime
from api.okx_client import OKXClient
from database.db_schema import DatabaseManager

class DataCollector:
    def __init__(self, db_manager, okx_client, inst_types=None):
        """
        数据采集器
        
        参数:
            db_manager (DatabaseManager): 数据库管理器
            okx_client (OKXClient): OKX API客户端
            inst_types (list): 要采集的产品类型列表，默认为["SPOT"]
        """
        self.db_manager = db_manager
        self.okx_client = okx_client
        self.inst_types = inst_types or ["SPOT"]
        self.logger = logging.getLogger(__name__)
        
    def collect_instruments(self):
        """
        采集交易对信息
        
        返回:
            int: 采集的交易对数量
        """
        total_count = 0
        
        for inst_type in self.inst_types:
            self.logger.info(f"开始采集 {inst_type} 类型的交易对信息")
            
            try:
                instruments = self.okx_client.get_instruments(inst_type)
                
                if instruments:
                    self.db_manager.save_instruments(instruments)
                    total_count += len(instruments)
                    self.logger.info(f"成功采集 {len(instruments)} 个 {inst_type} 类型的交易对")
                else:
                    self.logger.warning(f"未获取到 {inst_type} 类型的交易对信息")
                    
            except Exception as e:
                self.logger.error(f"采集 {inst_type} 类型的交易对信息失败: {str(e)}")
        
        return total_count
    
    def collect_kline_data_by_date(self, inst_id, start_date, end_date, batch_size=2000):
        """
        按日期范围采集指定交易对的K线数据
        
        参数:
            inst_id (str): 交易对ID
            start_date (datetime): 开始日期
            end_date (datetime): 结束日期
            batch_size (int): 每批处理的数据量
            
        返回:
            int: 采集的K线数据条数
        """
        self.logger.info(f"开始采集 {inst_id} 的K线数据，时间范围: {start_date} 到 {end_date}")
        
        # 转换为毫秒时间戳
        start_ts = int(start_date.timestamp() * 1000)
        end_ts = int(end_date.timestamp() * 1000)
        
        # 调用现有方法，复用逻辑
        return self._collect_kline_data_by_timestamp(inst_id, start_ts, end_ts, batch_size)
    
    def _collect_kline_data_by_timestamp(self, inst_id, start_ts, end_ts, batch_size=2000, skip_duplicates=False):
        """
        按时间戳范围采集指定交易对的K线数据（内部方法）
        
        参数:
            inst_id (str): 交易对ID
            start_ts (int): 开始时间戳（毫秒）
            end_ts (int): 结束时间戳（毫秒）
            batch_size (int): 每批处理的数据量
            
        返回:
            int: 采集的K线数据条数
        """
        self.logger.info(f"采集时间戳范围: {start_ts} 到 {end_ts}")
        
        # 分批获取和保存数据
        total_saved = 0
        current_end_ts = end_ts
        batch_klines = []
        
        try:
            while current_end_ts > start_ts:
                # 获取K线数据
                kline_batch = self.okx_client.get_kline(inst_id, start_ts, current_end_ts)
                
                if not kline_batch:
                    self.logger.warning(f"未获取到 {inst_id} 在时间戳 {start_ts} 到 {current_end_ts} 的K线数据")
                    break
                
                # 添加到当前批次
                batch_klines.extend(kline_batch)
                self.logger.info(f"已获取 {len(batch_klines)} 条K线数据")
                
                # 如果当前批次达到了指定大小，保存并清空
                if len(batch_klines) >= batch_size:
                    saved_count = self.db_manager.save_kline_data(inst_id, batch_klines, skip_duplicates)
                    total_saved += saved_count
                    self.logger.info(f"已保存 {total_saved} 条K线数据到数据库")
                    batch_klines = []  # 清空当前批次
                
                # 更新结束时间戳为当前批次中最早的K线的时间戳
                if kline_batch:
                    # 获取最早一条K线的时间戳，减去1毫秒作为下一批次的结束时间戳
                    earliest_ts = int(kline_batch[-1][0])
                    current_end_ts = earliest_ts - 1
                else:
                    break
            
            # 保存最后一批数据（如果有）
            if batch_klines:
                saved_count = self.db_manager.save_kline_data(inst_id, batch_klines, skip_duplicates)
                total_saved += saved_count
                self.logger.info(f"已保存最后 {saved_count} 条K线数据到数据库")
            
            self.logger.info(f"成功采集并保存 {total_saved} 条 {inst_id} 的K线数据")
            return total_saved
            
        except Exception as e:
            # 发生异常时，尝试保存已采集的数据
            self.logger.error(f"采集K线数据异常: {str(e)}")
            
            if batch_klines:
                try:
                    saved_count = self.db_manager.save_kline_data(inst_id, batch_klines)
                    total_saved += saved_count
                    self.logger.info(f"异常发生前已保存 {total_saved} 条K线数据到数据库")
                except Exception as save_error:
                    self.logger.error(f"保存已采集数据失败: {str(save_error)}")
            
            return total_saved
    
    def collect_kline_data(self, inst_id, days=1, batch_size=2000):
        """
        采集指定交易对的K线数据
        
        参数:
            inst_id (str): 交易对ID
            days (int): 采集多少天的数据
            batch_size (int): 每批处理的数据量
            
        返回:
            int: 采集的K线数据条数
        """
        # 计算开始和结束时间
        end_time = datetime.datetime.now()
        start_time = end_time - datetime.timedelta(days=days)
        
        # 转换为毫秒时间戳
        start_ts = int(start_time.timestamp() * 1000)
        end_ts = int(end_time.timestamp() * 1000)
        
        self.logger.info(f"采集时间范围: {start_time} 到 {end_time}")
        
        # 调用内部方法
        return self._collect_kline_data_by_timestamp(inst_id, start_ts, end_ts, batch_size)
    
    def collect_all_kline_data_by_date(self, inst_type="SPOT", start_date=None, end_date=None, limit=None, batch_size=2000):
        """
        按日期范围采集指定类型的所有交易对的K线数据
        
        参数:
            inst_type (str): 产品类型
            start_date (datetime): 开始日期
            end_date (datetime): 结束日期
            limit (int): 限制采集的交易对数量
            batch_size (int): 每批处理的数据量
            
        返回:
            int: 成功采集的交易对数量
        """
        self.logger.info(f"开始采集 {inst_type} 类型的所有交易对的K线数据，时间范围: {start_date} 到 {end_date}")
        
        # 获取所有交易对
        instruments = self.db_manager.get_all_instruments(inst_type)
        
        if not instruments:
            self.logger.warning(f"未找到 {inst_type} 类型的交易对，请先采集交易对信息")
            return 0
        
        # 限制交易对数量
        if limit and limit < len(instruments):
            instruments = instruments[:limit]
            self.logger.info(f"限制采集前 {limit} 个交易对")
        
        self.logger.info(f"共有 {len(instruments)} 个交易对需要采集")
        
        # 采集每个交易对的K线数据
        success_count = 0
        for i, inst_id in enumerate(instruments):
            try:
                self.logger.info(f"[{i+1}/{len(instruments)}] 开始采集 {inst_id} 的K线数据")
                count = self.collect_kline_data_by_date(inst_id, start_date, end_date, batch_size)
                
                if count > 0:
                    success_count += 1
                    self.logger.info(f"成功采集 {inst_id} 的 {count} 条K线数据")
                else:
                    self.logger.warning(f"未采集到 {inst_id} 的K线数据")
                    
            except Exception as e:
                self.logger.error(f"采集 {inst_id} 的K线数据失败: {str(e)}")
        
        self.logger.info(f"共成功采集 {success_count}/{len(instruments)} 个交易对的K线数据")
        return success_count