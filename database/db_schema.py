import psycopg2
from psycopg2 import sql
import logging

class DatabaseManager:
    def __init__(self, host='localhost', database='oktrade', user='postgres', password='your_password', port=5432):
        """
        数据库管理器
        
        参数:
            host (str): 数据库主机
            database (str): 数据库名
            user (str): 用户名
            password (str): 密码
            port (int): 端口
        """
        self.connection_params = {
            'host': host,
            'database': database,
            'user': user,
            'password': password,
            'port': port
        }
        self.logger = logging.getLogger(__name__)
        
    def connect(self):
        """
        连接到数据库
        
        返回:
            connection: 数据库连接对象
        """
        try:
            conn = psycopg2.connect(**self.connection_params)
            return conn
        except Exception as e:
            self.logger.error(f"数据库连接失败: {str(e)}")
            raise
    
    def initialize_database(self):
        """
        初始化数据库表结构
        """
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            # 创建交易对表
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS instruments (
                id SERIAL PRIMARY KEY,
                inst_id VARCHAR(50) UNIQUE NOT NULL,
                inst_type VARCHAR(20) NOT NULL,
                base_ccy VARCHAR(20),
                quote_ccy VARCHAR(20),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            
            # 创建1分钟K线数据表
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS kline_1m (
                id SERIAL PRIMARY KEY,
                inst_id VARCHAR(50) NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                open NUMERIC(24, 8) NOT NULL,
                high NUMERIC(24, 8) NOT NULL,
                low NUMERIC(24, 8) NOT NULL,
                close NUMERIC(24, 8) NOT NULL,
                volume NUMERIC(24, 8) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (inst_id, timestamp)
            )
            """)
            
            # 创建索引
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_kline_1m_inst_id ON kline_1m (inst_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_kline_1m_timestamp ON kline_1m (timestamp)")
            
            conn.commit()
            self.logger.info("数据库表结构初始化成功")
        except Exception as e:
            conn.rollback()
            self.logger.error(f"数据库表结构初始化失败: {str(e)}")
            raise
        finally:
            cursor.close()
            conn.close()
    
    def save_instruments(self, instruments):
        """
        保存交易对信息
        
        参数:
            instruments (list): 交易对列表
        """
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            for inst in instruments:
                cursor.execute("""
                INSERT INTO instruments (inst_id, inst_type, base_ccy, quote_ccy)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (inst_id) 
                DO UPDATE SET 
                    inst_type = EXCLUDED.inst_type,
                    base_ccy = EXCLUDED.base_ccy,
                    quote_ccy = EXCLUDED.quote_ccy,
                    updated_at = CURRENT_TIMESTAMP
                """, (
                    inst.get('instId'),
                    inst.get('instType'),
                    inst.get('baseCcy'),
                    inst.get('quoteCcy')
                ))
            
            conn.commit()
            self.logger.info(f"成功保存 {len(instruments)} 个交易对")
        except Exception as e:
            conn.rollback()
            self.logger.error(f"保存交易对失败: {str(e)}")
            raise
        finally:
            cursor.close()
            conn.close()
    
    def save_kline_data(self, inst_id, kline_data, skip_duplicates=False):
        """
        保存K线数据
        
        参数:
            inst_id (str): 交易对ID
            kline_data (list): K线数据列表
            skip_duplicates (bool): 是否跳过重复数据
        
        返回:
            int: 成功保存的记录数
        """
        if not kline_data:
            self.logger.warning(f"没有K线数据需要保存")
            return 0
            
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            # 准备批量插入的数据
            batch_data = []
            for kline in kline_data:
                # OKX K线数据格式: [timestamp, open, high, low, close, vol, ...]
                # 确保数据格式正确
                if len(kline) < 6:
                    self.logger.warning(f"K线数据格式不正确: {kline}")
                    continue
                    
                # 转换时间戳为秒
                timestamp_ms = int(kline[0])
                timestamp_sec = timestamp_ms / 1000
                
                # 添加到批量数据中
                batch_data.append((
                    inst_id,
                    timestamp_sec,
                    kline[1],
                    kline[2],
                    kline[3],
                    kline[4],
                    kline[5]
                ))
            
            if not batch_data:
                self.logger.warning(f"没有有效的K线数据需要保存")
                return 0
                
            # 使用psycopg2.extras.execute_values进行批量插入
            from psycopg2.extras import execute_values
            
            # 根据skip_duplicates参数决定冲突处理方式
            if skip_duplicates:
                # 跳过重复数据
                conflict_action = "DO NOTHING"
            else:
                # 更新重复数据
                conflict_action = """
                DO UPDATE SET 
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume,
                    created_at = CURRENT_TIMESTAMP
                """
            
            # 批量插入数据
            execute_values(
                cursor,
                f"""
                INSERT INTO kline_1m (inst_id, timestamp, open, high, low, close, volume)
                VALUES %s
                ON CONFLICT (inst_id, timestamp) 
                {conflict_action}
                """,
                batch_data,
                template="(%s, to_timestamp(%s), %s, %s, %s, %s, %s)",
                page_size=1000  # 每次最多插入1000条
            )
            
            # 提交事务
            conn.commit()
            self.logger.info(f"成功批量保存 {len(batch_data)} 条 {inst_id} 的K线数据")
            return len(batch_data)
            
        except Exception as e:
            conn.rollback()
            self.logger.error(f"批量保存K线数据失败: {str(e)}")
            return 0
        finally:
            cursor.close()
            conn.close()
    
    def get_all_instruments(self, inst_type=None):
        """
        获取所有交易对
        
        参数:
            inst_type (str): 交易对类型，不指定则获取所有类型
            
        返回:
            list: 交易对列表
        """
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            if inst_type:
                cursor.execute("SELECT inst_id FROM instruments WHERE inst_type = %s", (inst_type,))
            else:
                cursor.execute("SELECT inst_id FROM instruments")
                
            result = cursor.fetchall()
            return [row[0] for row in result]
        except Exception as e:
            self.logger.error(f"获取交易对失败: {str(e)}")
            return []
        finally:
            cursor.close()
            conn.close()