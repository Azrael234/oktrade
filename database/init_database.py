import os
import sys
import json
import logging
import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config():
    """
    加载配置文件
    """
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.json')
    
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"加载配置文件失败: {str(e)}")
        return {}

def create_database(host, user, password, port, db_name):
    """
    创建数据库
    """
    try:
        # 连接到PostgreSQL服务器的默认数据库postgres
        conn = psycopg2.connect(
            host=host,
            user=user,
            password=password,
            port=port,
            database="postgres"  # 连接到默认的postgres数据库
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # 检查数据库是否已存在
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
        exists = cursor.fetchone()
        
        if not exists:
            # 创建数据库
            cursor.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_name)))
            logger.info(f"数据库 {db_name} 创建成功")
        else:
            logger.info(f"数据库 {db_name} 已存在")
        
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"创建数据库失败: {str(e)}")
        return False

def create_tables(host, user, password, port, db_name):
    """
    创建表结构
    """
    try:
        # 连接到新创建的数据库
        conn = psycopg2.connect(
            host=host,
            user=user,
            password=password,
            port=port,
            database=db_name
        )
        cursor = conn.cursor()
        
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
        logger.info("数据库表结构创建成功")
        
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"创建表结构失败: {str(e)}")
        return False

def main():
    """
    主函数
    """
    print("=" * 50)
    print("OKTrade数据库初始化")
    print("=" * 50)
    
    # 加载配置
    config = load_config()
    db_config = config.get('database', {})
    
    host = db_config.get('host', 'localhost')
    user = db_config.get('user', 'postgres')
    password = db_config.get('password', 'postgres')
    port = db_config.get('port', 5432)
    db_name = db_config.get('database', 'oktrade')
    
    print(f"\n数据库配置信息:")
    print(f"主机: {host}")
    print(f"端口: {port}")
    print(f"用户名: {user}")
    print(f"数据库名: {db_name}")
    
    # 创建数据库
    print("\n正在创建数据库...")
    if create_database(host, user, password, port, db_name):
        # 创建表结构
        print("\n正在创建表结构...")
        if create_tables(host, user, password, port, db_name):
            print("\n数据库初始化成功!")
        else:
            print("\n创建表结构失败，请检查错误日志")
    else:
        print("\n创建数据库失败，请检查错误日志")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    main()