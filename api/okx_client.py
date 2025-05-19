import requests
import json
import time
import base64
import hmac
import datetime
import logging
from urllib.parse import urlencode

class OKXClient:
    def __init__(self, api_key=None, secret_key=None, passphrase=None, use_proxy=False, proxy_host="127.0.0.1", proxy_port=10808):
        """
        OKX API客户端
        
        参数:
            api_key (str): API Key
            secret_key (str): Secret Key
            passphrase (str): Passphrase
            use_proxy (bool): 是否使用代理
            proxy_host (str): 代理主机
            proxy_port (int): 代理端口
        """
        self.api_key = api_key
        self.secret_key = secret_key
        self.passphrase = passphrase
        self.base_url = "https://www.okx.com"
        self.use_proxy = use_proxy
        self.proxy_host = proxy_host
        self.proxy_port = proxy_port
        self.logger = logging.getLogger(__name__)
        self._request_timestamps = []
    
    def _get_timestamp(self):
        """
        获取ISO格式的时间戳
        """
        return datetime.datetime.utcnow().isoformat(timespec='milliseconds') + 'Z'
    
    def _sign(self, timestamp, method, request_path, body=''):
        """
        生成签名
        """
        if not self.secret_key:
            return None
            
        if body and isinstance(body, dict):
            body = json.dumps(body)
            
        message = timestamp + method + request_path + (body or '')
        
        mac = hmac.new(
            bytes(self.secret_key, encoding='utf8'),
            bytes(message, encoding='utf-8'),
            digestmod='sha256'
        )
        d = mac.digest()
        return base64.b64encode(d).decode('utf-8')
    
    def _get_headers(self, method, request_path, body=''):
        """
        获取请求头
        """
        timestamp = self._get_timestamp()
        
        headers = {
            'Content-Type': 'application/json'
        }
        
        if self.api_key and self.secret_key and self.passphrase:
            sign = self._sign(timestamp, method, request_path, body)
            headers.update({
                'OK-ACCESS-KEY': self.api_key,
                'OK-ACCESS-SIGN': sign,
                'OK-ACCESS-TIMESTAMP': timestamp,
                'OK-ACCESS-PASSPHRASE': self.passphrase
            })
            
        return headers
    
    def _rate_limit(self):
        now = time.time()
        # 只保留2秒内的请求
        self._request_timestamps = [t for t in self._request_timestamps if now - t < 2]
        if len(self._request_timestamps) >= 20:
            sleep_time = 2 - (now - self._request_timestamps[0])
            if sleep_time > 0:
                time.sleep(sleep_time)
        self._request_timestamps.append(time.time())

    def _request(self, method, request_path, params=None, body=None, max_retries=3):
        """
        发送请求
        """
        url = self.base_url + request_path
        
        # 处理GET请求的查询参数
        if method == 'GET' and params:
            query_string = urlencode(params)
            request_path = request_path + '?' + query_string
            url = self.base_url + request_path
        
        headers = self._get_headers(method, request_path, body)
        
        # 配置代理
        proxies = None
        if self.use_proxy:
            proxies = {
                'http': f'socks5h://{self.proxy_host}:{self.proxy_port}',
                'https': f'socks5h://{self.proxy_host}:{self.proxy_port}'
            }
        
        for attempt in range(max_retries):
            self._rate_limit()  # 新增：限速
            try:
                self.logger.debug(f"尝试请求 {url} (尝试 {attempt+1}/{max_retries})")
                
                if attempt > 0:
                    time.sleep(attempt)  # 增加重试间隔
                
                if method == 'GET':
                    response = requests.get(url, headers=headers, proxies=proxies, timeout=10)
                elif method == 'POST':
                    response = requests.post(url, headers=headers, json=body, proxies=proxies, timeout=10)
                else:
                    raise ValueError(f"不支持的请求方法: {method}")
                
                # 打印响应内容，帮助调试
                self.logger.debug(f"响应状态码: {response.status_code}")
                self.logger.debug(f"响应内容: {response.text[:200]}...")  # 只打印前200个字符
                
                if response.status_code == 200:
                    return response.json()  # 直接返回完整的JSON响应
                else:
                    self.logger.error(f"请求失败，状态码: {response.status_code}")
                    self.logger.error(f"响应内容: {response.text}")
                    
            except Exception as e:
                self.logger.error(f"请求异常: {str(e)}")
                
                if attempt == max_retries - 1:
                    raise
        
        return None  # 如果所有尝试都失败，返回None而不是空列表
    
    def get_kline(self, inst_id, start_ts=None, end_ts=None, bar='1m', limit=100):
        """
        获取K线数据
        
        参数:
            inst_id (str): 产品ID
            start_ts (int): 开始时间戳（毫秒）
            end_ts (int): 结束时间戳（毫秒）
            bar (str): K线周期，如1m、5m、15m、1H、4H、1D等
            limit (int): 返回的结果集数量，最大值为100
            
        返回:
            list: K线数据列表
        """
        path = '/api/v5/market/history-candles'  # 使用历史K线接口
        
        params = {
            'instId': inst_id,
            'bar': bar,
            'limit': limit
        }
        
        # OKX API使用after参数表示结束时间，before参数表示开始时间
        # 注意：OKX API的时间戳参数需要字符串格式
        if start_ts:
            params['before'] = str(start_ts)  # 注意这里是before
        if end_ts:
            params['after'] = str(end_ts)    # 注意这里是after
        
        self.logger.info(f"请求K线数据参数: {params}")
        
        try:
            response = self._request('GET', path, params)
            
            if response and response.get("code") == "0":
                data = response.get("data", [])
                self.logger.info(f"成功获取 {len(data)} 条K线数据")
                return data
            else:
                error_msg = response.get('msg', '未知错误') if response else '请求失败'
                self.logger.error(f"获取K线数据失败: {error_msg}")
                if response:
                    self.logger.error(f"响应内容: {response}")
                return []
        except Exception as e:
            self.logger.error(f"获取K线数据异常: {str(e)}")
            return []
    
    def get_tickers(self, inst_type="SPOT", inst_id=None):
        """
        获取产品行情信息
        
        参数:
            inst_type (str): 产品类型
            inst_id (str): 产品ID，如"BTC-USDT"，不填写则返回全部产品行情
            
        返回:
            list: 行情信息列表
        """
        params = {"instType": inst_type}
        
        if inst_id:
            params["instId"] = inst_id
            
        return self._request('GET', '/api/v5/market/tickers', params=params)
    
    def get_instruments(self, inst_type="SPOT"):
        """
        获取交易产品基础信息
        
        参数:
            inst_type (str): 产品类型
                SPOT：币币
                MARGIN：币币杠杆
                SWAP：永续合约
                FUTURES：交割合约
                OPTION：期权
                ANY：全部
        
        返回:
            list: 交易产品列表
        """
        path = '/api/v5/public/instruments'
        
        params = {"instType": inst_type}
        
        try:
            response = self._request('GET', path, params)
            
            if response and isinstance(response, dict) and response.get("code") == "0":
                return response.get("data", [])
            else:
                error_msg = response.get('msg', '未知错误') if response else '请求失败'
                self.logger.error(f"获取交易产品信息失败: {error_msg}")
                if response:
                    self.logger.error(f"响应内容: {response}")
                return []
        except Exception as e:
            self.logger.error(f"获取交易产品信息异常: {str(e)}")
            return []