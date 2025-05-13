import base64
import datetime
import hmac
import json
import os

def load_config():
    """
    加载配置文件
    """
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config_moni.json")
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"加载配置文件失败: {str(e)}")
        return None

def get_timestamp():
    """
    获取ISO格式的时间戳
    """
    return datetime.datetime.utcnow().isoformat(timespec='milliseconds') + 'Z'

def sign(timestamp, method, request_path, body, secret_key):
    """
    生成签名
    
    参数:
        timestamp (str): ISO格式的时间戳
        method (str): 请求方法，如GET、POST
        request_path (str): 请求路径
        body (str): 请求体，GET请求为空字符串
        secret_key (str): API密钥
    
    返回:
        str: 签名
    """
    if body is None:
        body = ""
    
    message = timestamp + method + request_path + body
    mac = hmac.new(
        bytes(secret_key, encoding='utf8'),
        bytes(message, encoding='utf-8'),
        digestmod='sha256'
    )
    d = mac.digest()
    return base64.b64encode(d).decode('utf-8')

def get_headers(method, request_path, body=None):
    """
    获取带签名的请求头
    
    参数:
        method (str): 请求方法，如GET、POST
        request_path (str): 请求路径
        body (str): 请求体，GET请求为空字符串
    
    返回:
        dict: 请求头
    """
    config = load_config()
    if not config:
        return None
    
    timestamp = get_timestamp()
    sign_str = sign(timestamp, method, request_path, body, config['secret_key'])
    
    headers = {
        'Content-Type': 'application/json',
        'OK-ACCESS-KEY': config['api_key'],
        'OK-ACCESS-SIGN': sign_str,
        'OK-ACCESS-TIMESTAMP': timestamp,
        'OK-ACCESS-PASSPHRASE': config['passphrase'],
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    return headers