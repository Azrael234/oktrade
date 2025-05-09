import requests
import json
import time
from okx_utils import get_headers

def get_tickers(inst_type="SPOT", inst_id=None, use_proxy=False, proxy_host="127.0.0.1", proxy_port=10808, max_retries=3):
    """
    获取产品行情信息
    
    参数:
        inst_type (str): 产品类型
            SPOT：币币
            MARGIN：币币杠杆
            SWAP：永续合约
            FUTURES：交割合约
            OPTION：期权
            ANY：全部
        inst_id (str): 产品ID，如"BTC-USDT"，不填写则返回全部产品行情
        use_proxy (bool): 是否使用代理
        proxy_host (str): 代理服务器地址
        proxy_port (int): 代理服务器端口
        max_retries (int): 最大重试次数
    
    返回:
        list: 行情信息列表
    """
    base_url = "https://www.okx.com"
    request_path = "/api/v5/market/tickers"
    
    # 构建请求参数
    params = {
        "instType": inst_type
    }
    
    # 如果指定了产品ID，则添加到参数中
    if inst_id:
        params["instId"] = inst_id
    
    # 构建完整URL和请求路径
    query_string = f"?instType={inst_type}"
    if inst_id:
        query_string += f"&instId={inst_id}"
    
    url = base_url + request_path
    
    # 获取带签名的请求头
    headers = get_headers("GET", request_path + query_string)
    if not headers:
        print("获取请求头失败，请检查配置文件")
        return []
    
    # 配置代理
    proxies = None
    if use_proxy:
        proxies = {
            'http': f'socks5://{proxy_host}:{proxy_port}',
            'https': f'socks5://{proxy_host}:{proxy_port}'
        }
    
    # 尝试连接
    for attempt in range(max_retries):
        try:
            print(f"尝试获取行情数据 (尝试 {attempt+1}/{max_retries})...")
            if attempt > 0:
                print(f"等待 {attempt} 秒后重试...")
                time.sleep(attempt)  # 增加重试间隔
                
            response = requests.get(url, params=params, headers=headers, proxies=proxies, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                if result.get("code") == "0":
                    print("获取行情数据成功！")
                    return result.get("data", [])
                else:
                    print(f"获取行情数据失败: {result.get('msg')}")
            else:
                print(f"请求失败，状态码: {response.status_code}")
                
        except Exception as e:
            print(f"连接错误: {str(e)}")
    
    print("所有获取行情数据的尝试均失败")
    return []

if __name__ == "__main__":
    # 获取所有现货产品的行情
    tickers = get_tickers("SPOT", use_proxy=True)
    
    if tickers:
        # 打印前10个产品的行情
        for i, ticker in enumerate(tickers[:10]):
            print(f"{i+1}. {ticker['instId']}:")
            print(f"   最新成交价: {ticker.get('last', 'N/A')}")
            print(f"   24小时最高价: {ticker.get('high24h', 'N/A')}")
            print(f"   24小时最低价: {ticker.get('low24h', 'N/A')}")
            print(f"   24小时成交量: {ticker.get('vol24h', 'N/A')}")
            print(f"   买一价: {ticker.get('bidPx', 'N/A')}")
            print(f"   卖一价: {ticker.get('askPx', 'N/A')}")
        
        print(f"总共获取到 {len(tickers)} 个产品的行情数据")
        
        # 获取特定产品的行情（以BTC-USDT为例）
        btc_ticker = get_tickers("SPOT", "BTC-USDT", use_proxy=True)
        if btc_ticker:
            print("\nBTC-USDT 行情:")
            ticker = btc_ticker[0]
            print(f"最新成交价: {ticker.get('last', 'N/A')}")
            print(f"24小时最高价: {ticker.get('high24h', 'N/A')}")
            print(f"24小时最低价: {ticker.get('low24h', 'N/A')}")
            print(f"24小时成交量: {ticker.get('vol24h', 'N/A')}")
            print(f"买一价: {ticker.get('bidPx', 'N/A')}")
            print(f"卖一价: {ticker.get('askPx', 'N/A')}")
    else:
        print("未能获取到任何行情数据")