import requests
import json
import time
from okx_utils import get_headers

def get_instruments(inst_type="SPOT", use_proxy=False, proxy_host="127.0.0.1", proxy_port=10808, max_retries=3):
    """
    获取数字货币对列表
    
    参数:
        inst_type (str): 产品类型
            SPOT：币币
            MARGIN：币币杠杆
            SWAP：永续合约
            FUTURES：交割合约
            OPTION：期权
            ANY：全部
        use_proxy (bool): 是否使用代理
        proxy_host (str): 代理服务器地址
        proxy_port (int): 代理服务器端口
        max_retries (int): 最大重试次数
    
    返回:
        list: 货币对列表
    """
    base_url = "https://www.okx.com"
    request_path = "/api/v5/public/instruments"
    url = base_url + request_path
    
    params = {
        "instType": inst_type
    }
    
    # 获取带签名的请求头
    headers = get_headers("GET", request_path + "?instType=" + inst_type)
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
            print(f"尝试连接 (尝试 {attempt+1}/{max_retries})...")
            if attempt > 0:
                print(f"等待 {attempt} 秒后重试...")
                time.sleep(attempt)  # 增加重试间隔
                
            response = requests.get(url, params=params, headers=headers, proxies=proxies, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                if result.get("code") == "0":
                    print("连接成功！")
                    return result.get("data", [])
                else:
                    print(f"获取货币对失败: {result.get('msg')}")
            else:
                print(f"请求失败，状态码: {response.status_code}")
                
        except Exception as e:
            print(f"连接错误: {str(e)}")
    
    print("所有连接尝试均失败")
    return []

if __name__ == "__main__":
    # 获取所有现货交易对
    spot_instruments = get_instruments("SPOT", use_proxy=True)
    
    if spot_instruments:
        # 打印前10个交易对
        for i, inst in enumerate(spot_instruments[:10]):
            print(f"{i+1}. {inst['instId']} - 基础货币: {inst.get('baseCcy', 'N/A')}, 计价货币: {inst.get('quoteCcy', 'N/A')}")
        
        print(f"总共获取到 {len(spot_instruments)} 个现货交易对")
    else:
        print("未能获取到任何交易对数据")