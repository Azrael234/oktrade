import requests
import time
import sys

def test_proxy(proxy_host="127.0.0.1", proxy_port=10808, test_url="https://www.google.com", timeout=10):
    """
    测试SOCKS5代理是否可用
    
    参数:
        proxy_host (str): 代理服务器地址
        proxy_port (int): 代理服务器端口
        test_url (str): 用于测试的URL
        timeout (int): 超时时间(秒)
    
    返回:
        bool: 代理是否可用
    """
    proxies = {
        'http': f'socks5h://{proxy_host}:{proxy_port}',
        'https': f'socks5h://{proxy_host}:{proxy_port}'
    }
    
    print(f"正在测试代理: {proxy_host}:{proxy_port}")
    print(f"测试URL: {test_url}")
    print(f"代理设置: {proxies}")
    
    try:
        print("开始连接...")
        start_time = time.time()
        response = requests.get(test_url, proxies=proxies, timeout=timeout)
        elapsed_time = time.time() - start_time
        
        if response.status_code == 200:
            print(f"代理测试成功! 响应时间: {elapsed_time:.2f}秒")
            print(f"响应状态码: {response.status_code}")
            return True
        else:
            print(f"代理连接成功但返回了非200状态码: {response.status_code}")
            return False
    except requests.exceptions.ProxyError as e:
        print(f"代理错误: {str(e)}")
        return False
    except requests.exceptions.ConnectTimeout:
        print(f"连接代理超时 (>{timeout}秒)")
        return False
    except requests.exceptions.ReadTimeout:
        print(f"从代理读取数据超时 (>{timeout}秒)")
        return False
    except requests.exceptions.ConnectionError as e:
        print(f"连接错误: {str(e)}")
        print("这可能是因为:")
        print("1. 代理服务器拒绝了连接")
        print("2. 代理服务器不支持访问该网站")
        print("3. 代理服务器配置问题")
        return False
    except Exception as e:
        print(f"未知错误: {str(e)}")
        return False

def test_direct_connection(test_url="https://www.okx.com", timeout=10):
    """
    测试直接连接是否可用
    
    参数:
        test_url (str): 用于测试的URL
        timeout (int): 超时时间(秒)
    
    返回:
        bool: 直连是否可用
    """
    print(f"正在测试直接连接: {test_url}")
    
    try:
        start_time = time.time()
        response = requests.get(test_url, timeout=timeout)
        elapsed_time = time.time() - start_time
        
        if response.status_code == 200:
            print(f"直连测试成功! 响应时间: {elapsed_time:.2f}秒")
            print(f"响应状态码: {response.status_code}")
            return True
        else:
            print(f"直连成功但返回了非200状态码: {response.status_code}")
            return False
    except Exception as e:
        print(f"直连错误: {str(e)}")
        return False

if __name__ == "__main__":
    # 测试参数
    proxy_host = "127.0.0.1"
    proxy_port = 10808
    
    # 从命令行参数获取代理设置（如果有）
    if len(sys.argv) >= 3:
        proxy_host = sys.argv[1]
        proxy_port = int(sys.argv[2])
    
    # 测试网站列表
    test_sites = [
        "https://www.okx.com",
        "https://www.baidu.com",
        "https://www.google.com",
        "https://api.github.com"
    ]
    
    # 测试代理连接
    print("=" * 50)
    print("测试SOCKS5代理连接")
    print("=" * 50)
    
    proxy_results = {}
    for site in test_sites:
        print(f"\n测试网站: {site}")
        proxy_results[site] = test_proxy(proxy_host, proxy_port, test_url=site)
    
    # 测试直接连接
    print("\n" + "=" * 50)
    print("测试直接连接")
    print("=" * 50)
    
    direct_results = {}
    for site in test_sites:
        print(f"\n测试网站: {site}")
        direct_results[site] = test_direct_connection(test_url=site)
    
    # 总结
    print("\n" + "=" * 50)
    print("连接测试结果")
    print("=" * 50)
    
    print("\nSOCKS5代理测试结果:")
    for site, result in proxy_results.items():
        print(f"{site}: {'可用' if result else '不可用'}")
    
    print("\n直接连接测试结果:")
    for site, result in direct_results.items():
        print(f"{site}: {'可用' if result else '不可用'}")
    
    # 建议
    print("\n建议:")
    if any(proxy_results.values()):
        print("- 代理连接对某些网站可用，可以在API调用中使用代理")
        print("  可访问的网站: " + ", ".join([site for site, result in proxy_results.items() if result]))
    elif any(direct_results.values()):
        print("- 代理连接不可用，但直连可以访问某些网站，建议在API调用中不使用代理")
        print("  可直连的网站: " + ", ".join([site for site, result in direct_results.items() if result]))
    else:
        print("- 代理和直连均不可用，请检查网络连接或防火墙设置")