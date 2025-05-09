# OKTrade

这是一个基于OKX交易所API的交易工具，用于获取行情数据、执行交易策略等功能。

## 功能特点

- 获取K线数据
- 获取产品行情信息
- 获取交易产品基础信息
- 支持代理设置

## 使用方法

1. 复制`config.json.example`为`config.json`并填入您的API密钥信息
2. 运行主程序：`python main.py`

## 项目结构

- `api/`: API客户端
- `data/`: 数据收集和聚合
- `database/`: 数据库操作
- `models/`: 交易模型
- `visualization/`: 数据可视化

## 配置说明

请在`config.json`中配置以下信息：

```json
{
    "api_key": "您的API Key",
    "secret_key": "您的Secret Key",
    "passphrase": "您的API密码",
    "proxy_host": "代理主机地址",
    "proxy_port": 代理端口,
    "database": {
        "host": "数据库主机",
        "database": "数据库名",
        "user": "用户名",
        "password": "密码",
        "port": 数据库端口
    }
}
```

## 注意事项

- 请勿将您的API密钥和数据库凭据提交到代码仓库
- 使用前请确保已安装所需的依赖包