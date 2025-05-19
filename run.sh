# 训练模型
python main.py train transformer --experiment exp1 --inst BTC-USDT --mode regression

# 测试模型
python main.py test --model_path ./models/model1.pth --experiment exp1 --mode regression

# 比较多个模型
python main.py compare --model_paths ./models/model1.pth ./models/model2.pth --experiment exp1 --inst BTC-USDT --mode regression