# 训练模型
python src/main.py train transformer --experiment train_BTC-USDT_transformer_classfication --inst BTC-USDT --mode classification

# 测试模型
python src/main.py test --model_path D:\code\oktrade\src\logs\train_BTC-USDT_transformer_classfication\train_BTC-USDT_transformer_classification_best.pth --experiment train_BTC-USDT_transformer_classfication --mode classification

# 比较多个模型
python src/main.py compare --model_paths ./models/model1.pth ./models/model2.pth --experiment exp1 --inst BTC-USDT --mode regression