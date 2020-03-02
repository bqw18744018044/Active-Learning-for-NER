import enum

# 主动学习样本选择策略
STRATEGY = enum.Enum('STRATEGY', ('RAND', 'LC', 'MNLP', 'TTE', 'TE'))