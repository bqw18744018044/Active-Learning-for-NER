import enum

# query strategy in active learning
STRATEGY = enum.Enum('STRATEGY', ('RAND', 'LC', 'MNLP', 'TTE', 'TE'))