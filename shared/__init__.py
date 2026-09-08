"""Modulos compartilhados entre api/, dashboard/, eval/, nl_to_sql/ e rag/.

Mesma motivacao que levou a extrair dashboard/roteador.py: qualquer coisa usada
por mais de um consumidor mora aqui, nunca duplicada manualmente em cada lugar
(ver o achado da regressao fantasma 67,9%->52% no historico de roteador.py).
"""
