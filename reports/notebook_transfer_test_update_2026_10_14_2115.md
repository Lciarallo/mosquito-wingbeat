# Adendo de testes - 14/10/2026

A auditoria e o pacote da rodada registraram corretamente 232 testes aprovados e uma falha v5 no estado anterior do workspace. Depois, o arquivo `tests/test_nested_species_v5.py` foi alterado fora deste experimento, sem alterar pesos, código de features ou limiares: o caso de norma 2,5 passou a usar estrutura harmônica explícita, e foi adicionado um caso separado que verifica que o piso não amplifica janelas contendo só ruído.

A nova execução completa `PYTHONPATH=. .venv/bin/python -m pytest -q tests` resultou em **234 passed, 59.933 warnings in 39,74 s**, código zero. Os 12 novos casos da transferência do notebook passaram. A correção da fixture não foi atribuída à transferência LFCC e não melhora as métricas externas: RF de espécie continua 0/17, dois novos positivos continuam errados/rejeitados pelo candidato escolhido.

`verify_freeze` e `verify_history` passaram novamente, mantendo os 1.239 arquivos históricos. O relatório, pacote ZIP, pesos e resultados já registrados foram preservados; este adendo não reescreve o snapshot anterior. Os logs, patch observado e SHA-256 do teste atual estão em `results/notebook_transfer_verification_2026_10_14_2115/`.
