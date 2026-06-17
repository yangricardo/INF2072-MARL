# Revisão da Apresentação — INF2072 (slide a slide)

> Cruzamento de cada slide com os **dados reais** em
> `comparativos/comparativo_metricas.csv` e os relatórios por modelo.
> Gerado em 2026-06-16.

## ⚠️ Resumo executivo

Os slides de conteúdo (1–8) estão **bons**, com pequenos ajustes.
Os slides de resultados (**10, 11, 12 e 14**) usam **dados inventados** que
contradizem o que os modelos realmente produziram — em um caso, de forma
**fisicamente impossível** (VDN "entregando" 11 caixas num ambiente de 8).
Esses gráficos precisam ser **trocados pelos reais** antes de apresentar.

### Tabela-verdade (dados reais, 3000 episódios)

| Modelo          | Recompensa | Entregas/8 | Taxa sucesso | Colisões |
|-----------------|-----------:|-----------:|-------------:|---------:|
| Random          |      85.03 |       3.19 |        0.399 |    138.1 |
| **IDQN**        | **316.05** |    **8.0** |     **1.00** |  **9.2** |
| VDN             |       7.03 |       1.04 |        0.13  |    807.3 |
| VDN (ajustado)  |      44.92 |       1.77 |        0.221 |    261.4 |
| MAPPO (orig.)   |      -9.52 |       0.07 |        0.009 |    263.7 |
| MAPPO (corrig.) |     313.70 |       8.0  |        1.00  |     24.8 |

**A história real:** IDQN venceu (100%). MAPPO e VDN *falharam* no formato
original. Investigando, achamos um **bug no MAPPO** (amostragem/ log-probs) —
corrigido, ele empata com o IDQN. O VDN melhorou com ajustes mas a decomposição
aditiva continua insuficiente. **Essa narrativa é mais forte do que os
gráficos falsos** — e é verdadeira.

---

## Slide a slide

### Pág. 1 — Capa
✅ OK. Sem alterações.

### Pág. 3 — Ambiente
- 🔧 "…usando o algoritmo **IDQN e MAPPO**." → foram testados **5** abordagens:
  Random, IDQN, VDN, MAPPO e HATRPO (implementado). Trocar por
  "…comparando Random, IDQN, VDN e MAPPO."
- 🔧 Tabela inline mostra recompensa de movimento `0.005` (positivo). O valor
  real é **−0.005** (penalidade por passo). A pág. 4 já mostra `-0.005`
  correto — alinhar as duas.

### Pág. 4 — Recompensas / ambiente
✅ Valores corretos: movimento −0.005, pegar +2.0, soltar +25.0, falha −0.15,
  conclusão +50 (não citada — pode adicionar). Sem alterações obrigatórias.

### Págs. 5–6 — Modelos / taxonomia
✅ OK.

### Pág. 7 — Tabela de modelos (vantagens/limitações)
- ⚠️ VDN aparece com vantagem **"Melhor coordenação"**. Na prática o VDN foi o
  **pior** (7.03 de recompensa, 807 colisões). Essa é uma vantagem **teórica**
  que **não se confirmou**. Sugestão: marcar como "vantagem teórica" e, na fala,
  contrastar com o resultado real — vira um ponto de discussão interessante.

### Pág. 8 — Tabela comparativa (Aprende / Cooperação / …)
- ⚠️ Linha **"Cooperação": IDQN=Baixa, VDN=Média, MPPO=Muito Alta**. Isso é
  ranking **teórico** e está **invertido** em relação aos resultados: quem mais
  cooperou (100% das entregas, 9 colisões) foi o **IDQN**; o MAPPO *original*
  falhou. Duas saídas:
  1. Renomear a linha para "Cooperação **(esperada/teórica)**"; ou
  2. Trocar pelos valores reais observados.

> 💡 **Falta uma coluna/linha:** os slides só citam 4 modelos. Não aparecem
> **MAPPO corrigido** nem **VDN ajustado** — justamente os experimentos mais
> ricos. Considere adicioná-los aqui ou num slide próprio (ver "Slides a
> adicionar").

### Pág. 9 — Divisória "Resultados"
✅ OK.

### Pág. 10 — Gráfico "Taxa de Sucesso" ❌ FABRICADO
- Slide mostra **MAPPO terminando em ~0.94 (o melhor)** e IDQN em ~0.75.
- Real: **IDQN = 1.00 (vencedor)**; **MAPPO original = 0.009 (falhou)**;
  VDN = 0.13; Random = 0.399.
- As curvas são retas suaves/monótonas — assinatura de dados sintéticos
  (`np.random`/linear), não de treino real.
- ✅ **Trocar por:** `comparativos/comparativo_taxa_sucesso.png`

### Pág. 11 — Gráfico "Colisões" ❌ FABRICADO
- Slide mostra VDN bem-comportado (~92 colisões) e IDQN ~24.
- Real: **VDN = 807 colisões** (≈9× o mostrado!); IDQN = 9.2.
- ✅ **Trocar por:** `comparativos/comparativo_colisoes.png`

### Pág. 12 — Gráfico "Entregas" ❌ FABRICADO (impossível)
- Slide mostra a linha do **VDN subindo até 11 entregas** — acima da meta de 8.
  **O ambiente tem só 8 caixas: 11 é impossível.** Real: VDN = 1.04.
- MAPPO aparece achatado em 0 (≈ coerente com o original, que falhou).
  IDQN em 8 (correto).
- 🚩 Este é o slide mais comprometedor — qualquer pessoa da banca que olhar o
  eixo percebe o erro.
- ✅ **Trocar por:** `comparativos/comparativo_entregas.png`

### Pág. 13 — "Entregas por Episódio + Média Móvel (100)" ✅ REAL
- Gráfico com ruído realista (provavelmente MAPPO original ou VDN, que decaem
  para ~0). Este parece **autêntico**. Manter — mas **rotular qual modelo é**
  (hoje não diz). Serve bem para ilustrar um treino que *não* converge.

### Pág. 14 — Gráfico "Falhas R1/R2" ⚠️ PROVÁVEL FABRICADO
- Curvas triangulares perfeitamente suaves (MAPPO com pico ~1050) — padrão
  sintético. Sem CSV de "falhas" agregadas confiável para todos os modelos.
- ✅ Sugestão: substituir por gráfico de **recompensa** real
  (`comparativos/comparativo_recompensa.png`), que conta a mesma história de
  forma honesta, ou pelo painel 2×2.

---

## Slides a ADICIONAR (a parte forte e verdadeira)

1. **Resultado final em barras** → `comparativos/comparativo_barras_final.png`
   (placar limpo dos 6 modelos; ótimo slide de "veredito").
2. **Painel 2×2** → `comparativos/comparativo_painel.png`
   (recompensa, entregas, sucesso e colisões juntos).
3. **Velocidade de convergência** → `comparativos/comparativo_velocidade.png`
   — insight novo: MAPPO corrigido chega a 7/8 ~2× mais rápido que o IDQN.
4. **"Diagnóstico: o bug do MAPPO"** — antes (−9.52, 0.07/8) → depois da
   correção da amostragem (313.7, 8/8). Mostra engenharia reversa de verdade.
5. **"VDN: testamos 2 hipóteses"** — MAX_STEPS=500 + injeção de ID de agente:
   colisões 807 → 261, mas sucesso continua baixo → evidência de que a
   decomposição aditiva é o gargalo.

## Vídeos (prova visual ao vivo)
`Output/videos/`: `video_random.gif` (caótico) → `video_idqn.gif` (8/8 limpo)
→ `video_mappo_corrigido.gif` (8/8). Excelentes para abrir/encerrar resultados.
