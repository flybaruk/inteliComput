# Delivery Bot - Versão 1: mudancaPrioridade.py

Este é um projeto de simulação em Pygame que apresenta um robô de entregas autônomo navegando em um ambiente com obstáculos. O objetivo do robô é coletar pacotes e entregá-los em metas específicas, otimizando sua rota para obter a maior pontuação possível.

Esta versão implementa um agente **inteligente e estratégico** que planeja suas ações com base em cálculos precisos de caminho.

## 🚀 Como Executar

1.  **Requisitos:** Certifique-se de ter a biblioteca Pygame instalada.
    ```bash
    pip install pygame
    ```
2.  **Imagens:** Crie uma pasta chamada `images` no mesmo diretório do script e coloque os arquivos `cargo.png` e `operator.png` dentro dela.
3.  **Execução:** Rode o script a partir do seu terminal.
    ```bash
    python3 mudancaPrioridade.py
    ```
4.  **Recriar um Cenário (Opcional):** Para testar o robô no mesmo mapa e com a mesma sequência de eventos, você pode usar a flag `--seed`.
    ```bash
    python3 mudancaPrioridade.py --seed 123
    ```

## 🤖 Estratégia do Robô

O "cérebro" deste robô é a classe `DefaultPlayer`, que emprega uma estratégia de planejamento sofisticada:

1.  **Cálculo de Caminho Real (A*):** Ao contrário de estimativas simples, este robô usa o **algoritmo A*** para calcular a distância exata do caminho entre dois pontos, levando em conta todos os obstáculos. Isso garante que suas decisões sejam baseadas em informações precisas.

2.  **Planejamento da Viagem Completa:** Quando está sem carga, o robô não escolhe apenas o pacote mais próximo. Ele avalia cada **viagem completa** possível (Posição Atual → Pacote → Meta). Para cada par, ele calcula o custo total em passos e o atraso potencial.

3.  **Sistema de Pontuação Ponderado:** A decisão é baseada em uma pontuação que busca o melhor equilíbrio entre eficiência e pontualidade. A fórmula efetiva é:
    `Pontuação = (-Atraso * Penalidade_de_Atraso) - Distância_Total_da_Viagem`

    Isso significa que ele prioriza fortemente o cumprimento dos prazos, mas, entre duas viagens igualmente urgentes, escolherá a mais curta.

## ⭐ Características Principais

-   **Navegação Inteligente:** Utiliza o algoritmo A* para desviar de obstáculos de forma eficiente.
-   **Tomada de Decisão Estratégica:** Avalia o custo-benefício de cada rota completa antes de se mover.
-   **Sensibilidade a Prazos:** Prioriza entregas urgentes para minimizar penalidades e maximizar a pontuação.
-   **Ambiente Dinâmico:** Novas metas de entrega surgem ao longo do tempo, forçando o robô a se adaptar.


# Delivery Bot - Versão 2: mudancaDistancia

Este é um projeto de simulação em Pygame que apresenta um robô de entregas autônomo. O objetivo do robô é coletar pacotes e entregá-los em metas específicas da forma mais eficiente possível.

Esta versão implementa um agente **simples e reativo**, que toma decisões rápidas com base em estimativas de distância.

## 🚀 Como Executar

1.  **Requisitos:** Certifique-se de ter a biblioteca Pygame instalada.
    ```bash
    pip install pygame
    ```
2.  **Imagens:** Crie uma pasta chamada `images` no mesmo diretório do script e coloque os arquivos `cargo.png` e `operator.png` dentro dela.
3.  **Execução:** Rode o script a partir do seu terminal.
    ```bash
    python3 mudancaDistancia.py
    ```
4.  **Recriar um Cenário (Opcional):** Para usar o mesmo mapa e sequência de eventos, utilize a flag `--seed`.
    ```bash
    python mudancaDistancia.py --seed 123
    ```

## 🤖 Estratégia do Robô

A lógica de decisão deste robô é projetada para ser rápida e simples, baseando-se em uma estratégia "gulosa" (greedy):

1.  **Cálculo de Distância Estimada (Manhattan):** Para avaliar a proximidade dos alvos, o robô usa a **Distância de Manhattan** (`abs(x1-x2) + abs(y1-y2)`). Esta é uma estimativa muito rápida, mas que ignora a existência de paredes ou obstáculos.

2.  **Decisão Imediata:**
    * **Se está sem carga:** O robô simplesmente escolhe o pacote que está mais próximo, de acordo com a Distância de Manhattan.
    * **Se está com carga:** Tenta encontrar a melhor meta combinando urgência (tempo restante) e distância (estimada).

3.  **Atraso Inicial:** O robô espera por 5 "passos" no início do jogo antes de começar a se mover, permitindo que o cenário inicial se desenvolva.

## ⚠️ Ponto de Atenção: Bug Conhecido

Esta versão do código contém um **bug crítico** na lógica de seleção de metas de entrega. O comando de retorno da função está indentado incorretamente, fazendo com que o robô **sempre escolha a primeira meta da lista**, sem avaliar as demais. Isso limita severamente a eficácia de sua estratégia de entrega.

## ⭐ Características Principais

-   **Decisões Rápidas:** Usa uma heurística simples para escolher alvos rapidamente.
-   **Estratégia Reativa:** Foca no objetivo imediato que parece ser o melhor, sem planejamento a longo prazo.
-   **Início Atrasado:** Pausa no começo do jogo para "observar" o ambiente.

# Delivery Bot - Versão 3: ultimaMudanca

Este é um projeto de simulação em Pygame que apresenta um robô de entregas autônomo. O objetivo do robô é coletar e entregar pacotes maximizando sua pontuação.

Esta é uma versão **avançada** do agente, que não só planeja sua viagem atual de forma ótima, mas também considera como suas ações impactarão suas **oportunidades futuras**.

## 🚀 Como Executar

1.  **Requisitos:** Certifique-se de ter a biblioteca Pygame instalada.
    ```bash
    pip install pygame
    ```
2.  **Imagens:** Crie uma pasta chamada `images` no mesmo diretório do script e coloque os arquivos `cargo.png` e `operator.png` dentro dela.
3.  **Execução:** Rode o script a partir do seu terminal.
    ```bash
    python ultimaMudanca.py
    ```
4.  **Recriar um Cenário (Opcional):** Para usar o mesmo mapa e sequência de eventos, utilize a flag `--seed`.
    ```bash
    python ultimaMudanca.py --seed 123
    ```

## 🤖 Estratégia do Robô

Este robô aprimora a estratégia da "Versão 1" adicionando uma camada extra de inteligência, simulando um planejamento de múltiplos passos.

1.  **Base Estratégica (A*):** Assim como na V1, o robô usa o algoritmo A* para calcular caminhos reais e avalia a viagem completa (coleta + entrega) com base no atraso e na distância.

2.  **NOVO - Bônus de Oportunidade Futura:** A grande inovação aqui é a avaliação do "estado do mundo" *após* a conclusão de uma viagem potencial. Ao avaliar uma rota (Pacote A → Meta B), o robô calcula um bônus adicional.

3.  **Sistema de Pontuação Aprimorado:** A fórmula de decisão se torna:
    `Pontuação Final = Pontuação da Viagem + Bônus de Oportunidade`

    * **Bônus de Oportunidade:** É calculado com base na distância entre o local de entrega (Meta B) e o pacote restante mais próximo. Se a entrega deixar o robô perto de sua próxima coleta potencial, a rota recebe um bônus significativo.

Isto faz com que o robô tome decisões mais holísticas, preferindo uma rota um pouco mais longa agora se ela o posicionar perfeitamente para a próxima tarefa, economizando tempo a longo prazo.

## 🔧 Parâmetros para Ajuste

Dentro da classe `DefaultPlayer`, você pode ajustar dois pesos para modificar o comportamento do robô:

-   `LATENESS_PENALTY_MULTIPLIER`: Aumente para tornar o robô mais obcecado em cumprir prazos.
-   `OPPORTUNITY_BONUS_WEIGHT`: Aumente para fazer o robô se importar mais com o planejamento futuro e o posicionamento estratégico.

## ⭐ Características Principais

-   **Planejamento de Múltiplos Passos (Heurístico):** Simula um planejamento a longo prazo ao valorizar boas posições futuras.
-   **Posicionamento Estratégico:** Entende que o local onde uma tarefa termina é tão importante quanto a tarefa em si.
-   **Comportamento Altamente Otimizado:** Busca a solução mais eficiente globalmente, não apenas para a tarefa imediata.
-   **Configurável:** Permite ajustar os "pilares" da sua personalidade (urgência vs. oportunidade).