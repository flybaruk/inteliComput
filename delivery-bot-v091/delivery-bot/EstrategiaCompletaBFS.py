import pygame
import random
import heapq
import sys
import argparse
from abc import ABC, abstractmethod
import math # Importado para usar 'inf'

# ==========================
# CLASSES DE PLAYER
# ==========================
class BasePlayer(ABC):
    """
    Classe base para o jogador (robô).
    Para criar uma nova estratégia de jogador, basta herdar dessa classe e implementar o método escolher_alvo.
    """
    def __init__(self, position):
        self.position = position  # Posição no grid [x, y]
        self.cargo = 0            # Número de pacotes atualmente carregados

    @abstractmethod
    def escolher_alvo(self, world, current_steps, maze_solver):
        """
        Retorna o alvo (posição) que o jogador deseja ir.
        Recebe o objeto world, os passos atuais e o objeto maze para cálculos.
        """
        pass

class DefaultPlayer(BasePlayer):
    """
    Implementação padrão do jogador (lógica antiga).
    Se não estiver carregando pacotes, escolhe o pacote mais próximo.
    Caso contrário, escolhe a meta (entrega) mais próxima.
    """
    def get_remaining_steps(self, goal, current_steps):
        prioridade = goal["priority"]
        idade = current_steps - goal["created_at"]
        return prioridade - idade

    def escolher_alvo(self, world, current_steps, maze_solver=None):
        sx, sy = self.position
        if self.cargo == 0 and world.packages:
            best = None
            best_dist = float('inf')
            for pkg in world.packages:
                d = abs(pkg[0] - sx) + abs(pkg[1] - sy)
                if d < best_dist:
                    best_dist = d
                    best = pkg
            return best
        elif world.goals:
            best = None
            best_dist = float('inf')
            for goal in world.goals:
                gx, gy = goal["pos"]
                d = abs(gx - sx) + abs(gy - sy)
                if d < best_dist:
                    best_dist = d
                    best = goal["pos"]
            return best
        return None

# ######################################################################################
# ### NOVO: Classe SmartPlayer com lógica de decisão para múltiplas cargas ###
# ######################################################################################
class SmartPlayer(BasePlayer):
    """
    Implementação de um jogador inteligente com capacidade para múltiplas cargas.
    A cada decisão, ele avalia se é mais vantajoso coletar um novo pacote
    ou entregar um dos que já possui, sempre respeitando as restrições do ambiente.
    """
    def __init__(self, position):
        super().__init__(position)
        # NOVA CARACTERÍSTICA: Capacidade máxima de carga do robô.
        self.max_cargo = 2

    def _evaluate_best_goal(self, world, current_steps, maze_solver):
        """
        Função auxiliar para calcular qual é o melhor OBJETIVO para ir no momento.
        Retorna o melhor objetivo e sua pontuação.
        """
        best_goal = None
        best_score = -math.inf
        
        for goal in world.goals:
            # Usa A* para a distância real
            dist_to_goal = len(maze_solver.greedy_bfs(self.position, goal["pos"]))
            if dist_to_goal == 0 and self.position != goal["pos"]: continue # Ignora se inalcançável

            # Calcula a penalidade por atraso
            arrival_time_at_goal = current_steps + dist_to_goal
            deadline = goal["created_at"] + goal["priority"]
            lateness = max(0, arrival_time_at_goal - deadline)

            # A pontuação é baseada na recompensa, menos o custo (distância) e a penalidade
            reward = 50 
            score = reward - dist_to_goal - (lateness * 1.5)
            
            if score > best_score:
                best_score = score
                best_goal = goal
        
        return best_goal, best_score

    def _evaluate_best_package(self, world, current_steps, maze_solver):
        """
        Função auxiliar para calcular qual é o melhor PACOTE para coletar no momento.
        Retorna o melhor pacote e sua pontuação.
        """
        best_pkg = None
        best_score = -math.inf

        for package in world.packages:
            dist_to_pkg = len(maze_solver.greedy_bfs(self.position, package))
            if dist_to_pkg == 0 and self.position != package: continue # Ignora se inalcançável
            
            # A pontuação para pegar um pacote é inversamente proporcional à distância.
            # Damos uma pequena recompensa base para incentivá-lo a não ficar parado.
            # O objetivo é simplesmente encontrar o mais próximo e mais "barato" de pegar.
            score = 10 - dist_to_pkg 
            
            if score > best_score:
                best_score = score
                best_pkg = package
                
        return best_pkg, best_score

    def escolher_alvo(self, world, current_steps, maze_solver):
        """
        Lógica principal de decisão, agora com suporte a múltiplas cargas.
        O robô decide seu PRÓXIMO passo (coletar ou entregar) a cada chamada.
        """
        # --- LÓGICA DE DECISÃO ---

        # RESTRIÇÃO: O robô só pode pensar em pegar mais pacotes se:
        # 1. Sua carga atual for menor que a capacidade máxima.
        # 2. O número de pacotes que ele carrega for estritamente menor que o número de objetivos.
        # 3. Existirem pacotes disponíveis no mapa.
        can_pickup_more = (self.cargo < self.max_cargo) and \
                          (self.cargo < len(world.goals)) and \
                          world.packages

        # --- CASO 1: DEVE ENTREGAR ---
        # Motivos: A carga está cheia, ou ele tem pacotes mas não pode/deve pegar mais.
        if self.cargo > 0 and not can_pickup_more:
            if not world.goals: return None # Não há onde entregar, então espera.
            print(f"DECISÃO: Carga ({self.cargo}/{self.max_cargo}) impede nova coleta. Procurando melhor entrega.")
            best_goal, _ = self._evaluate_best_goal(world, current_steps, maze_solver)
            return best_goal["pos"] if best_goal else None

        # --- CASO 2: DEVE COLETAR ---
        # Motivo: A carga está vazia e existem pacotes disponíveis.
        if self.cargo == 0 and world.packages:
            print("DECISÃO: Carga vazia. Procurando pacote mais vantajoso.")
            best_pkg, _ = self._evaluate_best_package(world, current_steps, maze_solver)
            return best_pkg

        # --- CASO 3: A ESCOLHA ESTRATÉGICA ---
        # Motivo: Tem carga, mas ainda tem espaço e PODE coletar mais.
        # Aqui, ele precisa decidir se vale a pena pegar mais um ou já ir entregar.
        if self.cargo > 0 and can_pickup_more:
            # Opção A: Ir para um objetivo
            best_goal, goal_score = self._evaluate_best_goal(world, current_steps, maze_solver)
            
            # Opção B: Pegar outro pacote
            best_pkg, pkg_score = self._evaluate_best_package(world, current_steps, maze_solver)

            print(f"DECISÃO: Escolhendo... Pontuação Entrega: {goal_score:.2f} vs Pontuação Coleta: {pkg_score:.2f}")

            # Compara as duas opções e escolhe a de maior pontuação
            if best_goal and best_pkg:
                if goal_score >= pkg_score:
                    print("--> ESCOLHA: Ir para a entrega.")
                    return best_goal["pos"]
                else:
                    print("--> ESCOLHA: Pegar mais um pacote.")
                    return best_pkg
        
        # --- Caso padrão: Nenhuma ação válida encontrada ---
        print("DECISÃO: Nenhuma ação válida no momento. Aguardando.")
        return None

# ==========================
# CLASSE WORLD (MUNDO)
# ==========================
class World:
    def __init__(self, seed=None):
        if seed is not None:
            random.seed(seed)
        # Parâmetros do grid e janela
        self.maze_size = 30
        self.width = 600
        self.height = 600
        self.block_size = self.width // self.maze_size

        # Cria uma matriz 2D para planejamento de caminhos:
        # 0 = livre, 1 = obstáculo
        self.map = [[0 for _ in range(self.maze_size)] for _ in range(self.maze_size)]
        # Geração de obstáculos com padrão de linha (assembly line)
        self.generate_obstacles()
        # Gera a lista de paredes a partir da matriz
        self.walls = []
        for row in range(self.maze_size):
            for col in range(self.maze_size):
                if self.map[row][col] == 1:
                    self.walls.append((col, row))

        # Número total de entregas (metas) planejadas ao longo do jogo
        self.total_items = 6

        # Geração dos locais de coleta (pacotes)
        self.packages = []
        while len(self.packages) < self.total_items + 1:
            x = random.randint(0, self.maze_size - 1)
            y = random.randint(0, self.maze_size - 1)
            if self.map[y][x] == 0 and [x, y] not in self.packages:
                self.packages.append([x, y])

        # Metas (goals) com surgimento ao longo do tempo
        self.goals = []

        # ### ALTERAÇÃO: Usa o SmartPlayer em vez do DefaultPlayer ###
        self.player = self.generate_player()

        # Inicializa a janela do Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Delivery Bot")

        # Carrega imagens
        self.package_image = pygame.image.load("images/cargo.png")
        self.package_image = pygame.transform.scale(self.package_image, (self.block_size, self.block_size))
        self.goal_image = pygame.image.load("images/operator.png")
        self.goal_image = pygame.transform.scale(self.goal_image, (self.block_size, self.block_size))

        # Cores
        self.wall_color = (100, 100, 100)
        self.ground_color = (255, 255, 255)
        self.player_color = (0, 255, 0)
        self.path_color = (200, 200, 0)

    def generate_obstacles(self):
        """Gera obstáculos com sensação de linha de montagem."""
        for _ in range(7):
            row = random.randint(5, self.maze_size - 6)
            start = random.randint(0, self.maze_size - 10)
            length = random.randint(5, 10)
            for col in range(start, start + length):
                if random.random() < 0.7: self.map[row][col] = 1
        for _ in range(7):
            col = random.randint(5, self.maze_size - 6)
            start = random.randint(0, self.maze_size - 10)
            length = random.randint(5, 10)
            for row in range(start, start + length):
                if random.random() < 0.7: self.map[row][col] = 1
        block_size = random.choice([4, 6])
        top_row = random.randint(0, self.maze_size - block_size)
        top_col = random.randint(0, self.maze_size - block_size)
        for r in range(top_row, top_row + block_size):
            for c in range(top_col, top_col + block_size):
                self.map[r][c] = 1

    def generate_player(self):
        """Cria o jogador em uma célula livre."""
        while True:
            x = random.randint(0, self.maze_size - 1)
            y = random.randint(0, self.maze_size - 1)
            if self.map[y][x] == 0 and [x, y] not in self.packages:
                # ### ALTERAÇÃO: Instancia o SmartPlayer ###
                return SmartPlayer([x, y])

    def random_free_cell(self):
        """Retorna uma célula livre aleatória."""
        while True:
            x = random.randint(0, self.maze_size - 1)
            y = random.randint(0, self.maze_size - 1)
            occupied = (
                self.map[y][x] == 1 or
                [x, y] in self.packages or
                [x, y] == self.player.position or
                any(g["pos"] == [x, y] for g in self.goals)
            )
            if not occupied: return [x, y]

    def add_goal(self, created_at_step):
        pos = self.random_free_cell()
        priority = random.randint(40, 110)
        self.goals.append({"pos": pos, "priority": priority, "created_at": created_at_step})

    def can_move_to(self, pos):
        x, y = pos
        if 0 <= x < self.maze_size and 0 <= y < self.maze_size:
            return self.map[y][x] == 0
        return False

    def draw_world(self, path=None):
        self.screen.fill(self.ground_color)
        for (x, y) in self.walls:
            rect = pygame.Rect(x * self.block_size, y * self.block_size, self.block_size, self.block_size)
            pygame.draw.rect(self.screen, self.wall_color, rect)
        for pkg in self.packages:
            self.screen.blit(self.package_image, (pkg[0] * self.block_size, pkg[1] * self.block_size))
        for goal in self.goals:
            self.screen.blit(self.goal_image, (goal["pos"][0] * self.block_size, goal["pos"][1] * self.block_size))
        if path:
            for pos in path:
                rect = pygame.Rect(pos[0] * self.block_size + self.block_size // 4,
                                   pos[1] * self.block_size + self.block_size // 4,
                                   self.block_size // 2, self.block_size // 2)
                pygame.draw.rect(self.screen, self.path_color, rect)
        x, y = self.player.position
        rect = pygame.Rect(x * self.block_size, y * self.block_size, self.block_size, self.block_size)
        pygame.draw.rect(self.screen, self.player_color, rect)
        pygame.display.flip()

# ==========================
# CLASSE MAZE: Lógica do jogo e planejamento de caminhos (A*)
# ==========================
class Maze:
    def __init__(self, seed=None):
        self.world = World(seed)
        self.running = True
        self.score = 0
        self.steps = 0
        self.delay = 100
        self.path = []
        self.num_deliveries = 0

        self.world.add_goal(created_at_step=0)
        self.spawn_intervals = [random.randint(2, 5)] + [random.randint(5, 10)] + [random.randint(10, 15) for _ in range(3)]
        self.next_spawn_step = self.spawn_intervals.pop(0)

        self.current_target = None
        self.initial_wait_completed = False

    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def greedy_bfs(self, start, goal):
        """
        Implementação da Busca Gulosa pelo Melhor Primeiro (Greedy Best-First Search).
        """
        maze = self.world.map
        size = self.world.maze_size
        neighbors = [(0, -1), (0, 1), (-1, 0), (1, 0)]

        # Fila de prioridade armazena (prioridade, nó)
        # A prioridade é apenas a distância heurística até o objetivo.
        pqueue = []
        heapq.heappush(pqueue, (self.heuristic(start, goal), start))
        
        # Dicionário para reconstruir o caminho
        came_from = {tuple(start): None}

        while pqueue:
            # Pega o nó que parece estar mais perto do objetivo
            current_node = heapq.heappop(pqueue)[1]

            if current_node == goal:
                path = []
                temp = tuple(current_node)
                while temp is not None:
                    path.append(list(temp))
                    temp = came_from.get(temp)
                # O caminho é construído do fim para o início, então o removemos
                # e o revertemos. O ponto inicial é removido.
                path.pop() 
                path.reverse()
                return path

            for dx, dy in neighbors:
                neighbor = [current_node[0] + dx, current_node[1] + dy]
                neighbor_tuple = tuple(neighbor)

                if (0 <= neighbor[0] < size and
                    0 <= neighbor[1] < size and
                    maze[neighbor[1]][neighbor[0]] == 0 and
                    neighbor_tuple not in came_from):
                    
                    # A prioridade é calculada apenas com base na heurística
                    priority = self.heuristic(neighbor, goal)
                    heapq.heappush(pqueue, (priority, neighbor))
                    came_from[neighbor_tuple] = tuple(current_node)
        
        return []

    def maybe_spawn_goal(self):
        while self.next_spawn_step is not None and self.steps >= self.next_spawn_step:
            self.world.add_goal(created_at_step=self.steps)
            if self.spawn_intervals:
                self.next_spawn_step += self.spawn_intervals.pop(0)
            else:
                self.next_spawn_step = None

    def delayed_goals_penalty(self):
        delayed = 0
        for g in self.world.goals:
            age = self.steps - g["created_at"]
            if age > g["priority"]:
                delayed += 1
        return delayed

    def get_goal_at(self, pos):
        for g in self.world.goals:
            if g["pos"] == pos:
                return g
        return None

    def idle_tick(self):
        self.steps += 1
        self.score -= 1
        self.score -= self.delayed_goals_penalty()
        self.maybe_spawn_goal()
        self.world.draw_world(self.path)
        pygame.time.wait(self.delay)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

    def game_loop(self):
        while self.running:
            if not self.initial_wait_completed:
                wait_steps = 0000 // self.delay
                if self.steps < wait_steps:
                    self.idle_tick()
                    if not self.running: break
                    continue
                else:
                    self.initial_wait_completed = True
                    print("Espera inicial concluída. Iniciando movimento!")

            if self.num_deliveries >= self.world.total_items:
                self.running = False
                break

            self.maybe_spawn_goal()

            if self.current_target is None:
                target = self.world.player.escolher_alvo(self.world, self.steps, self)
                if target is None:
                    self.idle_tick()
                    continue
                self.current_target = target

            self.path = self.greedy_bfs(self.world.player.position, self.current_target)
            if not self.path:
                print("Nenhum caminho encontrado para o alvo", self.current_target)
                self.current_target = None 
                self.idle_tick()
                continue
            
            # Movimento do jogador
            for pos in self.path:
                self.world.player.position = pos
                self.steps += 1
                self.score -= 1
                self.score -= self.delayed_goals_penalty()
                self.maybe_spawn_goal()
                self.world.draw_world(self.path)
                pygame.time.wait(self.delay)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                if not self.running: break
            if not self.running: break

            # Lógica de Coleta e Entrega
            if self.world.player.position == self.current_target:
                if self.current_target in self.world.packages:
                    self.world.player.cargo += 1
                    self.world.packages.remove(self.current_target)
                    print(f"Pacote coletado em {self.current_target}. Cargo: {self.world.player.cargo}")
                else:
                    goal = self.get_goal_at(self.current_target)
                    if goal is not None and self.world.player.cargo > 0:
                        self.world.player.cargo -= 1
                        self.num_deliveries += 1
                        self.world.goals.remove(goal)
                        # A lógica de limpar o plano do jogador foi removida daqui,
                        # pois agora a decisão é tomada a cada ciclo.
                        self.score += 50
                        print(f"Pacote entregue em {self.current_target} | Cargo: {self.world.player.cargo} | Prio: {goal['priority']} | Idade: {self.steps - goal['created_at']}")

            self.current_target = None
            delayed_count = sum(1 for g in self.world.goals if (self.steps - g["created_at"]) > g["priority"])
            print(f"Passos: {self.steps}, Pontuação: {self.score}, Cargo: {self.world.player.cargo}, Entregas: {self.num_deliveries}, Atrasados: {delayed_count}")

        print("\nFim de jogo!")
        print("Total de passos:", self.steps)
        print("Pontuação final:", self.score)
        pygame.quit()

# ==========================
# PONTO DE ENTRADA PRINCIPAL
# ==========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delivery Bot: Navegue, colete e entregue.")
    parser.add_argument("--seed", type=int, help="Seed para recriar o mesmo mundo.")
    args = parser.parse_args()

    maze = Maze(seed=args.seed)
    maze.game_loop()