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
    def escolher_alvo(self, world, current_steps):
        """
        Retorna o alvo (posição) que o jogador deseja ir.
        Recebe o objeto world para acesso a pacotes e metas.
        """
        pass

class DefaultPlayer(BasePlayer):

    def get_remaining_steps(self, goal, current_steps):
        prioridade = goal["priority"]
        idade = current_steps - goal["created_at"]
        print(f"Goal em {goal['pos']} tem prioridade {prioridade} e idade {idade}")
        return prioridade - idade
    
    """
    Implementação de um jogador mais inteligente para múltiplas cargas.
    Agora ele avalia se é melhor coletar ou entregar.
    """
    def escolher_alvo(self, world, current_steps):
        # ================================================================= #
        # ### INÍCIO DA ALTERAÇÃO: LÓGICA DE DECISÃO INTELIGENTE ###
        # ================================================================= #
        
        max_cargo = 3
        sx, sy = self.position

        # --- CASO 1: DEVE COLETAR ---
        # Se a carga está vazia, a única opção lógica é procurar um pacote.
        if self.cargo == 0 and world.packages:
            print(f"INFO: Carga vazia. Procurando o pacote mais próximo...")
            best_pkg = None
            best_dist = math.inf
            for pkg in world.packages:
                d = abs(pkg[0] - sx) + abs(pkg[1] - sy)
                if d < best_dist:
                    best_dist = d
                    best_pkg = pkg
            return best_pkg

        # --- CASO 2: DEVE ENTREGAR ---
        # Se a carga está cheia (ou se não há mais pacotes para pegar), ele deve entregar.
        elif (self.cargo >= max_cargo or not world.packages) and self.cargo > 0 and world.goals:
            print(f"INFO: Carga cheia ({self.cargo}/{max_cargo}) ou sem pacotes. Procurando entrega...")
            best_goal = None
            best_dist = math.inf
            for goal in world.goals:
                gx, gy = goal["pos"]
                d = abs(gx - sx) + abs(gy - sy)
                if d < best_dist:
                    best_dist = d
                    best_goal = goal["pos"]
            return best_goal

        # --- CASO 3: A ESCOLHA ESTRATÉGICA ---
        # Se ele tem carga (1 ou 2) e AINDA existem pacotes e objetivos, ele decide!
        elif self.cargo > 0 and world.packages and world.goals:
            # 1. Encontrar o pacote mais próximo e sua distância
            best_pkg = None
            best_dist_pkg = math.inf
            for pkg in world.packages:
                d = abs(pkg[0] - sx) + abs(pkg[1] - sy)
                if d < best_dist_pkg:
                    best_dist_pkg = d
                    best_pkg = pkg

            # 2. Encontrar o objetivo mais próximo e sua distância
            best_goal = None
            best_dist_goal = math.inf
            for goal in world.goals:
                gx, gy = goal["pos"]
                d = abs(gx - sx) + abs(gy - sy)
                if d < best_dist_goal:
                    best_dist_goal = d
                    best_goal = goal["pos"]

            # 3. Comparar e decidir o que está mais perto
            print(f"DECISÃO: Comparando... Dist. Pacote: {best_dist_pkg} vs Dist. Objetivo: {best_dist_goal}")
            if best_dist_pkg <= best_dist_goal:
                print("--> ESCOLHA: Pegar outro pacote está mais perto.")
                return best_pkg
            else:
                print("--> ESCOLHA: Entregar um pacote está mais perto.")
                return best_goal
        
        # Se chegou até aqui, mas só tem carga e nenhum objetivo, espera.
        elif self.cargo > 0 and world.goals:
             print(f"INFO: Carga ({self.cargo}/{max_cargo}) mas sem pacotes restantes. Procurando entrega...")
             best_goal = None
             best_dist = math.inf
             for goal in world.goals:
                gx, gy = goal["pos"]
                d = abs(gx - sx) + abs(gy - sy)
                if d < best_dist:
                    best_dist = d
                    best_goal = goal["pos"]
             return best_goal

        # Se nenhuma das condições for atendida, não faz nada.
        else:
            return None
        
        # =============================================================== #
        # ### FIM DA ALTERAÇÃO ###
        # =============================================================== #

# ==========================
# CLASSE WORLD (MUNDO)
# (O restante do código permanece o mesmo)
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

        # Cria o jogador usando a classe DefaultPlayer
        self.player = self.generate_player()

        # Inicializa a janela do Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Delivery Bot")

        # Carrega imagens para pacote e meta a partir de arquivos
        self.package_image = pygame.image.load("images/cargo.png")
        self.package_image = pygame.transform.scale(self.package_image, (self.block_size, self.block_size))

        self.goal_image = pygame.image.load("images/operator.png")
        self.goal_image = pygame.transform.scale(self.goal_image, (self.block_size, self.block_size))

        # Cores utilizadas para desenho
        self.wall_color = (100, 100, 100)
        self.ground_color = (255, 255, 255)
        self.player_color = (0, 255, 0)
        self.path_color = (200, 200, 0)

    def generate_obstacles(self):
        """
        Gera obstáculos com sensação de linha de montagem.
        """
        for _ in range(7):
            row = random.randint(5, self.maze_size - 6)
            start = random.randint(0, self.maze_size - 10)
            length = random.randint(5, 10)
            for col in range(start, start + length):
                if random.random() < 0.7:
                    self.map[row][col] = 1

        for _ in range(7):
            col = random.randint(5, self.maze_size - 6)
            start = random.randint(0, self.maze_size - 10)
            length = random.randint(5, 10)
            for row in range(start, start + length):
                if random.random() < 0.7:
                    self.map[row][col] = 1

        block_size = random.choice([4, 6])
        max_row = self.maze_size - block_size
        max_col = self.maze_size - block_size
        top_row = random.randint(0, max_row)
        top_col = random.randint(0, max_col)
        for r in range(top_row, top_row + block_size):
            for c in range(top_col, top_col + block_size):
                self.map[r][c] = 1

    def generate_player(self):
        while True:
            x = random.randint(0, self.maze_size - 1)
            y = random.randint(0, self.maze_size - 1)
            if self.map[y][x] == 0 and [x, y] not in self.packages:
                return DefaultPlayer([x, y])

    def random_free_cell(self):
        while True:
            x = random.randint(0, self.maze_size - 1)
            y = random.randint(0, self.maze_size - 1)
            occupied = (
                self.map[y][x] == 1 or
                [x, y] in self.packages or
                [x, y] == self.player.position or
                any(g["pos"] == [x, y] for g in self.goals)
            )
            if not occupied:
                return [x, y]

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
            x, y = pkg
            self.screen.blit(self.package_image, (x * self.block_size, y * self.block_size))
        for goal in self.goals:
            x, y = goal["pos"]
            self.screen.blit(self.goal_image, (x * self.block_size, y * self.block_size))
        if path:
            for pos in path:
                x, y = pos
                rect = pygame.Rect(x * self.block_size + self.block_size // 4,
                                   y * self.block_size + self.block_size // 4,
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

    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def astar(self, start, goal):
        maze = self.world.map
        size = self.world.maze_size
        neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        close_set = set()
        came_from = {}
        gscore = {tuple(start): 0}
        fscore = {tuple(start): self.heuristic(start, goal)}
        oheap = []
        heapq.heappush(oheap, (fscore[tuple(start)], tuple(start)))
        while oheap:
            current = heapq.heappop(oheap)[1]
            if list(current) == goal:
                data = []
                while current in came_from:
                    data.append(list(current))
                    current = came_from[current]
                data.reverse()
                return data
            close_set.add(current)
            for dx, dy in neighbors:
                neighbor = (current[0] + dx, current[1] + dy)
                tentative_g = gscore[current] + 1
                if 0 <= neighbor[0] < size and 0 <= neighbor[1] < size:
                    if maze[neighbor[1]][neighbor[0]] == 1:
                        continue
                else:
                    continue
                if neighbor in close_set and tentative_g >= gscore.get(neighbor, 0):
                    continue
                if tentative_g < gscore.get(neighbor, float('inf')) or neighbor not in [i[1] for i in oheap]:
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g
                    fscore[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(oheap, (fscore[neighbor], neighbor))
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

    def game_loop(self):
        while self.running:
            if self.num_deliveries >= self.world.total_items:
                self.running = False
                break

            self.maybe_spawn_goal()

            if self.current_target is None:
                target = self.world.player.escolher_alvo(self.world, self.steps)
                if target is None:
                    self.idle_tick()
                    continue
                self.current_target = target

            self.path = self.astar(self.world.player.position, self.current_target)
            if not self.path:
                print("Nenhum caminho encontrado para o alvo", self.current_target)
                self.running = False
                break

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
                        break
                if not self.running:
                    break

            if not self.running:
                break

            if self.world.player.position == self.current_target:
                if self.current_target in self.world.packages:
                    self.world.player.cargo += 1
                    self.world.packages.remove(self.current_target)
                    print("Pacote coletado em", self.current_target, "Cargo agora:", self.world.player.cargo)
                else:
                    goal = self.get_goal_at(self.current_target)
                    if goal is not None and self.world.player.cargo > 0:
                        self.world.player.cargo -= 1
                        self.num_deliveries += 1
                        self.world.goals.remove(goal)
                        self.score += 50
                        print(
                            f"Pacote entregue em {self.current_target} | "
                            f"Cargo: {self.world.player.cargo} | "
                            f"Priority: {goal['priority']} | "
                            f"Age: {self.steps - goal['created_at']}"
                        )

            self.current_target = None

            delayed_count = sum(1 for g in self.world.goals if (self.steps - g["created_at"]) > g["priority"])
            print(
                f"Passos: {self.steps}, Pontuação: {self.score}, Cargo: {self.world.player.cargo}, "
                f"Entregas: {self.num_deliveries}, Goals ativos: {len(self.world.goals)}, "
                f"Atrasados: {delayed_count}"
            )

        print("Fim de jogo!")
        print("Total de passos:", self.steps)
        print("Pontuação final:", self.score)
        pygame.quit()

# ==========================
# PONTO DE ENTRADA PRINCIPAL
# ==========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Delivery Bot: Navegue no grid, colete pacotes e realize entregas."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Valor do seed para recriar o mesmo mundo (opcional)."
    )
    args = parser.parse_args()

    maze = Maze(seed=args.seed)
    maze.game_loop()