import random
import heapq
import sys
import argparse
from abc import ABC, abstractmethod
import math
import csv
import os
import multiprocessing
import queue

# ==========================
# CLASSES DE PLAYER
# ==========================
class BasePlayer(ABC):
    """
    Classe base para o jogador (robô).
    """
    def __init__(self, position):
        self.position = position
        self.cargo = 0

    @abstractmethod
    def escolher_alvo(self, world, current_steps):
        """
        Retorna o alvo (posição) que o jogador deseja ir.
        """
        pass

class DefaultPlayer(BasePlayer):
    """
    Implementação de um jogador mais inteligente para múltiplas cargas.
    Agora ele avalia se é melhor coletar ou entregar.
    """
    # Removido o 'print' para não poluir a saída de milhares de simulações.
    def get_remaining_steps(self, goal, current_steps):
        prioridade = goal["priority"]
        idade = current_steps - goal["created_at"]
        return prioridade - idade
    
    def escolher_alvo(self, world, current_steps):
        # ============================================================== #
        # ### INÍCIO DA LÓGICA DE DECISÃO INTELIGENTE (PRESERVADA) ###
        # ============================================================== #
        
        max_cargo = 3
        sx, sy = self.position

        # --- CASO 1: DEVE COLETAR ---
        # Se a carga está vazia, a única opção lógica é procurar um pacote.
        if self.cargo == 0 and world.packages:
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
            if best_dist_pkg <= best_dist_goal:
                return best_pkg
            else:
                return best_goal
        
        # Se chegou até aqui, mas só tem carga e nenhum pacote restante, deve entregar.
        elif self.cargo > 0 and world.goals:
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
        # ============================================================ #
        # ### FIM DA LÓGICA DE DECISÃO ###
        # ============================================================ #

# ==========================
# CLASSE WORLD (MUNDO) - Versão "Headless"
# ==========================
class World:
    def __init__(self, seed=None):
        if seed is not None:
            random.seed(seed)
        self.maze_size = 30
        self.map = [[0 for _ in range(self.maze_size)] for _ in range(self.maze_size)]
        self.generate_obstacles()
        self.total_items = 6
        self.packages = []
        while len(self.packages) < self.total_items + 1:
            x, y = random.randint(0, self.maze_size - 1), random.randint(0, self.maze_size - 1)
            if self.map[y][x] == 0 and [x, y] not in self.packages:
                self.packages.append([x, y])
        self.goals = []
        # Usa a classe DefaultPlayer com a nova lógica
        self.player = self.generate_player()

    def generate_obstacles(self):
        for _ in range(7):
            row, start, length = random.randint(5, self.maze_size - 6), random.randint(0, self.maze_size - 10), random.randint(5, 10)
            for col in range(start, start + length):
                if random.random() < 0.7: self.map[row][col] = 1
        for _ in range(7):
            col, start, length = random.randint(5, self.maze_size - 6), random.randint(0, self.maze_size - 10), random.randint(5, 10)
            for row in range(start, start + length):
                if random.random() < 0.7: self.map[row][col] = 1
        block_size = random.choice([4, 6])
        top_row, top_col = random.randint(0, self.maze_size - block_size), random.randint(0, self.maze_size - block_size)
        for r in range(top_row, top_row + block_size):
            for c in range(top_col, top_col + block_size): self.map[r][c] = 1

    def generate_player(self):
        while True:
            x, y = random.randint(0, self.maze_size - 1), random.randint(0, self.maze_size - 1)
            if self.map[y][x] == 0 and [x, y] not in self.packages:
                return DefaultPlayer([x, y])

    def random_free_cell(self):
        while True:
            x, y = random.randint(0, self.maze_size - 1), random.randint(0, self.maze_size - 1)
            occupied = (self.map[y][x] == 1 or [x, y] in self.packages or [x, y] == self.player.position or any(g["pos"] == [x, y] for g in self.goals))
            if not occupied: return [x, y]

    def add_goal(self, created_at_step):
        self.goals.append({"pos": self.random_free_cell(), "priority": random.randint(40, 110), "created_at": created_at_step})

# ==========================
# CLASSE MAZE (Lógica do Jogo)
# ==========================
class Maze:
    def __init__(self, seed=None, verbose=False):
        self.world = World(seed)
        self.verbose = verbose
        self.running = True
        self.score = 0
        self.steps = 0
        self.num_deliveries = 0
        self.world.add_goal(created_at_step=0)
        self.spawn_intervals = [random.randint(2, 5)] + [random.randint(5, 10)] + [random.randint(10, 15) for _ in range(3)]
        self.next_spawn_step = self.spawn_intervals.pop(0)
        self.current_target = None

    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def astar(self, start, goal):
        maze, size = self.world.map, self.world.maze_size
        neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        close_set, came_from, gscore = set(), {}, {tuple(start): 0}
        fscore = {tuple(start): self.heuristic(start, goal)}
        oheap = [(fscore[tuple(start)], tuple(start))]
        while oheap:
            current = heapq.heappop(oheap)[1]
            if list(current) == goal:
                path = []
                while current in came_from:
                    path.append(list(current))
                    current = came_from[current]
                return path[::-1]
            close_set.add(current)
            for dx, dy in neighbors:
                neighbor = (current[0] + dx, current[1] + dy)
                if not (0 <= neighbor[0] < size and 0 <= neighbor[1] < size and maze[neighbor[1]][neighbor[0]] == 0): continue
                tentative_g = gscore[current] + 1
                if neighbor in close_set and tentative_g >= gscore.get(neighbor, 0): continue
                if tentative_g < gscore.get(neighbor, float('inf')) or neighbor not in [i[1] for i in oheap]:
                    came_from[neighbor], gscore[neighbor] = current, tentative_g
                    fscore[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(oheap, (fscore[neighbor], neighbor))
        return []

    def maybe_spawn_goal(self):
        if self.next_spawn_step is not None and self.steps >= self.next_spawn_step:
            self.world.add_goal(created_at_step=self.steps)
            if self.spawn_intervals:
                self.next_spawn_step += self.spawn_intervals.pop(0)
            else:
                self.next_spawn_step = None

    def delayed_goals_penalty(self):
        return sum(1 for g in self.world.goals if self.steps - g["created_at"] > g["priority"])

    def get_goal_at(self, pos):
        return next((g for g in self.world.goals if g["pos"] == pos), None)

    def tick(self):
        self.steps += 1
        self.score -= 1 + self.delayed_goals_penalty()
        self.maybe_spawn_goal()

    def game_loop(self):
        while self.running:
            if self.num_deliveries >= self.world.total_items:
                self.running = False
                break
            
            self.maybe_spawn_goal()

            if self.current_target is None:
                self.current_target = self.world.player.escolher_alvo(self.world, self.steps)
                if self.current_target is None:
                    self.tick()
                    continue
            
            path = self.astar(self.world.player.position, self.current_target)
            
            if not path and self.world.player.position != self.current_target:
                if self.verbose: print(f"[INFO] Alvo {self.current_target} inacessível com A*. Reposicionando.")
                if self.current_target in self.world.packages:
                    self.world.packages.remove(self.current_target)
                    self.world.packages.append(self.world.random_free_cell())
                else:
                    goal = self.get_goal_at(self.current_target)
                    if goal:
                        self.world.goals.remove(goal)
                        self.world.add_goal(self.steps)
                self.current_target = None
                self.tick()
                continue
            
            for pos in path:
                self.world.player.position = pos
                self.tick()

            if self.world.player.position == self.current_target:
                if self.current_target in self.world.packages:
                    # Permite carregar múltiplos pacotes até o máximo definido na classe do player
                    if self.world.player.cargo < 3: # max_cargo
                        self.world.player.cargo += 1
                        self.world.packages.remove(self.current_target)
                else:
                    goal = self.get_goal_at(self.current_target)
                    if goal and self.world.player.cargo > 0:
                        self.world.player.cargo -= 1
                        self.num_deliveries += 1
                        self.world.goals.remove(goal)
                        self.score += 50
                        game_is_not_over = self.num_deliveries < self.world.total_items
                        if not self.world.goals and game_is_not_over:
                            if self.verbose: print(f"💡 [INFO] Último goal utilizado. Criando um novo para evitar impasse.")
                            self.world.add_goal(self.steps)
            
            self.current_target = None
        
        return {"score": self.score, "deliveries": self.num_deliveries, "steps": self.steps}

# ==========================
# PONTO DE ENTRADA PRINCIPAL (COM TIMEOUT E CSV)
# ==========================
def run_simulation_worker(seed, verbose, result_queue):
    try:
        maze = Maze(seed=seed, verbose=verbose)
        resultados = maze.game_loop()
        result_queue.put(resultados)
    except Exception as e:
        print(f"Erro na simulação com seed {seed}: {e}")
        result_queue.put(None)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delivery Bot: Simulação de logística com saída em CSV e timeout.")
    parser.add_argument("--simulacoes", type=int, default=150, help="Número de simulações a serem executadas.")
    parser.add_argument("--arquivo_saida", type=str, default="simulacao-MultiCargo.csv", help="Nome do arquivo CSV de saída (Agente Multi-Carga).")
    parser.add_argument("--seed", type=int, default=None, help="Semente para o gerador de números aleatórios.")
    parser.add_argument('--verbose', action='store_true', help="Imprime logs de status durante a simulação.")
    parser.add_argument('--timeout', type=int, default=20, help="Tempo máximo em segundos que cada simulação pode durar.") # Aumentado por causa da lógica mais complexa
    args = parser.parse_args()

    TIMEOUT_SECONDS = args.timeout
    file_exists = os.path.exists(args.arquivo_saida)
    
    with open(args.arquivo_saida, 'a', newline='') as csvfile:
        fieldnames = ['Id da RUN', 'Score', 'Deliveres', 'steps', 'seed', 'status']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        num_runs = 1 if args.seed is not None else args.simulacoes
        print(f"Executando {num_runs} simulação(ões) com Agente Multi-Carga. Timeout: {TIMEOUT_SECONDS}s. Saída: '{args.arquivo_saida}'...")

        for i in range(1, num_runs + 1):
            current_seed = args.seed if args.seed is not None else i
            print(f"Iniciando simulação {i}/{num_runs} (Seed: {current_seed})...", end='\r')

            result_queue = multiprocessing.Queue()
            simulation_process = multiprocessing.Process(target=run_simulation_worker, args=(current_seed, args.verbose, result_queue))
            simulation_process.start()
            simulation_process.join(timeout=TIMEOUT_SECONDS)

            if simulation_process.is_alive():
                print(f"\n⚠️ Simulação {i}/{num_runs} (Seed: {current_seed}) excedeu o tempo limite. Abortando.")
                simulation_process.terminate()
                simulation_process.join()
                writer.writerow({'Id da RUN': i, 'Score': 'N/A', 'Deliveres': 'N/A', 'steps': 'N/A', 'seed': current_seed, 'status': 'timeout'})
            else:
                try:
                    resultados = result_queue.get_nowait()
                    if resultados:
                        writer.writerow({'Id da RUN': i, 'Score': resultados['score'], 'Deliveres': resultados['deliveries'], 'steps': resultados['steps'], 'seed': current_seed, 'status': 'completed'})
                    else:
                        writer.writerow({'Id da RUN': i, 'Score': 'N/A', 'Deliveres': 'N/A', 'steps': 'N/A', 'seed': current_seed, 'status': 'error'})
                except queue.Empty:
                    print(f"\n❌ Erro: Simulação {i}/{num_runs} (Seed: {current_seed}) terminou sem retornar resultados.")
                    writer.writerow({'Id da RUN': i, 'Score': 'N/A', 'Deliveres': 'N/A', 'steps': 'N/A', 'seed': current_seed, 'status': 'crashed'})

    print(f"\n\nConcluído! Os resultados de {num_runs} simulação(ões) foram salvos em '{args.arquivo_saida}'.")