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
    def escolher_alvo(self, world, current_steps, maze_solver):
        """
        Retorna o alvo (posição) que o jogador deseja ir.
        """
        pass

class DefaultPlayer(BasePlayer):
    """
    Implementação padrão do jogador (lógica antiga).
    """
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
        elif self.cargo > 0 and world.goals:
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
# ### CLASSE SMARTPLAYER USANDO DIJKSTRA PARA TOMADA DE DECISÃO ###
# ######################################################################################
class SmartPlayer(BasePlayer):
    """
    Implementação de um jogador inteligente que usa Dijkstra para suas avaliações.
    """
    def __init__(self, position):
        super().__init__(position)
        self.max_cargo = 2

    def _evaluate_best_goal(self, world, current_steps, maze_solver):
        best_goal = None
        best_score = -math.inf
        
        for goal in world.goals:
            path_to_goal = maze_solver.dijkstra(self.position, goal["pos"])
            dist_to_goal = len(path_to_goal)
            
            if not path_to_goal and self.position != goal["pos"]: continue

            arrival_time_at_goal = current_steps + dist_to_goal
            deadline = goal["created_at"] + goal["priority"]
            lateness = max(0, arrival_time_at_goal - deadline)

            reward = 50 
            score = reward - dist_to_goal - (lateness * 1.5)
            
            if score > best_score:
                best_score = score
                best_goal = goal
        
        return best_goal, best_score

    def _evaluate_best_package(self, world, current_steps, maze_solver):
        best_pkg = None
        best_score = -math.inf

        for package in world.packages:
            path_to_pkg = maze_solver.dijkstra(self.position, package)
            dist_to_pkg = len(path_to_pkg)
            
            if not path_to_pkg and self.position != package: continue
            
            score = 10 - dist_to_pkg 
            
            if score > best_score:
                best_score = score
                best_pkg = package
                
        return best_pkg, best_score

    def escolher_alvo(self, world, current_steps, maze_solver):
        can_pickup_more = (self.cargo < self.max_cargo) and \
                          (self.cargo < len(world.goals)) and \
                          world.packages

        if self.cargo > 0 and not can_pickup_more:
            if not world.goals: return None
            best_goal, _ = self._evaluate_best_goal(world, current_steps, maze_solver)
            return best_goal["pos"] if best_goal else None

        if self.cargo == 0 and world.packages:
            best_pkg, _ = self._evaluate_best_package(world, current_steps, maze_solver)
            return best_pkg

        if self.cargo > 0 and can_pickup_more:
            best_goal, goal_score = self._evaluate_best_goal(world, current_steps, maze_solver)
            best_pkg, pkg_score = self._evaluate_best_package(world, current_steps, maze_solver)

            if best_goal and best_pkg:
                if goal_score >= pkg_score:
                    return best_goal["pos"]
                else:
                    return best_pkg
            elif best_goal:
                return best_goal["pos"]
            elif best_pkg:
                return best_pkg

        return None

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
                return SmartPlayer([x, y])

    def random_free_cell(self):
        while True:
            x, y = random.randint(0, self.maze_size - 1), random.randint(0, self.maze_size - 1)
            occupied = (self.map[y][x] == 1 or [x, y] in self.packages or [x, y] == self.player.position or any(g["pos"] == [x, y] for g in self.goals))
            if not occupied: return [x, y]

    def add_goal(self, created_at_step):
        pos = self.random_free_cell()
        priority = random.randint(40, 110)
        self.goals.append({"pos": pos, "priority": priority, "created_at": created_at_step})

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

    def dijkstra(self, start, goal):
        maze = self.world.map
        size = self.world.maze_size
        neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        oheap = []
        heapq.heappush(oheap, (0, tuple(start)))
        came_from = {}
        distances = {tuple(start): 0}

        while oheap:
            current_dist, current_node = heapq.heappop(oheap)
            if list(current_node) == goal:
                path = []
                while current_node in came_from:
                    path.append(list(current_node))
                    current_node = came_from[current_node]
                path.reverse()
                return path
            if current_dist > distances.get(current_node, float('inf')):
                continue
            for dx, dy in neighbors:
                neighbor = (current_node[0] + dx, current_node[1] + dy)
                if not (0 <= neighbor[0] < size and 0 <= neighbor[1] < size and maze[neighbor[1]][neighbor[0]] == 0):
                    continue
                new_dist = current_dist + 1
                if new_dist < distances.get(neighbor, float('inf')):
                    distances[neighbor] = new_dist
                    came_from[neighbor] = current_node
                    heapq.heappush(oheap, (new_dist, neighbor))
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
                self.current_target = self.world.player.escolher_alvo(self.world, self.steps, self)
                if self.current_target is None:
                    self.tick()
                    continue
            
            path = self.dijkstra(self.world.player.position, self.current_target)
            
            if not path and self.world.player.position != self.current_target:
                if self.verbose: print(f"[INFO] Alvo {self.current_target} inacessível com Dijkstra. Reposicionando.")
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
                    if self.world.player.cargo < self.world.player.max_cargo:
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
    parser.add_argument("--arquivo_saida", type=str, default="simulacao-SmartPlayer-Dijkstra.csv", help="Nome do arquivo CSV de saída (Agente Inteligente com Dijkstra).")
    parser.add_argument("--seed", type=int, default=None, help="Semente para o gerador de números aleatórios.")
    parser.add_argument('--verbose', action='store_true', help="Imprime logs de status durante a simulação.")
    parser.add_argument('--timeout', type=int, default=45, help="Tempo máximo em segundos que cada simulação pode durar.")
    args = parser.parse_args()

    TIMEOUT_SECONDS = args.timeout
    file_exists = os.path.exists(args.arquivo_saida)
    
    with open(args.arquivo_saida, 'a', newline='') as csvfile:
        fieldnames = ['Id da RUN', 'Score', 'Deliveres', 'steps', 'seed', 'status']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        num_runs = 1 if args.seed is not None else args.simulacoes
        print(f"Executando {num_runs} simulação(ões) com SmartPlayer (Dijkstra). Timeout: {TIMEOUT_SECONDS}s. Saída: '{args.arquivo_saida}'...")

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