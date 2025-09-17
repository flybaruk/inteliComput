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
    def __init__(self, position):
        self.position = tuple(position)
        self.cargo = 0

    @abstractmethod
    def escolher_alvo(self, world, current_steps, maze_solver):
        pass

class ScoreGreedyPlayer(BasePlayer):
    """
    Estratégia "Gananciosa por Pontos".
    A cada turno, compara a pontuação da melhor entrega possível com a pontuação
    da melhor coleta+entrega possível e escolhe a ação mais lucrativa.
    """
    def __init__(self, position):
        super().__init__(position)
        self.max_cargo = 99 

    def escolher_alvo(self, world, current_steps, maze_solver):
        best_delivery_score = -math.inf
        best_delivery_target = None
        
        best_collection_score = -math.inf
        best_collection_target = None

        if self.cargo > 0 and world.goals:
            goal_obj, score = self._evaluate_best_goal(world, current_steps, maze_solver)
            if goal_obj:
                best_delivery_score = score
                best_delivery_target = goal_obj["pos"]

        if self.cargo < self.max_cargo and world.packages and world.goals:
            pkg_target, score = self._evaluate_best_package_holistic(world, current_steps, maze_solver)
            if pkg_target:
                best_collection_score = score
                best_collection_target = pkg_target
        
        if maze_solver.verbose:
            print(f"DECISÃO: Score Entrega: {best_delivery_score:.2f} vs Score Coleta: {best_collection_score:.2f}")

        if best_delivery_score >= best_collection_score:
            if maze_solver.verbose: print("--> ESCOLHA: Entregar é mais lucrativo.")
            return best_delivery_target
        else:
            if maze_solver.verbose: print("--> ESCOLHA: Coletar é mais lucrativo.")
            return best_collection_target

    def _evaluate_best_goal(self, world, current_steps, maze_solver):
        best_goal, best_score = None, -math.inf
        for goal in world.goals:
            path = maze_solver.astar(self.position, goal["pos"])
            dist = len(path)
            if not path and self.position != tuple(goal["pos"]): continue
            lateness = max(0, (current_steps + dist) - (goal["created_at"] + goal["priority"]))
            score = 50 - dist - (lateness * 1.5)
            if score > best_score:
                best_score, best_goal = score, goal
        return best_goal, best_score

    def _evaluate_best_package_holistic(self, world, current_steps, maze_solver):
        best_pkg_in_plan, best_overall_score = None, -math.inf
        for package in world.packages:
            dist_to_pkg = len(maze_solver.astar(self.position, package))
            if dist_to_pkg == 0 and self.position != tuple(package): continue
            
            best_future_score_from_pkg = -math.inf
            for goal in world.goals:
                dist_pkg_to_goal = len(maze_solver.astar(package, goal["pos"]))
                if dist_pkg_to_goal == 0 and package != tuple(goal["pos"]): continue
                
                total_dist = dist_to_pkg + dist_pkg_to_goal
                arrival = current_steps + total_dist
                deadline = goal["created_at"] + goal["priority"]
                lateness = max(0, arrival - deadline)
                score = 50 - total_dist - (lateness * 1.5)
                if score > best_future_score_from_pkg:
                    best_future_score_from_pkg = score
            
            if best_future_score_from_pkg > best_overall_score:
                best_overall_score = best_future_score_from_pkg
                best_pkg_in_plan = package
                
        return best_pkg_in_plan, best_overall_score

# ==========================
# CLASSE WORLD (Headless)
# ==========================
class World:
    def __init__(self, seed=None):
        if seed is not None: random.seed(seed)
        self.maze_size = 30
        self.map = [[0 for _ in range(self.maze_size)] for _ in range(self.maze_size)]
        self.generate_obstacles()
        self.total_items = 6
        self.packages = []
        while len(self.packages) < self.total_items + 4:
            x, y = random.randint(0, self.maze_size - 1), random.randint(0, self.maze_size - 1)
            if self.map[y][x] == 0 and (x, y) not in self.packages:
                self.packages.append((x, y))
        self.goals = []
        self.player = self.generate_player()

    def generate_obstacles(self):
        for _ in range(7):
            row, start, length = random.randint(5, self.maze_size-6), random.randint(0, self.maze_size-10), random.randint(5, 10)
            for col in range(start, start + length):
                if random.random() < 0.7: self.map[row][col] = 1
        for _ in range(7):
            col, start, length = random.randint(5, self.maze_size-6), random.randint(0, self.maze_size-10), random.randint(5, 10)
            for row in range(start, start + length):
                if random.random() < 0.7: self.map[row][col] = 1
        block_size = random.choice([4, 6])
        top_row, top_col = random.randint(0, self.maze_size-block_size), random.randint(0, self.maze_size-block_size)
        for r in range(top_row, top_row + block_size):
            for c in range(top_col, top_col + block_size): self.map[r][c] = 1

    def generate_player(self):
        while True:
            x, y = random.randint(0, self.maze_size-1), random.randint(0, self.maze_size-1)
            if self.map[y][x] == 0 and (x, y) not in self.packages:
                return ScoreGreedyPlayer((x, y))

    def random_free_cell(self):
        while True:
            x, y = random.randint(0, self.maze_size-1), random.randint(0, self.maze_size-1)
            occupied = (self.map[y][x] == 1 or (x, y) in self.packages or (x, y) == self.player.position or any(g["pos"] == (x, y) for g in self.goals))
            if not occupied: return (x, y)

    def add_goal(self, created_at_step):
        pos = self.random_free_cell()
        priority = random.randint(40, 110)
        self.goals.append({"pos": pos, "priority": priority, "created_at": created_at_step})

# ==========================
# CLASSE MAZE (Headless)
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
        self.next_spawn_step = self.spawn_intervals.pop(0) if self.spawn_intervals else None
        self.current_target = None

    def astar(self, start, goal):
        start_t, goal_t = tuple(start), tuple(goal)
        if start_t == goal_t: return []
        maze, size = self.world.map, self.world.maze_size
        neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        close_set, came_from, gscore = set(), {}, {start_t: 0}
        fscore = {start_t: abs(start_t[0] - goal_t[0]) + abs(start_t[1] - goal_t[1])}
        oheap = [(fscore[start_t], start_t)]
        while oheap:
            current = heapq.heappop(oheap)[1]
            if current == goal_t:
                path = []
                while current in came_from:
                    path.append(list(current))
                    current = came_from[current]
                return path[::-1]
            close_set.add(current)
            for dx, dy in neighbors:
                neighbor = (current[0] + dx, current[1] + dy)
                tentative_g = gscore[current] + 1
                if not (0 <= neighbor[0] < size and 0 <= neighbor[1] < size and maze[neighbor[1]][neighbor[0]] == 0): continue
                if neighbor in close_set and tentative_g >= gscore.get(neighbor, float('inf')): continue
                if tentative_g < gscore.get(neighbor, float('inf')) or not any(neighbor == i[1] for i in oheap):
                    came_from[neighbor], gscore[neighbor] = current, tentative_g
                    fscore[neighbor] = tentative_g + (abs(neighbor[0] - goal_t[0]) + abs(neighbor[1] - goal_t[1]))
                    heapq.heappush(oheap, (fscore[neighbor], neighbor))
        return []

    def maybe_spawn_goal(self):
        if self.next_spawn_step is not None and self.steps >= self.next_spawn_step:
            self.world.add_goal(created_at_step=self.steps)
            self.next_spawn_step = (self.next_spawn_step + self.spawn_intervals.pop(0)) if self.spawn_intervals else None

    def delayed_goals_penalty(self):
        return sum(1 for g in self.world.goals if self.steps - g["created_at"] > g["priority"])

    def get_goal_at(self, pos):
        return next((g for g in self.world.goals if g["pos"] == tuple(pos)), None)

    def tick(self):
        self.steps += 1
        self.score -= (1 + self.delayed_goals_penalty())
        self.maybe_spawn_goal()

    def game_loop(self):
        while self.running:
            if self.num_deliveries >= self.world.total_items: self.running = False; break
            self.maybe_spawn_goal()
            if self.current_target is None:
                self.current_target = self.world.player.escolher_alvo(self.world, self.steps, self)
                if self.current_target is None: self.tick(); continue
            
            path = self.astar(self.world.player.position, self.current_target)
            if not path and self.world.player.position != self.current_target:
                if self.verbose: print(f"ALERTA: Alvo {self.current_target} inacessível.")
                self.current_target = None; continue

            for pos in path:
                self.world.player.position = tuple(pos); self.tick()
            
            if self.world.player.position == self.current_target:
                if self.current_target in self.world.packages:
                    self.world.player.cargo += 1
                    self.world.packages.remove(self.current_target)
                else:
                    goal = self.get_goal_at(self.current_target)
                    if goal and self.world.player.cargo > 0:
                        self.world.player.cargo -= 1; self.num_deliveries += 1
                        self.world.goals.remove(goal); self.score += 50
                self.current_target = None
        return {"score": self.score, "deliveries": self.num_deliveries, "steps": self.steps}

# ==========================
# PONTO DE ENTRADA PRINCIPAL
# ==========================
def run_simulation_worker(seed, verbose, result_queue):
    try:
        maze = Maze(seed=seed, verbose=verbose)
        resultados = maze.game_loop()
        result_queue.put(resultados)
    except Exception as e:
        print(f"\nERRO na simulação com seed {seed}: {e}")
        result_queue.put(None)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delivery Bot: Simulação de logística em massa.")
    parser.add_argument("--simulacoes", type=int, default=150, help="Número de simulações.")
    parser.add_argument("--arquivo_saida", type=str, default="simulacao-SmartPlayer-Prioridade.csv", help="Nome do arquivo CSV de saída.")
    parser.add_argument("--seed", type=int, help="Semente para uma única simulação.")
    parser.add_argument('--verbose', action='store_true', help="Imprime logs durante a simulação.")
    parser.add_argument('--timeout', type=int, default=60, help="Tempo máximo em segundos por simulação.")
    args = parser.parse_args()

    TIMEOUT_SECONDS = args.timeout
    file_exists = os.path.exists(args.arquivo_saida)
    
    with open(args.arquivo_saida, 'a', newline='') as csvfile:
        fieldnames = ['Id da RUN', 'Score', 'Deliveres', 'steps', 'seed', 'status']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        num_runs = 1 if args.seed is not None else args.simulacoes
        print(f"Executando {num_runs} simulação(ões). Timeout: {TIMEOUT_SECONDS}s. Saída: '{args.arquivo_saida}'...")

        for i in range(1, num_runs + 1):
            current_seed = args.seed if args.seed is not None else i
            if not args.verbose: print(f"Iniciando simulação {i}/{num_runs} (Seed: {current_seed})...", end='\r')

            result_queue = multiprocessing.Queue()
            process = multiprocessing.Process(target=run_simulation_worker, args=(current_seed, args.verbose, result_queue))
            process.start()
            process.join(timeout=TIMEOUT_SECONDS)

            row_data = {'Id da RUN': i, 'Score': 'N/A', 'Deliveres': 'N/A', 'steps': 'N/A', 'seed': current_seed, 'status': 'timeout'}

            if process.is_alive():
                if args.verbose: print(f"\nALERTA: Simulação {i} (Seed: {current_seed}) excedeu o tempo limite. Abortando.")
                process.terminate(); process.join()
            else:
                try:
                    resultados = result_queue.get_nowait()
                    if resultados:
                        row_data.update({'Score': resultados['score'], 'Deliveres': resultados['deliveries'], 'steps': resultados['steps'], 'status': 'completed'})
                    else:
                        row_data['status'] = 'error'
                except queue.Empty:
                    row_data['status'] = 'crashed'
                    if args.verbose: print(f"\nERRO: Simulação {i} (Seed: {current_seed}) terminou sem retornar resultados.")
            
            writer.writerow(row_data)

    print(f"\n\nConcluído! Os resultados foram salvos em '{args.arquivo_saida}'.")