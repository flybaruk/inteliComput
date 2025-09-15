import random
import heapq
import sys
import argparse
from abc import ABC, abstractmethod
import csv
import os
import multiprocessing
import queue

# ==========================
# CLASSES DE PLAYER (Sua lógica inteligente está intacta)
# ==========================
class BasePlayer(ABC):
    def __init__(self, position):
        self.position = position
        self.cargo = 0

    @abstractmethod
    def escolher_alvo(self, world, current_steps):
        pass

class DefaultPlayer(BasePlayer):
    def __init__(self, position):
        super().__init__(position)
        self.max_cargo = 2
        self.LATENESS_PENALTY_MULTIPLIER = 10
        self.OPPORTUNITY_BONUS_WEIGHT = 120

    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def astar(self, start, goal, world_map):
        size = len(world_map)
        neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        close_set, came_from = set(), {}
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
                return data
            close_set.add(current)
            for dx, dy in neighbors:
                neighbor = (current[0] + dx, current[1] + dy)
                if not (0 <= neighbor[0] < size and 0 <= neighbor[1] < size and world_map[neighbor[1]][neighbor[0]] == 0):
                    continue
                tentative_g = gscore[current] + 1
                if neighbor in close_set and tentative_g >= gscore.get(neighbor, 0):
                    continue
                if tentative_g < gscore.get(neighbor, float('inf')) or neighbor not in [i[1] for i in oheap]:
                    came_from[neighbor], gscore[neighbor] = current, tentative_g
                    fscore[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(oheap, (fscore[neighbor], neighbor))
        return []

    def escolher_alvo(self, world, current_steps):
        if self.cargo == 0:
            best_package_pos, best_trip_score = None, -float('inf')
            if not world.packages or not world.goals: return None
            for pkg_pos in world.packages:
                path_to_pkg = self.astar(self.position, pkg_pos, world.map)
                if not path_to_pkg: continue
                for goal in world.goals:
                    path_from_pkg_to_goal = self.astar(pkg_pos, goal["pos"], world.map)
                    if not path_from_pkg_to_goal: continue
                    total_trip_len = len(path_to_pkg) + len(path_from_pkg_to_goal)
                    lateness = max(0, (current_steps + total_trip_len) - (goal["created_at"] + goal["priority"]))
                    trip_score = -lateness * self.LATENESS_PENALTY_MULTIPLIER - total_trip_len
                    opportunity_bonus = 0
                    remaining_packages = [p for p in world.packages if p != pkg_pos]
                    if remaining_packages:
                        min_dist = min([self.heuristic(goal["pos"], p) for p in remaining_packages])
                        opportunity_bonus = self.OPPORTUNITY_BONUS_WEIGHT / (1 + min_dist)
                    if (trip_score + opportunity_bonus) > best_trip_score:
                        best_trip_score = trip_score + opportunity_bonus
                        best_package_pos = pkg_pos
            return best_package_pos
        elif self.cargo > 0:
            best_target, best_score = None, -float('inf')
            for goal in world.goals:
                path = self.astar(self.position, goal["pos"], world.map)
                if not path: continue
                lateness = max(0, (current_steps + len(path)) - (goal["created_at"] + goal["priority"]))
                score = -lateness * self.LATENESS_PENALTY_MULTIPLIER - len(path)
                if score > best_score:
                    best_score, best_target = score, goal["pos"]
            if self.cargo < self.max_cargo and world.packages:
                for next_pkg_pos in world.packages:
                    path_to_pkg = self.astar(self.position, next_pkg_pos, world.map)
                    if not path_to_pkg: continue
                    for goal in world.goals:
                        path_from_pkg = self.astar(next_pkg_pos, goal["pos"], world.map)
                        if not path_from_pkg: continue
                        total_len = len(path_to_pkg) + len(path_from_pkg)
                        lateness = max(0, (current_steps + total_len) - (goal["created_at"] + goal["priority"]))
                        score = -lateness * self.LATENESS_PENALTY_MULTIPLIER - total_len
                        opportunity_bonus = 0
                        remaining_packages = [p for p in world.packages if p != next_pkg_pos]
                        if remaining_packages:
                             min_dist = min([self.heuristic(goal["pos"], p) for p in remaining_packages])
                             opportunity_bonus = self.OPPORTUNITY_BONUS_WEIGHT / (1 + min_dist)
                        if (score + opportunity_bonus) > best_score:
                            best_score = score + opportunity_bonus
                            best_target = next_pkg_pos
            return best_target
        return None

# ==========================
# CLASSE WORLD (MUNDO) - Versão Headless
# ==========================
class World:
    def __init__(self, seed=None):
        if seed is not None:
            random.seed(seed)
        self.maze_size = 30
        self.map = [[0 for _ in range(self.maze_size)] for _ in range(self.maze_size)]
        self.generate_obstacles()
        self.walls = [(c, r) for r, row in enumerate(self.map) for c, val in enumerate(row) if val == 1]
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
                return DefaultPlayer([x, y])

    def random_free_cell(self):
        while True:
            x, y = random.randint(0, self.maze_size - 1), random.randint(0, self.maze_size - 1)
            occupied = (self.map[y][x] == 1 or [x, y] in self.packages or [x, y] == self.player.position or any(g["pos"] == [x, y] for g in self.goals))
            if not occupied: return [x, y]

    def add_goal(self, created_at_step):
        self.goals.append({"pos": self.random_free_cell(), "priority": random.randint(40, 110), "created_at": created_at_step})

# ==========================
# CLASSE MAZE (Lógica do Jogo) - Versão Headless
# ==========================
class Maze:
    def __init__(self, seed=None, verbose=False):
        self.world = World(seed=seed)
        self.seed = seed
        self.verbose = verbose
        self.running = True
        self.score = 0
        self.steps = 0
        self.path = []
        self.num_deliveries = 0
        self.world.add_goal(created_at_step=0)
        self.spawn_intervals = [random.randint(2, 5)] + [random.randint(5, 10)] + [random.randint(10, 15) for _ in range(3)]
        self.next_spawn_step = self.spawn_intervals.pop(0)
        self.current_target = None

    def astar(self, start, goal):
        # A* já está no player, podemos chamar diretamente
        return self.world.player.astar(start, goal, self.world.map)

    def maybe_spawn_goal(self):
        if self.next_spawn_step is not None and self.steps >= self.next_spawn_step:
            self.world.add_goal(created_at_step=self.steps)
            self.next_spawn_step += self.spawn_intervals.pop(0) if self.spawn_intervals else float('inf')

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
            
            self.path = self.astar(self.world.player.position, self.current_target)
            
            if not self.path:
                if self.verbose: print(f"Alvo {self.current_target} inacessível.")
                if self.current_target in self.world.packages:
                    self.world.packages.remove(self.current_target)
                goal = self.get_goal_at(self.current_target)
                if goal: self.world.goals.remove(goal)
                self.current_target = None
                continue
            
            # Movimentação passo a passo
            for pos in self.path:
                self.world.player.position = pos
                self.tick()

            # Ações ao chegar no alvo
            if self.world.player.position == self.current_target:
                if self.current_target in self.world.packages:
                    self.world.player.cargo += 1
                    self.world.packages.remove(self.current_target)
                else:
                    goal = self.get_goal_at(self.current_target)
                    if goal and self.world.player.cargo > 0:
                        self.world.player.cargo -= 1
                        self.num_deliveries += 1
                        self.world.goals.remove(goal)
                        self.score += 50
                        # Se for o último goal, cria um novo para evitar impasse
                        if not self.world.goals and self.num_deliveries < self.world.total_items:
                           self.world.add_goal(self.steps)
            
            self.current_target = None

        return {"score": self.score, "deliveries": self.num_deliveries, "steps": self.steps}

# ==========================
# PONTO DE ENTRADA PRINCIPAL (Focado em Épocas)
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
    parser.add_argument("--simulacoes", type=int, default=1000, help="Número de simulações a serem executadas.")
    parser.add_argument("--arquivo_saida", type=str, default="simulacao-M.csv", help="Nome do arquivo CSV de saída.")
    parser.add_argument("--seed", type=int, default=None, help="Semente para o gerador de números aleatórios.")
    parser.add_argument('--verbose', action='store_true', help="Imprime logs de status durante a simulação.")
    parser.add_argument('--timeout', type=int, default=15, help="Tempo máximo em segundos que cada simulação pode durar.")
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
            print(f"Iniciando simulação {i}/{num_runs} (Seed: {current_seed})...", end='\r')

            result_queue = multiprocessing.Queue()
            simulation_process = multiprocessing.Process(target=run_simulation_worker, args=(current_seed, args.verbose, result_queue))
            simulation_process.start()
            simulation_process.join(timeout=TIMEOUT_SECONDS)

            if simulation_process.is_alive():
                print(f"\n⚠️  Simulação {i}/{num_runs} (Seed: {current_seed}) excedeu o tempo limite. Abortando.")
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