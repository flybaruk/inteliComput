import random
import heapq
import sys
import argparse
from abc import ABC, abstractmethod
import csv  # Importa a biblioteca para manipulação de CSV
import os   # Importa para verificar se o arquivo já existe
import multiprocessing
import queue # Módulo de fila, para comunicação entre processos

# ==========================
# CLASSES DE PLAYER (sem alterações)
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
        self.LATENESS_PENALTY_MULTIPLIER = 10
        self.OPPORTUNITY_BONUS_WEIGHT = 120

    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def astar(self, start, goal, world_map):
        size = len(world_map)
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
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g
                    fscore[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(oheap, (fscore[neighbor], neighbor))
        return []

    def escolher_alvo(self, world, current_steps):
        if self.cargo > 0:
            best_goal_pos = None
            best_goal_score = -float('inf')
            for goal in world.goals:
                path = self.astar(self.position, goal["pos"], world.map)
                path_len = len(path)
                if not path: continue
                arrival_step = current_steps + path_len
                deadline_step = goal["created_at"] + goal["priority"]
                lateness = max(0, arrival_step - deadline_step)
                score = -lateness * self.LATENESS_PENALTY_MULTIPLIER - path_len
                if score > best_goal_score:
                    best_goal_score = score
                    best_goal_pos = goal["pos"]
            return best_goal_pos
        elif self.cargo == 0 and world.packages and world.goals:
            best_package_pos = None
            best_trip_score = -float('inf')
            for pkg_pos in world.packages:
                path_to_pkg = self.astar(self.position, pkg_pos, world.map)
                if not path_to_pkg: continue
                path_to_pkg_len = len(path_to_pkg)
                for goal in world.goals:
                    path_from_pkg_to_goal = self.astar(pkg_pos, goal["pos"], world.map)
                    if not path_from_pkg_to_goal: continue
                    path_from_pkg_to_goal_len = len(path_from_pkg_to_goal)
                    total_trip_len = path_to_pkg_len + path_from_pkg_to_goal_len
                    arrival_step = current_steps + total_trip_len
                    deadline_step = goal["created_at"] + goal["priority"]
                    lateness = max(0, arrival_step - deadline_step)
                    trip_score = -lateness * self.LATENESS_PENALTY_MULTIPLIER - total_trip_len
                    opportunity_bonus = 0
                    remaining_packages = [p for p in world.packages if p != pkg_pos]
                    if remaining_packages:
                        min_dist_to_next_pkg = min([self.heuristic(goal["pos"], p) for p in remaining_packages])
                        opportunity_bonus = self.OPPORTUNITY_BONUS_WEIGHT / (1 + min_dist_to_next_pkg)
                    final_trip_score = trip_score + opportunity_bonus
                    if final_trip_score > best_trip_score:
                        best_trip_score = final_trip_score
                        best_package_pos = pkg_pos
            return best_package_pos
        return None

# ==========================
# CLASSE WORLD (sem alterações)
# ==========================
class World:
    def __init__(self, seed=None):
        if seed is not None:
            random.seed(seed)
        self.maze_size = 30
        self.map = [[0 for _ in range(self.maze_size)] for _ in range(self.maze_size)]
        self.generate_obstacles()
        self.walls = [(c, r) for r in range(self.maze_size) for c in range(self.maze_size) if self.map[r][c] == 1]
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
# CLASSE MAZE (Lógica do Jogo)
# ==========================
class Maze:
    def __init__(self, seed=None, verbose=False):
        self.world = World(seed)
        self.seed = seed
        self.running = True
        self.score = 0
        self.steps = 0
        self.path = []
        self.num_deliveries = 0
        self.verbose = verbose # Controla se os logs de passo a passo são impressos

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
                    # Se não há alvo, o comportamento correto é simplesmente esperar (tick)
                    # para que o estado do mundo mude (ex: um novo goal apareça).
                    # A lógica de impasse anterior foi removida daqui.
                    self.tick()
                    continue
            
            self.path = self.astar(self.world.player.position, self.current_target)
            if not self.path:
                # Lógica original de reposicionamento (mantida como segurança)
                if self.current_target in self.world.packages:
                    self.world.packages.remove(self.current_target)
                    new_pos = self.world.random_free_cell()
                    self.world.packages.append(new_pos)
                    if self.verbose:
                        print(f"[INFO] Pacote em {self.current_target} inalcançável. Reposicionado em {new_pos}.")
                else:
                    goal = self.get_goal_at(self.current_target)
                    if goal:
                        self.world.goals.remove(goal)
                        goal["pos"] = self.world.random_free_cell()
                        self.world.goals.append(goal)
                        if self.verbose:
                            print(f"[INFO] Goal em {self.current_target} inalcançável. Reposicionado em {goal['pos']}.")
                self.current_target = None
                self.tick()
                continue
            
            for pos in self.path:
                self.world.player.position = pos
                self.tick()

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
                        
                        # 💡 INÍCIO DA CORREÇÃO DEFINITIVA
                        # Se acabamos de usar o último goal e o jogo ainda não terminou,
                        # criamos um novo imediatamente para evitar o impasse.
                        game_is_not_over = self.num_deliveries < self.world.total_items
                        if not self.world.goals and game_is_not_over:
                            if self.verbose:
                                print(f"💡 [INFO] Último goal utilizado. Criando um novo para evitar impasse.")
                            self.world.add_goal(self.steps)
                        # 💡 FIM DA CORREÇÃO DEFINITIVA
            
            if self.verbose:
                print(f"-> Fim do ciclo: Passos: {self.steps}, Pontuação: {self.score}, Carga: {self.world.player.cargo}, Entregas: {self.num_deliveries}")

            self.current_target = None
        
        # Retorna os resultados finais
        return {
            "score": self.score,
            "deliveries": self.num_deliveries,
            "steps": self.steps
        }
# ==========================
# PONTO DE ENTRADA PRINCIPAL (COM TIMEOUT E CABEÇALHOS CORRIGIDOS)
# ==========================

# (A função 'run_simulation_worker' continua a mesma de antes)
def run_simulation_worker(seed, verbose, result_queue):
    """
    Cria e executa uma instância do Maze e coloca o resultado em uma fila.
    """
    try:
        maze = Maze(seed=seed, verbose=verbose)
        resultados = maze.game_loop()
        result_queue.put(resultados)
    except Exception as e:
        print(f"Erro na simulação com seed {seed}: {e}")
        result_queue.put(None)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Delivery Bot: Simulação de logística com saída em CSV e timeout."
    )
    # (Todos os argumentos do 'parser' continuam os mesmos)
    parser.add_argument("--simulacoes", type=int, default=1000, help="Número de simulações a serem executadas.")
    parser.add_argument("--arquivo_saida", type=str, default="simulacao-aleatoria{}.csv".format(DefaultPlayer(position=(0,0)).OPPORTUNITY_BONUS_WEIGHT), help="Nome do arquivo CSV de saída.")
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
        print(f"Executando {num_runs} simulação(ões). Timeout por simulação: {TIMEOUT_SECONDS}s. Resultados em '{args.arquivo_saida}'...")

        for i in range(1, num_runs + 1):
            current_seed = args.seed if args.seed is not None else i
            print(f"Iniciando simulação {i}/{num_runs} (Seed: {current_seed})...", end='\r')

            result_queue = multiprocessing.Queue()
            simulation_process = multiprocessing.Process(
                target=run_simulation_worker, 
                args=(current_seed, args.verbose, result_queue)
            )
            simulation_process.start()
            simulation_process.join(timeout=TIMEOUT_SECONDS)

            if simulation_process.is_alive():
                print(f"\n⚠️  Simulação {i}/{num_runs} (Seed: {current_seed}) excedeu o tempo limite. Abortando.")
                simulation_process.terminate()
                simulation_process.join()

                writer.writerow({
                    'Id da RUN': i,
                    'Score': 'N/A',
                    'Deliveres': 'N/A',
                    'steps': 'N/A',
                    'seed': current_seed,
                    'status': 'timeout'
                })
            else:
                try:
                    resultados = result_queue.get_nowait()
                    if resultados:
                        writer.writerow({
                            'Id da RUN': i,
                            'Score': resultados['score'],
                            'Deliveres': resultados['deliveries'],
                            'steps': resultados['steps'],
                            'seed': current_seed,
                            'status': 'completed'
                        })
                    else:
                         writer.writerow({
                            'Id da RUN': i, 'Score': 'N/A', 'Deliveres': 'N/A', 
                            'steps': 'N/A', 'seed': current_seed, 'status': 'error'
                        })
                except queue.Empty:
                    print(f"\n❌ Erro: Simulação {i}/{num_runs} (Seed: {current_seed}) terminou sem retornar resultados.")
                    writer.writerow({
                        'Id da RUN': i, 'Score': 'N/A', 'Deliveres': 'N/A', 
                        'steps': 'N/A', 'seed': current_seed, 'status': 'crashed'
                    })

    print(f"\n\nConcluído! Os resultados de {num_runs} simulação(ões) foram salvos em '{args.arquivo_saida}'.")