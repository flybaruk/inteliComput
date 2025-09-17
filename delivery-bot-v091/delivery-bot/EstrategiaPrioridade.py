import pygame
import random
import heapq
import sys
import argparse
from abc import ABC, abstractmethod
import math

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

class SmartPlayer(BasePlayer):
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

        # Avalia a melhor jogada de ENTREGA possível AGORA
        if self.cargo > 0 and world.goals:
            goal_obj, score = self._evaluate_best_goal(world, current_steps, maze_solver)
            if goal_obj:
                best_delivery_score = score
                best_delivery_target = goal_obj["pos"]

        # Avalia a melhor jogada de COLETA possível AGORA
        if self.cargo < self.max_cargo and world.packages and world.goals:
            pkg_target, score = self._evaluate_best_package_holistic(world, current_steps, maze_solver)
            if pkg_target:
                best_collection_score = score
                best_collection_target = pkg_target
        
        if best_delivery_score >= best_collection_score:
            return best_delivery_target
        else:
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
# CLASSE WORLD
# ==========================
class World:
    def __init__(self, seed=None):
        if seed is not None: random.seed(seed)
        self.maze_size = 30
        self.width = 600
        self.height = 600
        self.block_size = self.width // self.maze_size
        self.map = [[0 for _ in range(self.maze_size)] for _ in range(self.maze_size)]
        self.generate_obstacles()
        self.walls = [(c, r) for r in range(self.maze_size) for c in range(self.maze_size) if self.map[r][c] == 1]
        self.total_items = 6
        self.packages = []
        while len(self.packages) < self.total_items + 4:
            x, y = random.randint(0, self.maze_size - 1), random.randint(0, self.maze_size - 1)
            if self.map[y][x] == 0 and (x, y) not in self.packages:
                self.packages.append((x, y))
        self.goals = []
        self.player = self.generate_player()
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Delivery Bot")
        try:
            self.package_image = pygame.image.load("images/cargo.png")
            self.package_image = pygame.transform.scale(self.package_image, (self.block_size, self.block_size))
            self.goal_image = pygame.image.load("images/operator.png")
            self.goal_image = pygame.transform.scale(self.goal_image, (self.block_size, self.block_size))
            self.use_images = True
        except: self.use_images = False
        self.wall_color = (100, 100, 100)
        self.ground_color = (255, 255, 255)
        self.player_color = (0, 255, 0)
        self.path_color = (200, 200, 0)
        self.package_color = (150, 50, 50)
        self.goal_color = (50, 50, 150)

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
            if self.map[y][x] == 0 and (x, y) not in self.packages:
                return SmartPlayer((x, y))

    def random_free_cell(self):
        while True:
            x, y = random.randint(0, self.maze_size - 1), random.randint(0, self.maze_size - 1)
            occupied = (self.map[y][x] == 1 or (x, y) in self.packages or (x, y) == self.player.position or any(g["pos"] == (x, y) for g in self.goals))
            if not occupied: return (x, y)

    def add_goal(self, created_at_step):
        pos = self.random_free_cell()
        priority = random.randint(40, 110)
        self.goals.append({"pos": pos, "priority": priority, "created_at": created_at_step})

    def draw_world(self, path=None):
        self.screen.fill(self.ground_color)
        for (x, y) in self.walls:
            rect = pygame.Rect(x * self.block_size, y * self.block_size, self.block_size, self.block_size)
            pygame.draw.rect(self.screen, self.wall_color, rect)
        for pkg in self.packages:
            if self.use_images: self.screen.blit(self.package_image, (pkg[0] * self.block_size, pkg[1] * self.block_size))
            else: pygame.draw.rect(self.screen, self.package_color, pygame.Rect(pkg[0]*self.block_size+4, pkg[1]*self.block_size+4, self.block_size-8, self.block_size-8))
        for goal in self.goals:
            if self.use_images: self.screen.blit(self.goal_image, (goal["pos"][0] * self.block_size, goal["pos"][1] * self.block_size))
            else: pygame.draw.rect(self.screen, self.goal_color, pygame.Rect(goal["pos"][0]*self.block_size+4, goal["pos"][1]*self.block_size+4, self.block_size-8, self.block_size-8))
        if path:
            for pos in path:
                rect = pygame.Rect(pos[0] * self.block_size + self.block_size // 4, pos[1] * self.block_size + self.block_size // 4, self.block_size // 2, self.block_size // 2)
                pygame.draw.rect(self.screen, self.path_color, rect)
        x, y = self.player.position
        rect = pygame.Rect(x * self.block_size, y * self.block_size, self.block_size, self.block_size)
        pygame.draw.rect(self.screen, self.player_color, rect)
        pygame.display.flip()

# ==========================
# CLASSE MAZE: A* e Loop
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
        self.next_spawn_step = self.spawn_intervals.pop(0) if self.spawn_intervals else None
        self.current_target = None

    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def astar(self, start, goal):
        start_t, goal_t = tuple(start), tuple(goal)
        if start_t == goal_t: return []
        maze, size = self.world.map, self.world.maze_size
        neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        close_set, came_from, gscore = set(), {}, {start_t: 0}
        fscore = {start_t: self.heuristic(start_t, goal_t)}
        oheap = [(fscore[start_t], start_t)]
        while oheap:
            current = heapq.heappop(oheap)[1]
            if current == goal_t:
                data = []
                while current in came_from:
                    data.append(list(current))
                    current = came_from[current]
                return data[::-1]
            close_set.add(current)
            for dx, dy in neighbors:
                neighbor = (current[0] + dx, current[1] + dy)
                tentative_g = gscore[current] + 1
                if not (0 <= neighbor[0] < size and 0 <= neighbor[1] < size and maze[neighbor[1]][neighbor[0]] == 0): continue
                if neighbor in close_set and tentative_g >= gscore.get(neighbor, float('inf')): continue
                if tentative_g < gscore.get(neighbor, float('inf')) or not any(neighbor == i[1] for i in oheap):
                    came_from[neighbor], gscore[neighbor] = current, tentative_g
                    fscore[neighbor] = tentative_g + self.heuristic(neighbor, goal_t)
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

    def idle_tick(self):
        self.steps += 1

        # Custo base por movimento
        self.score -= 1

        # Penalidade por metas atrasadas
        self.score -= self.delayed_goals_penalty()
        self.maybe_spawn_goal()
        self.world.draw_world(self.path)
        pygame.time.wait(self.delay)
        for event in pygame.event.get():
            if event.type == pygame.QUIT: self.running = False

    def game_loop(self):
        while self.running:
            if self.num_deliveries >= self.world.total_items: self.running = False; break
            self.maybe_spawn_goal()

            if self.current_target is None:
                target = self.world.player.escolher_alvo(self.world, self.steps, self)
                if target is None:
                    self.idle_tick()
                    continue
                self.current_target = tuple(target)

            self.path = self.astar(self.world.player.position, self.current_target)
            if not self.path and self.world.player.position != self.current_target:
                print(f"Alvo {self.current_target} inalcançável. Descartando alvo.")
                self.current_target = None; self.path = []
                continue

            for pos in self.path:
                self.world.player.position = tuple(pos)
                self.steps += 1

                # Custo base por movimento
                self.score -= 1

                # Penalidade por metas atrasadas
                self.score -= self.delayed_goals_penalty()

                self.world.draw_world(self.path)
                pygame.time.wait(self.delay)

                for event in pygame.event.get():
                    if event.type == pygame.QUIT: self.running = False
                if not self.running: break
            if not self.running: break

            if self.world.player.position == self.current_target:
                if self.current_target in self.world.packages:
                    self.world.player.cargo += 1
                    self.world.packages.remove(self.current_target)
                    print(f"Pacote coletado em {self.current_target}. Carga: {self.world.player.cargo}")
                else:
                    goal = self.get_goal_at(self.current_target)
                    if goal and self.world.player.cargo > 0:
                        self.world.player.cargo -= 1
                        self.num_deliveries += 1
                        self.world.goals.remove(goal)
                        # ### PONTUAÇÃO CORRIGIDA ###
                        self.score += 50 # Recompensa por entrega
                        print(f"Pacote entregue em {self.current_target} | Carga: {self.world.player.cargo}")
                self.current_target = None
        
        print(f"\nFim de jogo!\nPassos totais: {self.steps}\nPontuação final: {self.score}")
        pygame.quit()

# ==========================
# MAIN
# ==========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None, help="Seed para o mundo")
    args = parser.parse_args()
    game = Maze(seed=args.seed)
    game.game_loop()