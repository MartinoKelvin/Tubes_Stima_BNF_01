from typing import Optional, Tuple, List
from game.logic.base import BaseLogic
from game.models import Board, GameObject, Position
from game.util import get_direction

class GreedyBNF(BaseLogic):
    static_goals: List[Position] = []
    static_goal_teleport: Optional[GameObject] = None
    static_temp_goals: Optional[Position] = None

    LOW_TIME_THRESHOLD_MS = 7000
    MIN_DIAMONDS_FOR_LOW_TIME_RETURN = 2
    MIN_DIAMONDS_TO_FLEE_TRESHOLD = 3
    ENEMY_FLEE_DISTANCE = 2
    MIN_DIAMONDS_NEAR_BASE_RETURN = 3
    MAX_OWN_DIAMONDS_FOR_AGGRESSION = 2
    MIN_ENEMY_DIAMONDS_TO_ATTACK = 2

    def __init__(self) -> None:
        self.directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        self.current_direction_index = 0
        self.goal_position: Optional[Position] = None
        self.current_diamond_target_distance = float("inf")
        
        # Atribut ini akan di-update setiap giliran
        self.board_bot: GameObject
        self.board: Board
        self.diamonds: List[GameObject]
        self.other_bots: List[GameObject]
        self.teleporters: List[GameObject]
        self.red_button: Optional[GameObject]


    def _reset_major_goals(self):
        self.static_goals = []
        self.static_goal_teleport = None
        self.goal_position = None
        self.current_diamond_target_distance = float("inf")

    def _reset_temp_goals(self):
        self.static_temp_goals = None

    def _update_board_state(self, board_bot: GameObject, board: Board):
        self.board_bot = board_bot
        self.board = board
        self.diamonds = board.diamonds
        self.all_bots = board.bots # Perlu untuk self.other_bots
        self.other_bots = [b for b in self.all_bots if b.id != self.board_bot.id]
        self.teleporters = [go for go in self.board.game_objects if go.type == "TeleportGameObject"]
        self.red_button = next((go for go in self.board.game_objects if go.type == "DiamondButtonGameObject"), None)

    def next_move(self, board_bot: GameObject, board: Board) -> Tuple[int, int]:
        self._update_board_state(board_bot, board)
        
        props = self.board_bot.properties
        current_pos = self.board_bot.position
        my_base_pos = Position(props.base.y, props.base.x)

        # --- Update status goal berdasarkan posisi saat ini ---
        if current_pos == my_base_pos:
            self._reset_major_goals(); self._reset_temp_goals()
        elif self.static_temp_goals and current_pos == self.static_temp_goals:
            self._reset_temp_goals()
        elif self.static_goal_teleport and current_pos == self.find_other_teleport_pos(self.static_goal_teleport):
            if self.static_goals and self.static_goals[0] == self.static_goal_teleport.position:
                self.static_goals.pop(0)
            self.static_goal_teleport = None 
            if self.static_goals: self.goal_position = self.static_goals[0]
            else: self._reset_major_goals()
        elif self.goal_position and current_pos == self.goal_position: # Mencapai goal utama (non-teleport, non-temp)
            if self.static_goals and self.static_goals[0] == self.goal_position:
                self.static_goals.pop(0)
            if self.static_goals: self.goal_position = self.static_goals[0]
            else: self._reset_major_goals()
        
        # --- Tentukan goal baru jika perlu ---
        # Prioritas: Kondisi kembali ke base
        should_return_to_base = (
            props.diamonds == 5 or
            (props.milliseconds_left < self.LOW_TIME_THRESHOLD_MS and props.diamonds >= self.MIN_DIAMONDS_FOR_LOW_TIME_RETURN) or
            (self.is_threatened(current_pos) and props.diamonds >= self.MIN_DIAMONDS_TO_FLEE_TRESHOLD) or
            (self.is_opportunistic_to_return(my_base_pos, current_pos) and props.diamonds >= self.MIN_DIAMONDS_NEAR_BASE_RETURN)
        )

        if should_return_to_base:
            # Panggil set_goal_to_base jika belum menargetkan base, atau jika path ke base via teleporter perlu re-evaluasi
            # set_goal_to_base akan me-reset goal sebelumnya.
            self.set_goal_to_base(my_base_pos, current_pos)
        
        # Jika goal adalah base (mungkin di-set di atas), pastikan static_goals sinkron jika kosong.
        if self.goal_position == my_base_pos and not self.static_goals:
            self.static_goals = [my_base_pos]
        
        # Agresi atau mencari diamond/button jika tidak kembali ke base
        if not self.goal_position or self.goal_position != my_base_pos:
            if props.diamonds <= self.MAX_OWN_DIAMONDS_FOR_AGGRESSION:
                best_enemy_target = self.find_best_enemy_target(current_pos)
                if best_enemy_target:
                    enemy_dist = self.manhattan_distance(current_pos, best_enemy_target.position)
                    current_goal_dist = self.manhattan_distance(current_pos, self.goal_position) if self.goal_position and self.static_goals else float('inf')
                    
                    if enemy_dist < current_goal_dist or not self.goal_position : # Prioritaskan musuh jika lebih dekat atau tidak ada goal
                        self._reset_major_goals()
                        self.goal_position = best_enemy_target.position
                        self.static_goals = [self.goal_position]

            if not self.static_goals: # Jika tidak ada goal agresif atau goal sebelumnya selesai
                self.find_best_block_strategy()
                if not self.static_goals:
                    self.find_direct_diamond_strategy(current_pos)

                if self.static_goals:
                    self.goal_position = self.static_goals[0]
                    if self.goal_position != my_base_pos: # Hanya set jika bukan base, agar tidak menimpa jarak ke base
                        self.current_diamond_target_distance = self.manhattan_distance(current_pos, self.goal_position)
                elif self.red_button: # Fallback ke red button
                    self._reset_major_goals()
                    self.goal_position = self.red_button.position
                    self.static_goals = [self.red_button.position]

        # --- Logika Penghindaran Rintangan dan Gerakan ---
        effective_goal = self.static_temp_goals if self.static_temp_goals else self.goal_position
        delta_x, delta_y = 0, 0

        if effective_goal:
            if not self.static_temp_goals and self.goal_position: # Hanya cek obstacle jika tidak sedang menjalani temp_goal
                # Hindari teleporter kecuali itu tujuannya
                if not (self.static_goal_teleport and self.static_goal_teleport.position == self.goal_position):
                    self.check_obstacle_on_path('teleporter', current_pos, self.goal_position)
                if props.diamonds == 4: # Hindari diamond merah jika punya 4 diamond biru
                     self.check_obstacle_on_path('redDiamond', current_pos, self.goal_position)
                if self.red_button and self.goal_position != self.red_button.position: # Hindari tombol merah kecuali itu tujuannya
                    self.check_obstacle_on_path('redButton', current_pos, self.goal_position)
            
            effective_goal = self.static_temp_goals if self.static_temp_goals else self.goal_position # Evaluasi ulang effective_goal
            if effective_goal:
                delta_x, delta_y = get_direction(current_pos.x, current_pos.y, effective_goal.x, effective_goal.y)
        
        # Logika roaming atau jika stuck
        if not effective_goal or (delta_x == 0 and delta_y == 0 and current_pos != effective_goal):
            if effective_goal and current_pos != effective_goal: # get_direction gagal untuk goal yang valid
                self._reset_major_goals(); self._reset_temp_goals()
            
            moved_in_roam = False
            for i in range(len(self.directions)):
                next_dir_idx = (self.current_direction_index + i) % len(self.directions)
                dx_try, dy_try = self.directions[next_dir_idx]
                if 0 <= current_pos.x + dx_try < self.board.width and \
                   0 <= current_pos.y + dy_try < self.board.height:
                    delta_x, delta_y = dx_try, dy_try
                    self.current_direction_index = (next_dir_idx + 1) % len(self.directions)
                    moved_in_roam = True
                    break
            if not moved_in_roam: delta_x, delta_y = 0, 0 # Tidak ada gerakan valid
        
        # Jika masih tidak bergerak dan bukan di goal, reset (safety net)
        if delta_x == 0 and delta_y == 0 and self.goal_position and current_pos != self.goal_position:
             self._reset_major_goals(); self._reset_temp_goals()

        return delta_x, delta_y

    def manhattan_distance(self, pos1: Position, pos2: Position) -> int:
        return abs(pos1.x - pos2.x) + abs(pos1.y - pos2.y)

    def set_goal_to_base(self, my_base_pos: Position, current_pos: Position):
        path_to_base = self.find_best_way_to_base(my_base_pos, current_pos)
        self._reset_major_goals() # Penting untuk mereset state goal sebelumnya
        self.static_goals = path_to_base
        
        if not self.static_goals: # Safety net, seharusnya find_best_way_to_base selalu mengembalikan path
            self.static_goals = [my_base_pos]

        self.goal_position = self.static_goals[0]
        
        if len(self.static_goals) > 1 and self.static_goals[0] != my_base_pos: # Berarti langkah pertama adalah teleporter
            for tp_obj in self.teleporters:
                if tp_obj.position == self.static_goals[0]:
                    self.static_goal_teleport = tp_obj
                    break
        # current_diamond_target_distance direset oleh _reset_major_goals()

    def is_threatened(self, current_pos: Position) -> bool: # Sesuai implementasi asli: hanya cek jarak
        for enemy in self.other_bots:
            if self.manhattan_distance(current_pos, enemy.position) <= self.ENEMY_FLEE_DISTANCE:
                return True
        return False

    def is_opportunistic_to_return(self, my_base_pos: Position, current_pos: Position) -> bool:
        if self.board_bot.properties.diamonds == 0: # Tidak ada gunanya kembali jika tidak bawa apa-apa
            return False
        dist_to_base = self.manhattan_distance(current_pos, my_base_pos)
        if self.current_diamond_target_distance != float("inf") and \
           dist_to_base < self.current_diamond_target_distance * 0.75 and dist_to_base > 0:
            return True
        return False

    def find_best_enemy_target(self, current_pos: Position) -> Optional[GameObject]:
        best_target = None
        min_score = float('inf') # Untuk enemy, skor lebih rendah (jarak dekat) lebih baik

        for enemy in self.other_bots:
            if enemy.properties.diamonds >= self.MIN_ENEMY_DIAMONDS_TO_ATTACK:
                dist = self.manhattan_distance(current_pos, enemy.position)
                if dist == 0: continue 
                
                score = dist # Skor sederhana: jarak
                if score < min_score:
                    min_score = score
                    best_target = enemy
        return best_target
    
    def _can_collect_diamond(self, diamond: GameObject) -> bool:
        return not (diamond.properties.points == 2 and self.board_bot.properties.diamonds == 4)

    def find_direct_diamond_strategy(self, current_pos: Position):
        best_direct_diamond_pos: Optional[Position] = None
        min_direct_dist_score = float('inf')
        valid_diamonds = [d for d in self.diamonds if self._can_collect_diamond(d)]

        for diamond in valid_diamonds:
            dist = self.manhattan_distance(current_pos, diamond.position)
            if dist == 0: continue
            score = dist / diamond.properties.points 
            if score < min_direct_dist_score:
                min_direct_dist_score = score
                best_direct_diamond_pos = diamond.position

        best_teleport_path: List[Position] = []
        min_teleport_dist_score = float('inf')
        chosen_teleporter_obj: Optional[GameObject] = None

        if self.teleporters and len(self.teleporters) >= 2:
            nearest_tp_entry, nearest_tp_exit, nearest_tp_obj, dist_to_tp = self.find_nearest_teleporter_data(current_pos)
            if nearest_tp_entry and nearest_tp_exit and nearest_tp_obj:
                for diamond in valid_diamonds:
                    dist_exit_to_diamond = self.manhattan_distance(nearest_tp_exit, diamond.position)
                    total_dist = dist_to_tp + dist_exit_to_diamond
                    if total_dist == 0 and diamond.position != nearest_tp_exit : continue
                    
                    score = total_dist / diamond.properties.points
                    if score < min_teleport_dist_score:
                        min_teleport_dist_score = score
                        best_teleport_path = [nearest_tp_entry, diamond.position]
                        chosen_teleporter_obj = nearest_tp_obj
        
        if best_direct_diamond_pos and min_direct_dist_score <= min_teleport_dist_score:
            self.static_goals = [best_direct_diamond_pos]
            self.static_goal_teleport = None
        elif best_teleport_path and chosen_teleporter_obj:
            self.static_goals = best_teleport_path
            self.static_goal_teleport = chosen_teleporter_obj
        # Jika tidak ada, static_goals dan static_goal_teleport tidak diubah (atau sudah direset sebelumnya)

    def find_best_block_strategy(self):
        self._reset_major_goals() # Selalu reset sebelum mencari strategi block baru
        
        if not self.diamonds: self.static_goals = []; return

        blockH = max(1, self.board.height // 3)
        blockW = max(1, self.board.width // 3)
        block_values = [[0 for _ in range(3)] for _ in range(3)]
        block_diamonds_lists: List[List[List[Position]]] = [[[] for _ in range(3)] for _ in range(3)]
        
        diamonds_needed = 5 - self.board_bot.properties.diamonds
        if diamonds_needed <= 0: self.static_goals = []; return

        valid_diamonds = [d for d in self.diamonds if self._can_collect_diamond(d)]
        if not valid_diamonds: self.static_goals = []; return

        for diamond in valid_diamonds:
            if not (0 <= diamond.position.x < self.board.width and 0 <= diamond.position.y < self.board.height): continue
            block_row = min(diamond.position.y // blockH, 2)
            block_col = min(diamond.position.x // blockW, 2)
            block_values[block_row][block_col] += diamond.properties.points
            block_diamonds_lists[block_row][block_col].append(diamond.position)

        best_score = -1
        best_block_idx = None
        for r in range(3):
            for c in range(3):
                if block_values[r][c] == 0: continue
                current_block_score = block_values[r][c]
                if block_values[r][c] >= diamonds_needed: current_block_score += 10
                if current_block_score > best_score:
                    best_score = current_block_score
                    best_block_idx = (r, c)
        
        if best_block_idx:
            r_best, c_best = best_block_idx
            self.static_goals = sorted(
                block_diamonds_lists[r_best][c_best],
                key=lambda p: self.manhattan_distance(self.board_bot.position, p)
            )
        else:
            self.static_goals = []
            
    def find_nearest_teleporter_data(self, current_pos: Position) -> Tuple[Optional[Position], Optional[Position], Optional[GameObject], int]:
        if not self.teleporters or len(self.teleporters) < 2:
            return None, None, None, float('inf')

        nearest_entry_pos: Optional[Position] = None
        min_dist_to_entry = float('inf')
        selected_tp_obj: Optional[GameObject] = None

        for tp_obj in self.teleporters:
            dist = self.manhattan_distance(current_pos, tp_obj.position)
            if dist < min_dist_to_entry:
                min_dist_to_entry = dist
                nearest_entry_pos = tp_obj.position
                selected_tp_obj = tp_obj
        
        if nearest_entry_pos and selected_tp_obj:
            other_tp_pos = self.find_other_teleport_pos(selected_tp_obj)
            if other_tp_pos:
                 return nearest_entry_pos, other_tp_pos, selected_tp_obj, min_dist_to_entry
        return None, None, None, float('inf')

    def find_other_teleport_pos(self, teleporter_obj: GameObject) -> Optional[Position]:
        for tp in self.teleporters:
            if tp.id != teleporter_obj.id: return tp.position
        return None

    def find_best_way_to_base(self, my_base_pos: Position, current_pos: Position) -> List[Position]:
        dist_direct = self.manhattan_distance(current_pos, my_base_pos)
        if dist_direct == 0: return [my_base_pos] 

        path_via_teleporter: Optional[List[Position]] = None
        dist_via_teleporter = float('inf')

        if self.teleporters and len(self.teleporters) >= 2:
            entry, exit_pos, _, dist_to_entry = self.find_nearest_teleporter_data(current_pos)
            if entry and exit_pos:
                dist_exit_to_base = self.manhattan_distance(exit_pos, my_base_pos)
                dist_via_teleporter = dist_to_entry + dist_exit_to_base 
                path_via_teleporter = [entry, my_base_pos]

        if path_via_teleporter and dist_via_teleporter < dist_direct:
            return path_via_teleporter
        return [my_base_pos]

    def check_obstacle_on_path(self, obj_type_to_avoid: str, current_pos: Position, ultimate_dest_pos: Position):
        if current_pos == ultimate_dest_pos: return

        objects_to_check: List[GameObject] = []
        if obj_type_to_avoid == 'teleporter': objects_to_check = self.teleporters
        elif obj_type_to_avoid == 'redDiamond': objects_to_check = [d for d in self.diamonds if d.properties.points == 2]
        elif obj_type_to_avoid == 'redButton' and self.red_button: objects_to_check = [self.red_button]
        if not objects_to_check: return

        dx_naive, dy_naive = get_direction(current_pos.x, current_pos.y, ultimate_dest_pos.x, ultimate_dest_pos.y)
        if dx_naive == 0 and dy_naive == 0: return 
        next_pos_naive = Position(current_pos.x + dx_naive, current_pos.y + dy_naive)

        is_obstacle_on_naive_path = any(obs.position == next_pos_naive for obs in objects_to_check)
        if not is_obstacle_on_naive_path: return

        detour_options: List[Tuple[int,int]] = []
        if dx_naive != 0: detour_options = [(0, 1), (0, -1)] # Coba vertikal
        elif dy_naive != 0: detour_options = [(1, 0), (-1, 0)] # Coba horizontal
        
        for ddx, ddy in detour_options: # Coba langkah menyamping
            detour_pos = Position(current_pos.x + ddx, current_pos.y + ddy)
            if 0 <= detour_pos.x < self.board.width and 0 <= detour_pos.y < self.board.height and \
               not any(obs.position == detour_pos for obs in objects_to_check):
                self.static_temp_goals = detour_pos; return 

        # Jika menyamping gagal, coba mundur (jika ada arah naif)
        if dx_naive != 0 or dy_naive != 0:
            backward_pos = Position(current_pos.x - dx_naive, current_pos.y - dy_naive)
            if 0 <= backward_pos.x < self.board.width and 0 <= backward_pos.y < self.board.height and \
               not any(obs.position == backward_pos for obs in objects_to_check):
                self.static_temp_goals = backward_pos; return