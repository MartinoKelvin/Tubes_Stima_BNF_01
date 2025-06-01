from typing import Optional, Tuple, List

from game.logic.base import BaseLogic
from game.models import Board, GameObject, Position
from game.util import get_direction, position_equals

class TackleLogic(BaseLogic):
    HIGH_DIAMOND_THRESHOLD_FOR_RETURN = 4 
    MIN_DIAMONDS_FOR_PRIORITY_TARGET = 1  

    def __init__(self) -> None:
        self.directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        self.current_roaming_direction_index = 0
        
        self.board_bot: Optional[GameObject] = None
        self.enemies: List[GameObject] = []
        self.own_base_pos: Optional[Position] = None
        
        self.targeted_enemy: Optional[GameObject] = None
        # self.goal_position tidak lagi menjadi state instance utama, 
        # akan ditentukan secara dinamis di next_move
        # self.goal_position: Optional[Position] = None 

    def _update_internal_state(self, board_bot: GameObject, board: Board):
        self.board_bot = board_bot
        self.own_base_pos = board_bot.properties.base
        
        self.enemies = []
        for obj in board.game_objects:
            if obj.type == "BotGameObject" and obj.id != board_bot.id:
                # Pastikan musuh punya 'properties' dan 'position' sebelum diakses
                if hasattr(obj, 'properties') and hasattr(obj, 'position'):
                    self.enemies.append(obj)
                # else:
                #     print(f"Warning: Bot-like object {obj.id} is missing properties or position.")


    def _find_target_enemy(self) -> Optional[GameObject]:
        if not self.enemies:
            return None

        valid_enemies = [e for e in self.enemies if hasattr(e, 'properties') and hasattr(e.properties, 'diamonds')]

        priority_targets = []
        for enemy in valid_enemies:
            if enemy.properties.diamonds >= self.MIN_DIAMONDS_FOR_PRIORITY_TARGET:
                priority_targets.append(enemy)
        
        target_pool = priority_targets if priority_targets else valid_enemies
        if not target_pool:
            return None

        # Pilih target terdekat dari pool yang dipilih
        target_pool.sort(key=lambda e: self._manhattan_distance(self.board_bot.position, e.position))
        return target_pool[0]

    def _manhattan_distance(self, pos1: Position, pos2: Position) -> int:
        return abs(pos1.x - pos2.x) + abs(pos1.y - pos2.y)

    def next_move(self, board_bot: GameObject, board: Board) -> Tuple[int, int]:
        self._update_internal_state(board_bot, board)
        
        current_pos = self.board_bot.position
        props = self.board_bot.properties
        
        final_goal_position: Optional[Position] = None # Tujuan untuk langkah ini

        # 1. Prioritas: Kembali ke base jika membawa banyak diamond
        if props.diamonds >= self.HIGH_DIAMOND_THRESHOLD_FOR_RETURN:
            final_goal_position = self.own_base_pos
            self.targeted_enemy = None 
        else:
            # 2. Logika Penargetan Musuh
            # Cek apakah target saat ini masih valid dan ada di daftar musuh saat ini
            if self.targeted_enemy:
                current_target_info = next((e for e in self.enemies if e.id == self.targeted_enemy.id), None)
                if current_target_info:
                    self.targeted_enemy = current_target_info # Update dengan info terbaru (posisi)
                else:
                    self.targeted_enemy = None # Target hilang
            
            if not self.targeted_enemy:
                self.targeted_enemy = self._find_target_enemy()

            if self.targeted_enemy:
                # Pastikan target_enemy memiliki atribut position
                if hasattr(self.targeted_enemy, 'position'):
                    final_goal_position = self.targeted_enemy.position
                else:
                    # Jika target tidak punya posisi (seharusnya tidak terjadi), fallback
                    self.targeted_enemy = None # Anggap target tidak valid
                    # final_goal_position tetap None, akan memicu roaming
            # else: final_goal_position tetap None, akan memicu roaming
        
        # 3. Kalkulasi Pergerakan
        delta_x, delta_y = 0, 0

        if final_goal_position:
            # Jika sudah sampai di tujuan (base atau posisi musuh)
            if position_equals(current_pos, final_goal_position):
                # Jika tujuannya adalah base sendiri, diam.
                if final_goal_position == self.own_base_pos:
                    delta_x, delta_y = 0, 0
                # Jika tujuannya adalah posisi musuh (artinya sudah di-tackle),
                # di tick berikutnya, self.targeted_enemy akan dievaluasi ulang.
                # Untuk tick ini, kita bisa memilih untuk diam atau roam sedikit.
                # Diam (0,0) adalah pilihan aman setelah tackle.
                else: # Sudah di posisi musuh
                    delta_x, delta_y = 0, 0 
                    # Pertimbangkan untuk mereset self.targeted_enemy di sini
                    # agar langsung mencari target baru di tick berikutnya jika musuh
                    # tidak langsung hilang setelah ditackle.
                    # self.targeted_enemy = None
            else:
                delta_x, delta_y = get_direction(
                    current_pos.x,
                    current_pos.y,
                    final_goal_position.x,
                    final_goal_position.y,
                )
        else: # Tidak ada final_goal_position (roaming)
            # Pastikan bot tidak mencoba bergerak jika tidak ada arah yang valid
            # (misalnya, jika self.directions kosong, meskipun seharusnya tidak)
            if self.directions:
                delta = self.directions[self.current_roaming_direction_index]
                delta_x, delta_y = delta[0], delta[1]
                self.current_roaming_direction_index = \
                    (self.current_roaming_direction_index + 1) % len(self.directions)
            else:
                delta_x, delta_y = 0, 0 # Diam jika tidak ada arah roaming

        # Validasi akhir sederhana (opsional, tergantung game engine)
        # Jika game engine tidak menangani batas peta dengan baik, Anda bisa menambahkan cek di sini.
        # next_x, next_y = current_pos.x + delta_x, current_pos.y + delta_y
        # if not (0 <= next_x < board.width and 0 <= next_y < board.height):
        # delta_x, delta_y = 0,0 # Jangan bergerak jika akan keluar peta

        return delta_x, delta_y